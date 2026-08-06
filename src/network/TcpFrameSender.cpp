#include "network/TcpFrameSender.h"

#include "common/BlockingQueue.h"
#include "common/FrameData.h"
#include "network/TcpProtocol.h"
#include "utils/FileLogger.h"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace egcinet::network {

namespace {

struct PendingFrame {
    std::uint64_t frameId = 0;
    std::uint64_t timestampUs = 0;
    FrameCost cost;
    cv::Mat image;
};

std::uint32_t millisecondsToMicroseconds(double milliseconds) noexcept {
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0) {
        return 0;
    }

    constexpr double maximum =
        static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    return static_cast<std::uint32_t>(
        std::min(maximum, std::round(milliseconds * 1000.0)));
}

std::uint32_t saturateCounter(std::uint64_t value) noexcept {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        value, std::numeric_limits<std::uint32_t>::max()));
}

void writeU16(std::vector<std::uint8_t>& output, std::size_t offset, std::uint16_t value) {
    output[offset] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    output[offset + 1] = static_cast<std::uint8_t>(value & 0xFFU);
}

void writeU32(std::vector<std::uint8_t>& output, std::size_t offset, std::uint32_t value) {
    output[offset] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
    output[offset + 1] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
    output[offset + 2] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    output[offset + 3] = static_cast<std::uint8_t>(value & 0xFFU);
}

void writeU64(std::vector<std::uint8_t>& output, std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        const unsigned int shift = static_cast<unsigned int>((7U - index) * 8U);
        output[offset + index] = static_cast<std::uint8_t>((value >> shift) & 0xFFU);
    }
}

std::vector<std::uint8_t> buildHeader(
    const PendingFrame& frame,
    std::size_t imageBytes,
    std::uint64_t droppedFrames,
    std::uint64_t sentFrames
) {
    using namespace protocol;

    // 显式按偏移序列化，不直接发送 struct，规避编译器填充和主机字节序差异。
    std::vector<std::uint8_t> header(kHeaderBytes, 0);
    std::copy(kMagic, kMagic + sizeof(kMagic), header.begin());
    writeU16(header, kVersionOffset, kVersion);
    writeU16(header, kHeaderBytesOffset, kHeaderBytes);
    writeU32(header, kImageBytesOffset, static_cast<std::uint32_t>(imageBytes));
    writeU64(header, kFrameIdOffset, frame.frameId);
    writeU64(header, kTimestampUsOffset, frame.timestampUs);
    writeU32(header, kImageWidthOffset, static_cast<std::uint32_t>(frame.image.cols));
    writeU32(header, kImageHeightOffset, static_cast<std::uint32_t>(frame.image.rows));
    writeU32(header, kImageEncodingOffset, kJpegEncoding);
    writeU32(header, kAcquireUsOffset, millisecondsToMicroseconds(frame.cost.acquire_ms));
    writeU32(header, kH2dUsOffset, millisecondsToMicroseconds(frame.cost.h2d_ms));
    writeU32(
        header, kPreprocessUsOffset, millisecondsToMicroseconds(frame.cost.preprocess_ms));
    writeU32(header, kInferUsOffset, millisecondsToMicroseconds(frame.cost.infer_ms));
    writeU32(
        header, kPostprocessUsOffset, millisecondsToMicroseconds(frame.cost.postprocess_ms));
    writeU32(
        header, kVisualizeUsOffset, millisecondsToMicroseconds(frame.cost.visualize_ms));
    writeU32(header, kD2hUsOffset, millisecondsToMicroseconds(frame.cost.d2h_ms));
    writeU32(header, kTotalUsOffset, millisecondsToMicroseconds(frame.cost.total_ms));
    writeU32(header, kDroppedFramesOffset, saturateCounter(droppedFrames));
    writeU32(header, kSentFramesOffset, saturateCounter(sentFrames));
    writeU32(header, kReservedOffset, 0);
    return header;
}

} // namespace

class TcpFrameSender::Impl {
public:
    explicit Impl(TcpSenderConfig config)
        : config_(std::move(config)),
          queue_(config_.queueSize == 0 ? 1 : config_.queueSize) {
    }

    ~Impl() {
        stop();
    }

    bool start() {
        if (running_.exchange(true, std::memory_order_acq_rel)) {
            return true;
        }
        if (config_.host.empty() || config_.port == 0 ||
            config_.jpegQuality < 1 || config_.jpegQuality > 100 ||
            config_.connectTimeoutMs <= 0 || config_.reconnectIntervalMs <= 0) {
            running_.store(false, std::memory_order_release);
            utils::FileLogger::instance().error("[TCP] invalid sender configuration");
            return false;
        }

        queue_.reset();
        sentFrames_.store(0, std::memory_order_release);
        droppedFrames_.store(0, std::memory_order_release);
        try {
            worker_ = std::thread(&Impl::sendLoop, this);
        } catch (const std::exception& exception) {
            running_.store(false, std::memory_order_release);
            utils::FileLogger::instance().error(
                std::string("[TCP] failed to start sender thread: ") + exception.what());
            return false;
        }

        std::ostringstream message;
        message << "[TCP] sender started: " << config_.host << ':' << config_.port
                << ", queue=" << queue_.capacity()
                << ", jpeg_quality=" << config_.jpegQuality;
        utils::FileLogger::instance().info(message.str());
        return true;
    }

    void stop() noexcept {
        if (!running_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        queue_.stop();
        reconnectCv_.notify_all();
        const int socket = socketFd_.load(std::memory_order_acquire);
        if (socket >= 0) {
            ::shutdown(socket, SHUT_RDWR);
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        closeSocket();
    }

    bool enqueue(const FrameData& frame) noexcept {
        if (!running_.load(std::memory_order_acquire) ||
            frame.visualizedImage.empty() || frame.visualizedImage.type() != CV_8UC3) {
            return false;
        }

        try {
            PendingFrame pending;
            pending.frameId = frame.frameId;
            pending.timestampUs = frame.timestamp_ms <= 0.0
                ? 0
                : static_cast<std::uint64_t>(frame.timestamp_ms * 1000.0);
            pending.cost = frame.cost;
            // pipeline 随后会回收 slot，因此队列必须拥有独立图像数据。
            pending.image = frame.visualizedImage.clone();
            if (!queue_.tryPush(std::move(pending))) {
                droppedFrames_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            return true;
        } catch (...) {
            droppedFrames_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    bool isRunning() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    bool isConnected() const noexcept {
        return connected_.load(std::memory_order_acquire);
    }

    std::uint64_t sentFrames() const noexcept {
        return sentFrames_.load(std::memory_order_acquire);
    }

    std::uint64_t droppedFrames() const noexcept {
        return droppedFrames_.load(std::memory_order_acquire);
    }

private:
    bool connectSocket() {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        addrinfo* addresses = nullptr;
        const std::string service = std::to_string(config_.port);
        const int lookupResult = ::getaddrinfo(
            config_.host.c_str(), service.c_str(), &hints, &addresses);
        if (lookupResult != 0) {
            return false;
        }

        bool connected = false;
        for (addrinfo* address = addresses;
             address != nullptr && running_.load(std::memory_order_acquire);
             address = address->ai_next) {
            const int candidate = ::socket(
                address->ai_family, address->ai_socktype, address->ai_protocol);
            if (candidate < 0) {
                continue;
            }

            const int flags = ::fcntl(candidate, F_GETFL, 0);
            if (flags < 0 || ::fcntl(candidate, F_SETFL, flags | O_NONBLOCK) < 0) {
                ::close(candidate);
                continue;
            }

            int result = ::connect(candidate, address->ai_addr, address->ai_addrlen);
            if (result < 0 && errno == EINPROGRESS) {
                pollfd descriptor{candidate, POLLOUT, 0};
                result = ::poll(&descriptor, 1, config_.connectTimeoutMs);
                if (result > 0) {
                    int socketError = 0;
                    socklen_t errorBytes = sizeof(socketError);
                    result = ::getsockopt(
                        candidate, SOL_SOCKET, SO_ERROR, &socketError, &errorBytes);
                    if (result == 0 && socketError == 0) {
                        result = 1;
                    } else {
                        result = -1;
                    }
                }
            } else if (result == 0) {
                result = 1;
            }

            if (result > 0) {
                const int enabled = 1;
                ::setsockopt(candidate, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
                socketFd_.store(candidate, std::memory_order_release);
                connected_.store(true, std::memory_order_release);
                connected = true;
                break;
            }
            ::close(candidate);
        }

        ::freeaddrinfo(addresses);
        if (connected) {
            utils::FileLogger::instance().info(
                "[TCP] connected to " + config_.host + ':' + std::to_string(config_.port));
        }
        return connected;
    }

    void closeSocket() noexcept {
        connected_.store(false, std::memory_order_release);
        const int socket = socketFd_.exchange(-1, std::memory_order_acq_rel);
        if (socket >= 0) {
            ::shutdown(socket, SHUT_RDWR);
            ::close(socket);
        }
    }

    bool waitUntilConnected(PendingFrame& frame) {
        while (running_.load(std::memory_order_acquire)) {
            PendingFrame newer;
            // 断线期间只保留最新帧；恢复连接后界面立即显示当前状态。
            while (queue_.tryPop(newer)) {
                frame = std::move(newer);
                droppedFrames_.fetch_add(1, std::memory_order_relaxed);
            }

            if (isConnected() || connectSocket()) {
                return true;
            }

            std::unique_lock<std::mutex> lock(reconnectMutex_);
            reconnectCv_.wait_for(
                lock,
                std::chrono::milliseconds(config_.reconnectIntervalMs),
                [this]() { return !running_.load(std::memory_order_acquire); });
        }
        return false;
    }

    bool sendAll(const std::uint8_t* data, std::size_t bytes) {
        // send 允许短写；循环直到完整发送，才能维持接收端的帧边界。
        std::size_t sent = 0;
        while (sent < bytes && running_.load(std::memory_order_acquire)) {
            const int socket = socketFd_.load(std::memory_order_acquire);
            if (socket < 0) {
                return false;
            }

            const ssize_t result = ::send(
                socket, data + sent, bytes - sent, MSG_NOSIGNAL);
            if (result > 0) {
                sent += static_cast<std::size_t>(result);
                continue;
            }
            if (result < 0 && errno == EINTR) {
                continue;
            }
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd descriptor{socket, POLLOUT, 0};
                const int pollResult = ::poll(&descriptor, 1, 500);
                if (pollResult > 0 && (descriptor.revents & POLLOUT) != 0) {
                    continue;
                }
            }
            return false;
        }
        return sent == bytes;
    }

    bool sendFrame(const PendingFrame& frame) {
        std::vector<unsigned char> jpeg;
        const std::vector<int> parameters{cv::IMWRITE_JPEG_QUALITY, config_.jpegQuality};
        if (!cv::imencode(".jpg", frame.image, jpeg, parameters) ||
            jpeg.empty() || jpeg.size() > protocol::kMaximumImageBytes) {
            return false;
        }

        const std::uint64_t sent = sentFrames_.load(std::memory_order_relaxed);
        const std::vector<std::uint8_t> header = buildHeader(
            frame,
            jpeg.size(),
            droppedFrames_.load(std::memory_order_relaxed),
            sent + 1);
        return sendAll(header.data(), header.size()) && sendAll(jpeg.data(), jpeg.size());
    }

    void sendLoop() noexcept {
        try {
            PendingFrame frame;
            while (queue_.pop(frame)) {
                if (!waitUntilConnected(frame)) {
                    break;
                }

                PendingFrame newer;
                while (queue_.tryPop(newer)) {
                    frame = std::move(newer);
                    droppedFrames_.fetch_add(1, std::memory_order_relaxed);
                }

                if (!sendFrame(frame)) {
                    droppedFrames_.fetch_add(1, std::memory_order_relaxed);
                    if (running_.load(std::memory_order_acquire)) {
                        utils::FileLogger::instance().warning(
                            "[TCP] connection lost, waiting to reconnect");
                    }
                    closeSocket();
                    continue;
                }
                sentFrames_.fetch_add(1, std::memory_order_release);
            }
        } catch (const std::exception& exception) {
            utils::FileLogger::instance().error(
                std::string("[TCP] sender thread exception: ") + exception.what());
        } catch (...) {
            utils::FileLogger::instance().error("[TCP] sender thread caught unknown exception");
        }

        closeSocket();
    }

private:
    TcpSenderConfig config_;
    BlockingQueue<PendingFrame> queue_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<int> socketFd_{-1};
    std::atomic<std::uint64_t> sentFrames_{0};
    std::atomic<std::uint64_t> droppedFrames_{0};
    std::mutex reconnectMutex_;
    std::condition_variable reconnectCv_;
};

TcpFrameSender::TcpFrameSender(TcpSenderConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

TcpFrameSender::~TcpFrameSender() = default;

bool TcpFrameSender::start() {
    return impl_->start();
}

void TcpFrameSender::stop() noexcept {
    impl_->stop();
}

bool TcpFrameSender::enqueue(const FrameData& frame) noexcept {
    return impl_->enqueue(frame);
}

bool TcpFrameSender::isRunning() const noexcept {
    return impl_->isRunning();
}

bool TcpFrameSender::isConnected() const noexcept {
    return impl_->isConnected();
}

std::uint64_t TcpFrameSender::sentFrames() const noexcept {
    return impl_->sentFrames();
}

std::uint64_t TcpFrameSender::droppedFrames() const noexcept {
    return impl_->droppedFrames();
}

} // namespace egcinet::network
