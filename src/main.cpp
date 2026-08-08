#include "acquisition/ImageFolderSource.h"
#include "acquisition/ImageSource.h"
#include "acquisition/VideoSource.h"
#include "network/TcpFrameSender.h"
#include "pipeline/InspectionPipeline.h"
#include "utils/FileLogger.h"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning.store(false, std::memory_order_release);
}

std::string getArg(
    int argc,
    char** argv,
    const std::string& key,
    const std::string& defaultValue = ""
) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == key) {
            return argv[index + 1];
        }
    }
    return defaultValue;
}

// 解析 BGR 顺序的三通道参数，例如 140.505,157.845,135.66。
std::array<float, 3> parseTriplet(
    const std::string& text,
    const char* argumentName
) {
    std::array<float, 3> values{};
    std::istringstream stream(text);
    std::string item;

    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::getline(stream, item, ',') || item.empty()) {
            throw std::invalid_argument(
                std::string(argumentName) + " must contain three comma-separated values");
        }

        std::size_t consumed = 0;
        values[index] = std::stof(item, &consumed);
        if (consumed != item.size()) {
            throw std::invalid_argument(std::string(argumentName) + " contains invalid value");
        }
    }

    if (std::getline(stream, item, ',')) {
        throw std::invalid_argument(
            std::string(argumentName) + " must contain exactly three values");
    }
    return values;
}

std::size_t parseQueueSize(const std::string& text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 || value > 16) {
        throw std::invalid_argument("--queue_size must be in [1, 16]");
    }
    return static_cast<std::size_t>(value);
}

int parsePositiveDimension(const std::string& text, const char* argumentName) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(
            std::string(argumentName) + " must be a positive integer");
    }
    return value;
}

int parseIntegerRange(
    const std::string& text,
    const char* argumentName,
    int minimum,
    int maximum
) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < minimum || value > maximum) {
        std::ostringstream message;
        message << argumentName << " must be in [" << minimum << ", " << maximum << ']';
        throw std::invalid_argument(message.str());
    }
    return value;
}

std::unique_ptr<ImageSource> createSource(
    const std::string& sourceType,
    const std::string& sourcePath
) {
    if (sourceType == "folder") {
        return std::make_unique<ImageFolderSource>(sourcePath);
    }
    if (sourceType == "video") {
        return std::make_unique<VideoSource>(sourcePath);
    }
    if (sourceType == "camera") {
        std::size_t consumed = 0;
        const int cameraId = std::stoi(sourcePath, &consumed);
        if (consumed != sourcePath.size() || cameraId < 0) {
            throw std::invalid_argument("camera source must be a non-negative integer");
        }
        return std::make_unique<VideoSource>(cameraId);
    }
    throw std::invalid_argument("--type must be folder, video or camera");
}

void printUsage(const char* app) {
    std::cout
        << "Usage:\n"
        << "  " << app
        << " --engine models/egcienet_352_multiclass_fp16.engine"
        << " --source datasets/images/val --type folder [options]\n\n"
        << "Options:\n"
        << "  --type         folder | video | camera, default folder\n"
        << "  --queue_size   frame slot count, range 1-16, default 3\n"
        << "  --max_width    maximum source width, default 1920\n"
        << "  --max_height   maximum source height, default 1080\n"
        << "  --mean         B,G,R raw-pixel mean, default 140.505,157.845,135.66\n"
        << "  --std          B,G,R raw-pixel std, default 61.455,60.18,62.22\n"
        << "  --save_dir     save class mask and visualization; empty means disabled\n"
        << "  --log_dir      runtime log directory, default results/logs\n"
        << "  --tcp_host     upper-computer address; empty means TCP disabled\n"
        << "  --tcp_port     upper-computer listening port, default 9000\n"
        << "  --tcp_queue    asynchronous display queue size, range 1-8, default 2\n"
        << "  --jpeg_quality transmitted JPEG quality, range 1-100, default 85\n";
}

std::string frameStem(std::uint64_t frameId) {
    std::ostringstream stream;
    stream << "frame_" << std::setw(8) << std::setfill('0') << frameId;
    return stream.str();
}

// 结果回调位于完成线程，保存图像会对槽位回收和后续帧产生背压。
void handleResult(
    const FrameData& frame,
    const std::string& saveDirectory,
    egcinet::network::TcpFrameSender* tcpSender
) {
    std::ostringstream timingLog;
    timingLog << std::fixed << std::setprecision(3)
              << "[Pipeline] frame=" << frame.frameId
              << " acquire=" << frame.cost.acquire_ms << " ms"
              << " h2d=" << frame.cost.h2d_ms << " ms"
              << " preprocess=" << frame.cost.preprocess_ms << " ms"
              << " infer=" << frame.cost.infer_ms << " ms"
              << " postprocess=" << frame.cost.postprocess_ms << " ms"
              << " visualize=" << frame.cost.visualize_ms << " ms"
              << " d2h=" << frame.cost.d2h_ms << " ms"
              << " total=" << frame.cost.total_ms << " ms";
    utils::FileLogger::instance().info(timingLog.str());

    if (tcpSender != nullptr) {
        tcpSender->enqueue(frame);
    }

    if (saveDirectory.empty()) {
        return;
    }

    const std::filesystem::path outputDirectory(saveDirectory);
    const std::string stem = frameStem(frame.frameId);
    const std::filesystem::path classMaskPath =
        outputDirectory / (stem + "_classes.png");
    const std::filesystem::path visualizationPath =
        outputDirectory / (stem + "_visualized.jpg");

    // 类别图必须原样保存 0~4，不能乘 255，否则会破坏训练标签 ID。
    if (!cv::imwrite(classMaskPath.string(), frame.classMask) ||
        !cv::imwrite(visualizationPath.string(), frame.visualizedImage)) {
        throw std::runtime_error("failed to save output masks for frame "
            + std::to_string(frame.frameId));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    const std::string enginePath = getArg(argc, argv, "--engine");
    const std::string sourcePath = getArg(argc, argv, "--source");
    if (enginePath.empty() || sourcePath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::string sourceType = getArg(argc, argv, "--type", "folder");
        const std::string saveDirectory = getArg(argc, argv, "--save_dir");
        const std::string logDirectory =
            getArg(argc, argv, "--log_dir", "results/logs");
        const std::string tcpHost = getArg(argc, argv, "--tcp_host");
        const int tcpPort = parseIntegerRange(
            getArg(argc, argv, "--tcp_port", "9000"), "--tcp_port", 1, 65535);
        const int tcpQueueSize = parseIntegerRange(
            getArg(argc, argv, "--tcp_queue", "2"), "--tcp_queue", 1, 8);
        const int jpegQuality = parseIntegerRange(
            getArg(argc, argv, "--jpeg_quality", "85"), "--jpeg_quality", 1, 100);

        if (!saveDirectory.empty()) {
            std::filesystem::create_directories(saveDirectory);
        }
        if (!utils::FileLogger::instance().open(
                logDirectory, "egcinet_pipeline")) {
            return 1;
        }

        std::unique_ptr<egcinet::network::TcpFrameSender> tcpSender;
        if (!tcpHost.empty()) {
            egcinet::network::TcpSenderConfig tcpConfig;
            tcpConfig.host = tcpHost;
            tcpConfig.port = static_cast<std::uint16_t>(tcpPort);
            tcpConfig.queueSize = static_cast<std::size_t>(tcpQueueSize);
            tcpConfig.jpegQuality = jpegQuality;
            tcpSender = std::make_unique<egcinet::network::TcpFrameSender>(
                std::move(tcpConfig));
            if (!tcpSender->start()) {
                return 1;
            }
        }

        egcinet::pipeline::PipelineConfig config;
        config.enginePath = enginePath;
        config.queueSize = parseQueueSize(getArg(argc, argv, "--queue_size", "3"));
        config.maxSourceWidth = parsePositiveDimension(
            getArg(argc, argv, "--max_width", "1920"), "--max_width");
        config.maxSourceHeight = parsePositiveDimension(
            getArg(argc, argv, "--max_height", "1080"), "--max_height");
        config.mean = parseTriplet(
            getArg(argc, argv, "--mean", "140.505,157.845,135.66"),
            "--mean");
        config.std = parseTriplet(
            getArg(argc, argv, "--std", "61.455,60.18,62.22"),
            "--std");
        config.enableVisualization = !saveDirectory.empty() || tcpSender != nullptr;
        config.resultCallback = [saveDirectory, sender = tcpSender.get()](const FrameData& frame) {
            handleResult(frame, saveDirectory, sender);
        };

        std::ostringstream startupLog;
        startupLog << "[Pipeline] configuration: engine=" << enginePath
                   << ", source=" << sourcePath
                   << ", type=" << sourceType
                   << ", slots=" << config.queueSize
                   << ", max_source=" << config.maxSourceWidth << "x"
                   << config.maxSourceHeight
                   << ", visualization=" << (config.enableVisualization ? "on" : "off")
                   << ", tcp=" << (tcpHost.empty()
                       ? "off"
                       : tcpHost + ':' + std::to_string(tcpPort));
        utils::FileLogger::instance().info(startupLog.str());

        auto source = createSource(sourceType, sourcePath);
        egcinet::pipeline::InspectionPipeline pipeline(std::move(source), std::move(config));
        if (!pipeline.start()) {
            return 1;
        }

        utils::FileLogger::instance().info(
            "[Pipeline] running, press Ctrl+C to stop");
        while (gRunning.load(std::memory_order_acquire) && pipeline.isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        pipeline.stop();
        const std::string pipelineError = pipeline.lastError();
        utils::FileLogger::instance().info(
            "[Pipeline] processed frames: " +
            std::to_string(pipeline.processedFrames()));
        if (tcpSender != nullptr) {
            tcpSender->stop();
            utils::FileLogger::instance().info(
                "[TCP] summary: sent=" + std::to_string(tcpSender->sentFrames()) +
                ", dropped=" + std::to_string(tcpSender->droppedFrames()));
        }
        if (!pipelineError.empty()) {
            return 1;
        }
    } catch (const std::exception& exception) {
        utils::FileLogger::instance().error(
            std::string("[Pipeline] invalid argument or runtime error: ") +
            exception.what());
        return 1;
    }

    return 0;
}
