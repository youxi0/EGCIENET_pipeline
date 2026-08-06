#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct FrameData;

namespace egcinet::network {

struct TcpSenderConfig {
    // host/port 指向运行 Qt 监视器的上位机；host 既可为 IPv4/IPv6，也可为主机名。
    std::string host;
    std::uint16_t port = 9000;

    // 小队列只保留近期显示帧。网络跟不上时允许丢帧，不能反压推理流水线。
    std::size_t queueSize = 2;
    int jpegQuality = 85;

    // 连接和重连都只发生在发送线程，不占用 pipeline 完成线程。
    int connectTimeoutMs = 500;
    int reconnectIntervalMs = 1000;
};

// 将可视化结果和性能指标异步发送到上位机。
// 网络线程与 pipeline 完成线程隔离；队列满时丢弃显示帧，不阻塞槽位回收。
class TcpFrameSender {
public:
    explicit TcpFrameSender(TcpSenderConfig config);
    ~TcpFrameSender();

    TcpFrameSender(const TcpFrameSender&) = delete;
    TcpFrameSender& operator=(const TcpFrameSender&) = delete;
    TcpFrameSender(TcpFrameSender&&) = delete;
    TcpFrameSender& operator=(TcpFrameSender&&) = delete;

    // 启动发送线程。上位机暂未监听不视为启动失败，线程会自动重连。
    bool start();
    void stop() noexcept;

    // 克隆当前可视化图后非阻塞入队；返回 false 表示该显示帧被丢弃。
    bool enqueue(const FrameData& frame) noexcept;

    bool isRunning() const noexcept;
    bool isConnected() const noexcept;
    std::uint64_t sentFrames() const noexcept;
    std::uint64_t droppedFrames() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace egcinet::network
