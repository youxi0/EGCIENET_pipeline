#pragma once

#include <cuda_runtime_api.h>

#include <chrono>

// 使用单调时钟统计纯 CPU 代码耗时。
class CpuTimer {
public:
    CpuTimer() noexcept;

    // 重新记录 CPU 计时起点。
    void reset() noexcept;

    // 返回从最近一次 reset 到当前时刻的毫秒数。
    double elapsedMs() const noexcept;

private:
    std::chrono::steady_clock::time_point start_;
};

// 使用 CUDA Event 统计指定 stream 上的 GPU 工作耗时。
// elapsedMs 会等待结束事件完成；返回负数表示 CUDA 调用失败。
class CudaEventTimer {
public:
    explicit CudaEventTimer(cudaStream_t stream) noexcept;
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&&) = delete;
    CudaEventTimer& operator=(CudaEventTimer&&) = delete;

    // 检查两个 CUDA Event 是否创建成功。
    bool isValid() const noexcept;

    // 在绑定的 stream 上记录起始事件。
    bool start() noexcept;

    // 在绑定的 stream 上记录结束事件；该函数不会同步 GPU。
    bool stop() noexcept;

    // 等待结束事件完成，返回两个事件之间的毫秒数。
    double elapsedMs() noexcept;

private:
    void release() noexcept;

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t stopEvent_ = nullptr;
    bool startRecorded_ = false;
    bool stopRecorded_ = false;
};
