#include "common/Timer.h"

#include <iostream>

namespace {

// 统一打印 CUDA Event 错误，便于在服务器日志中定位计时失败位置。
bool checkCudaEvent(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "[CUDA Timer] " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

} // namespace

CpuTimer::CpuTimer() noexcept
    : start_(std::chrono::steady_clock::now()) {}

void CpuTimer::reset() noexcept {
    start_ = std::chrono::steady_clock::now();
}

double CpuTimer::elapsedMs() const noexcept {
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

CudaEventTimer::CudaEventTimer(cudaStream_t stream) noexcept
    : stream_(stream) {
    if (!checkCudaEvent(cudaEventCreate(&startEvent_), "cudaEventCreate start")) {
        return;
    }

    if (!checkCudaEvent(cudaEventCreate(&stopEvent_), "cudaEventCreate stop")) {
        release();
        return;
    }
}

CudaEventTimer::~CudaEventTimer() {
    release();
}

bool CudaEventTimer::isValid() const noexcept {
    return startEvent_ != nullptr && stopEvent_ != nullptr;
}

bool CudaEventTimer::start() noexcept {
    startRecorded_ = false;
    stopRecorded_ = false;
    if (!isValid()) {
        return false;
    }

    if (!checkCudaEvent(cudaEventRecord(startEvent_, stream_), "cudaEventRecord start")) {
        return false;
    }

    startRecorded_ = true;
    return true;
}

bool CudaEventTimer::stop() noexcept {
    if (!isValid() || !startRecorded_) {
        std::cerr << "[CUDA Timer] timer is not ready" << std::endl;
        return false;
    }

    if (!checkCudaEvent(cudaEventRecord(stopEvent_, stream_), "cudaEventRecord stop")) {
        return false;
    }

    stopRecorded_ = true;
    return true;
}

double CudaEventTimer::elapsedMs() noexcept {
    if (!isValid() || !startRecorded_ || !stopRecorded_) {
        std::cerr << "[CUDA Timer] timer is not ready" << std::endl;
        return -1.0;
    }

    if (!checkCudaEvent(cudaEventSynchronize(stopEvent_), "cudaEventSynchronize")) {
        return -1.0;
    }

    float milliseconds = 0.0f;
    if (!checkCudaEvent(
            cudaEventElapsedTime(&milliseconds, startEvent_, stopEvent_),
            "cudaEventElapsedTime")) {
        return -1.0;
    }

    startRecorded_ = false;
    stopRecorded_ = false;
    return static_cast<double>(milliseconds);
}

void CudaEventTimer::release() noexcept {
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(stopEvent_);
        stopEvent_ = nullptr;
    }

    if (startEvent_ != nullptr) {
        cudaEventDestroy(startEvent_);
        startEvent_ = nullptr;
    }

    startRecorded_ = false;
    stopRecorded_ = false;
}
