#include "postprocess/SegPostprocessor.h"

#include <cuda_fp16.h>

#include <iostream>
#include <limits>

void launchSegPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelWidth,
    int modelHeight,
    float* probabilityMask,
    std::uint8_t* binaryMask,
    int outputWidth,
    int outputHeight,
    float threshold,
    cudaStream_t stream
);

namespace {

// 统一检查后处理阶段的 CUDA 调用。
bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "[Seg Postprocess] " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

} // namespace

SegPostprocessor::SegPostprocessor(SegPostprocessConfig config)
    : config_(config) {}

SegPostprocessor::~SegPostprocessor() {
    release();
}

bool SegPostprocessor::process(
    const void* modelMaskDevice,
    size_t modelBufferBytes,
    size_t modelElementSize,
    int modelWidth,
    int modelHeight,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
) {
    outputWidth_ = 0;
    outputHeight_ = 0;

    if (modelMaskDevice == nullptr) {
        std::cerr << "[Seg Postprocess] model mask device buffer is null" << std::endl;
        return false;
    }

    if (modelElementSize != sizeof(float) && modelElementSize != sizeof(__half)) {
        std::cerr << "[Seg Postprocess] only FP32 or FP16 model output is supported" << std::endl;
        return false;
    }

    if (modelWidth <= 0 || modelHeight <= 0 || outputWidth <= 0 || outputHeight <= 0) {
        std::cerr << "[Seg Postprocess] invalid input or output shape" << std::endl;
        return false;
    }

    const size_t modelWidthValue = static_cast<size_t>(modelWidth);
    const size_t modelHeightValue = static_cast<size_t>(modelHeight);
    if (modelHeightValue > std::numeric_limits<size_t>::max() / modelWidthValue) {
        std::cerr << "[Seg Postprocess] model pixel count overflow" << std::endl;
        return false;
    }

    const size_t modelPixels = modelWidthValue * modelHeightValue;
    if (modelPixels > std::numeric_limits<size_t>::max() / modelElementSize ||
        modelPixels * modelElementSize > modelBufferBytes) {
        std::cerr << "[Seg Postprocess] model output buffer is too small" << std::endl;
        return false;
    }

    if (config_.maskThreshold < 0.0f || config_.maskThreshold > 1.0f) {
        std::cerr << "[Seg Postprocess] mask threshold must be in [0, 1]" << std::endl;
        return false;
    }

    const size_t width = static_cast<size_t>(outputWidth);
    const size_t height = static_cast<size_t>(outputHeight);
    if (height > std::numeric_limits<size_t>::max() / width) {
        std::cerr << "[Seg Postprocess] output pixel count overflow" << std::endl;
        return false;
    }

    if (!ensureOutputBuffers(width * height)) {
        return false;
    }

    launchSegPostprocessKernel(
        modelMaskDevice,
        modelElementSize,
        modelWidth,
        modelHeight,
        probabilityDevice_,
        binaryDevice_,
        outputWidth,
        outputHeight,
        config_.maskThreshold,
        stream
    );

    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        return false;
    }

    outputWidth_ = outputWidth;
    outputHeight_ = outputHeight;
    return true;
}

bool SegPostprocessor::enqueueDownload(
    cv::Mat& probabilityMask,
    cv::Mat& binaryMask,
    cudaStream_t stream
) const {
    if (probabilityDevice_ == nullptr || binaryDevice_ == nullptr ||
        outputWidth_ <= 0 || outputHeight_ <= 0) {
        std::cerr << "[Seg Postprocess] no valid GPU result to download" << std::endl;
        return false;
    }

    probabilityMask.create(outputHeight_, outputWidth_, CV_32F);
    binaryMask.create(outputHeight_, outputWidth_, CV_8U);
    if (!checkCuda(
            cudaMemcpy2DAsync(
                probabilityMask.data,
                probabilityMask.step,
                probabilityDevice_,
                static_cast<size_t>(outputWidth_) * sizeof(float),
                static_cast<size_t>(outputWidth_) * sizeof(float),
                static_cast<size_t>(outputHeight_),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "probability mask D2H")) {
        probabilityMask.release();
        binaryMask.release();
        return false;
    }

    if (!checkCuda(
            cudaMemcpy2DAsync(
                binaryMask.data,
                binaryMask.step,
                binaryDevice_,
                static_cast<size_t>(outputWidth_) * sizeof(std::uint8_t),
                static_cast<size_t>(outputWidth_) * sizeof(std::uint8_t),
                static_cast<size_t>(outputHeight_),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "binary mask D2H")) {
        probabilityMask.release();
        binaryMask.release();
        return false;
    }

    return true;
}

const float* SegPostprocessor::probabilityDeviceBuffer() const noexcept {
    return probabilityDevice_;
}

const std::uint8_t* SegPostprocessor::binaryDeviceBuffer() const noexcept {
    return binaryDevice_;
}

int SegPostprocessor::outputWidth() const noexcept {
    return outputWidth_;
}

int SegPostprocessor::outputHeight() const noexcept {
    return outputHeight_;
}

bool SegPostprocessor::ensureOutputBuffers(size_t pixelCount) {
    if (pixelCount <= capacityPixels_ &&
        probabilityDevice_ != nullptr && binaryDevice_ != nullptr) {
        return true;
    }

    release();

    if (pixelCount > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::cerr << "[Seg Postprocess] probability buffer size overflow" << std::endl;
        return false;
    }

    if (!checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&probabilityDevice_),
                pixelCount * sizeof(float)
            ),
            "cudaMalloc probability mask")) {
        release();
        return false;
    }

    if (!checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&binaryDevice_),
                pixelCount * sizeof(std::uint8_t)
            ),
            "cudaMalloc binary mask")) {
        release();
        return false;
    }

    capacityPixels_ = pixelCount;
    return true;
}

void SegPostprocessor::release() noexcept {
    if (binaryDevice_ != nullptr) {
        cudaFree(binaryDevice_);
        binaryDevice_ = nullptr;
    }

    if (probabilityDevice_ != nullptr) {
        cudaFree(probabilityDevice_);
        probabilityDevice_ = nullptr;
    }

    capacityPixels_ = 0;
    outputWidth_ = 0;
    outputHeight_ = 0;
}
