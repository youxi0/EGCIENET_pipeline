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

bool SegPostprocessor::attachOutputBuffers(
    float* probabilityDevice,
    size_t probabilityBufferBytes,
    std::uint8_t* binaryDevice,
    size_t binaryBufferBytes
) noexcept {
    if (probabilityDevice == nullptr || probabilityBufferBytes == 0 ||
        binaryDevice == nullptr || binaryBufferBytes == 0) {
        return false;
    }
    probabilityDevice_ = probabilityDevice;
    probabilityBufferBytes_ = probabilityBufferBytes;
    binaryDevice_ = binaryDevice;
    binaryBufferBytes_ = binaryBufferBytes;
    outputWidth_ = 0;
    outputHeight_ = 0;
    return true;
}

void SegPostprocessor::detachOutputBuffers() noexcept {
    probabilityDevice_ = nullptr;
    binaryDevice_ = nullptr;
    probabilityBufferBytes_ = 0;
    binaryBufferBytes_ = 0;
    outputWidth_ = 0;
    outputHeight_ = 0;
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

    if (probabilityDevice_ == nullptr || binaryDevice_ == nullptr) {
        std::cerr << "[Seg Postprocess] output buffers are not attached" << std::endl;
        return false;
    }

    if (!processToBuffers(
            modelMaskDevice,
            modelBufferBytes,
            modelElementSize,
            modelWidth,
            modelHeight,
            probabilityDevice_,
            probabilityBufferBytes_,
            binaryDevice_,
            binaryBufferBytes_,
            outputWidth,
            outputHeight,
            stream)) {
        return false;
    }

    outputWidth_ = outputWidth;
    outputHeight_ = outputHeight;
    return true;
}

bool SegPostprocessor::processToBuffers(
    const void* modelMaskDevice,
    size_t modelBufferBytes,
    size_t modelElementSize,
    int modelWidth,
    int modelHeight,
    float* probabilityDevice,
    size_t probabilityBufferBytes,
    std::uint8_t* binaryDevice,
    size_t binaryBufferBytes,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
) const {

    if (modelMaskDevice == nullptr || probabilityDevice == nullptr ||
        binaryDevice == nullptr) {
        std::cerr << "[Seg Postprocess] device buffer is null" << std::endl;
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

    const size_t outputPixels = width * height;
    if (outputPixels > std::numeric_limits<size_t>::max() / sizeof(float) ||
        probabilityBufferBytes < outputPixels * sizeof(float) ||
        binaryBufferBytes < outputPixels * sizeof(std::uint8_t)) {
        std::cerr << "[Seg Postprocess] output buffer is too small" << std::endl;
        return false;
    }

    launchSegPostprocessKernel(
        modelMaskDevice,
        modelElementSize,
        modelWidth,
        modelHeight,
        probabilityDevice,
        binaryDevice,
        outputWidth,
        outputHeight,
        config_.maskThreshold,
        stream
    );

    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        return false;
    }

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

    return enqueueDownloadFromBuffers(
        probabilityDevice_,
        probabilityBufferBytes_,
        binaryDevice_,
        binaryBufferBytes_,
        outputWidth_,
        outputHeight_,
        probabilityMask,
        binaryMask,
        stream);
}

bool SegPostprocessor::enqueueDownloadFromBuffers(
    const float* probabilityDevice,
    size_t probabilityBufferBytes,
    const std::uint8_t* binaryDevice,
    size_t binaryBufferBytes,
    int outputWidth,
    int outputHeight,
    cv::Mat& probabilityMask,
    cv::Mat& binaryMask,
    cudaStream_t stream
) const {
    if (probabilityDevice == nullptr || binaryDevice == nullptr ||
        outputWidth <= 0 || outputHeight <= 0) {
        std::cerr << "[Seg Postprocess] invalid download buffer or shape" << std::endl;
        return false;
    }

    const size_t width = static_cast<size_t>(outputWidth);
    const size_t height = static_cast<size_t>(outputHeight);
    if (height > std::numeric_limits<size_t>::max() / width) {
        std::cerr << "[Seg Postprocess] download pixel count overflow" << std::endl;
        return false;
    }
    const size_t pixelCount = width * height;
    if (pixelCount > std::numeric_limits<size_t>::max() / sizeof(float) ||
        probabilityBufferBytes < pixelCount * sizeof(float) ||
        binaryBufferBytes < pixelCount * sizeof(std::uint8_t)) {
        std::cerr << "[Seg Postprocess] download buffer is too small" << std::endl;
        return false;
    }

    probabilityMask.create(outputHeight, outputWidth, CV_32F);
    binaryMask.create(outputHeight, outputWidth, CV_8U);
    if (!checkCuda(
            cudaMemcpy2DAsync(
                probabilityMask.data,
                probabilityMask.step,
                probabilityDevice,
                width * sizeof(float),
                width * sizeof(float),
                height,
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
                binaryDevice,
                width * sizeof(std::uint8_t),
                width * sizeof(std::uint8_t),
                height,
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
