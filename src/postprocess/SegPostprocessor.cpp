#include "postprocess/SegPostprocessor.h"

#include "common/SegmentationClasses.h"

#include <cuda_fp16.h>

#include <iostream>
#include <limits>

void launchSegPostprocessKernel(
    const void* modelMask,
    size_t modelElementSize,
    int modelChannels,
    int modelWidth,
    int modelHeight,
    std::uint8_t* classMask,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
);

namespace {

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "[Seg Postprocess] " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

bool checkedModelBytes(
    int channels,
    int width,
    int height,
    size_t elementSize,
    size_t& requiredBytes
) noexcept {
    requiredBytes = 0;
    if (channels <= 0 || width <= 0 || height <= 0 || elementSize == 0) {
        return false;
    }

    size_t elements = static_cast<size_t>(channels);
    if (elements > std::numeric_limits<size_t>::max() /
            static_cast<size_t>(height)) {
        return false;
    }
    elements *= static_cast<size_t>(height);
    if (elements > std::numeric_limits<size_t>::max() /
            static_cast<size_t>(width)) {
        return false;
    }
    elements *= static_cast<size_t>(width);
    if (elements > std::numeric_limits<size_t>::max() / elementSize) {
        return false;
    }
    requiredBytes = elements * elementSize;
    return true;
}

bool checkedOutputPixels(int width, int height, size_t& pixels) noexcept {
    pixels = 0;
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (static_cast<size_t>(height) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(width)) {
        return false;
    }
    pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    return true;
}

} // namespace

bool SegPostprocessor::attachOutputBuffer(
    std::uint8_t* classMaskDevice,
    size_t classMaskBufferBytes
) noexcept {
    if (classMaskDevice == nullptr || classMaskBufferBytes == 0) {
        return false;
    }
    classMaskDevice_ = classMaskDevice;
    classMaskBufferBytes_ = classMaskBufferBytes;
    outputWidth_ = 0;
    outputHeight_ = 0;
    return true;
}

void SegPostprocessor::detachOutputBuffer() noexcept {
    classMaskDevice_ = nullptr;
    classMaskBufferBytes_ = 0;
    outputWidth_ = 0;
    outputHeight_ = 0;
}

bool SegPostprocessor::process(
    const void* modelMaskDevice,
    size_t modelBufferBytes,
    size_t modelElementSize,
    int modelChannels,
    int modelWidth,
    int modelHeight,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
) {
    outputWidth_ = 0;
    outputHeight_ = 0;
    if (classMaskDevice_ == nullptr) {
        std::cerr << "[Seg Postprocess] output buffer is not attached" << std::endl;
        return false;
    }

    if (!processToBuffers(
            modelMaskDevice,
            modelBufferBytes,
            modelElementSize,
            modelChannels,
            modelWidth,
            modelHeight,
            classMaskDevice_,
            classMaskBufferBytes_,
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
    int modelChannels,
    int modelWidth,
    int modelHeight,
    std::uint8_t* classMaskDevice,
    size_t classMaskBufferBytes,
    int outputWidth,
    int outputHeight,
    cudaStream_t stream
) const {
    if (modelMaskDevice == nullptr || classMaskDevice == nullptr || stream == nullptr) {
        std::cerr << "[Seg Postprocess] device buffer or stream is null" << std::endl;
        return false;
    }
    if (modelElementSize != sizeof(float) && modelElementSize != sizeof(__half)) {
        std::cerr << "[Seg Postprocess] only FP32 or FP16 model output is supported"
                  << std::endl;
        return false;
    }
    if (modelChannels != egcinet::segmentation::kClassCount) {
        std::cerr << "[Seg Postprocess] expected 5 output channels, got "
                  << modelChannels << std::endl;
        return false;
    }

    size_t requiredModelBytes = 0;
    if (!checkedModelBytes(
            modelChannels,
            modelWidth,
            modelHeight,
            modelElementSize,
            requiredModelBytes) ||
        requiredModelBytes > modelBufferBytes) {
        std::cerr << "[Seg Postprocess] model output buffer is too small or shape overflows"
                  << std::endl;
        return false;
    }

    size_t outputPixels = 0;
    if (!checkedOutputPixels(outputWidth, outputHeight, outputPixels) ||
        classMaskBufferBytes < outputPixels * sizeof(std::uint8_t)) {
        std::cerr << "[Seg Postprocess] class mask buffer is too small or shape is invalid"
                  << std::endl;
        return false;
    }

    launchSegPostprocessKernel(
        modelMaskDevice,
        modelElementSize,
        modelChannels,
        modelWidth,
        modelHeight,
        classMaskDevice,
        outputWidth,
        outputHeight,
        stream
    );
    return checkCuda(cudaGetLastError(), "argmax kernel launch");
}

bool SegPostprocessor::enqueueDownload(
    cv::Mat& classMask,
    cudaStream_t stream
) const {
    if (classMaskDevice_ == nullptr || outputWidth_ <= 0 || outputHeight_ <= 0) {
        std::cerr << "[Seg Postprocess] no valid GPU result to download" << std::endl;
        return false;
    }
    return enqueueDownloadFromBuffers(
        classMaskDevice_,
        classMaskBufferBytes_,
        outputWidth_,
        outputHeight_,
        classMask,
        stream);
}

bool SegPostprocessor::enqueueDownloadFromBuffers(
    const std::uint8_t* classMaskDevice,
    size_t classMaskBufferBytes,
    int outputWidth,
    int outputHeight,
    cv::Mat& classMask,
    cudaStream_t stream
) const {
    size_t pixelCount = 0;
    if (classMaskDevice == nullptr || stream == nullptr ||
        !checkedOutputPixels(outputWidth, outputHeight, pixelCount) ||
        classMaskBufferBytes < pixelCount * sizeof(std::uint8_t)) {
        std::cerr << "[Seg Postprocess] invalid class mask download buffer or shape"
                  << std::endl;
        return false;
    }

    classMask.create(outputHeight, outputWidth, CV_8UC1);
    if (!checkCuda(
            cudaMemcpy2DAsync(
                classMask.data,
                classMask.step,
                classMaskDevice,
                static_cast<size_t>(outputWidth),
                static_cast<size_t>(outputWidth),
                static_cast<size_t>(outputHeight),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "class mask D2H")) {
        classMask.release();
        return false;
    }
    return true;
}

const std::uint8_t* SegPostprocessor::classMaskDeviceBuffer() const noexcept {
    return classMaskDevice_;
}

int SegPostprocessor::outputWidth() const noexcept {
    return outputWidth_;
}

int SegPostprocessor::outputHeight() const noexcept {
    return outputHeight_;
}
