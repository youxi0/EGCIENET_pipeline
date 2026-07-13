#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

struct SegPostprocessConfig {
    float maskThreshold = 0.5f;
};

// 在 GPU 上把模型尺度概率图恢复到原图尺寸，并生成二值分割图。
class SegPostprocessor {
public:
    explicit SegPostprocessor(SegPostprocessConfig config = {});
    ~SegPostprocessor();

    SegPostprocessor(const SegPostprocessor&) = delete;
    SegPostprocessor& operator=(const SegPostprocessor&) = delete;
    SegPostprocessor(SegPostprocessor&&) = delete;
    SegPostprocessor& operator=(SegPostprocessor&&) = delete;

    // 输入为 TensorRT output binding；输出仍保留在 GPU，不执行 D2H。
    bool process(
        const void* modelMaskDevice,
        size_t modelBufferBytes,
        size_t modelElementSize,
        int modelWidth,
        int modelHeight,
        int outputWidth,
        int outputHeight,
        cudaStream_t stream
    );

    // 在同一 stream 上排队下载概率图和二值图；该函数不主动同步。
    bool enqueueDownload(
        cv::Mat& probabilityMask,
        cv::Mat& binaryMask,
        cudaStream_t stream
    ) const;

    const float* probabilityDeviceBuffer() const noexcept;
    const std::uint8_t* binaryDeviceBuffer() const noexcept;
    int outputWidth() const noexcept;
    int outputHeight() const noexcept;

private:
    // 只在原图像素数超过历史容量时扩容。
    bool ensureOutputBuffers(size_t pixelCount);
    void release() noexcept;

private:
    SegPostprocessConfig config_;
    float* probabilityDevice_ = nullptr;
    std::uint8_t* binaryDevice_ = nullptr;
    size_t capacityPixels_ = 0;
    int outputWidth_ = 0;
    int outputHeight_ = 0;
};
