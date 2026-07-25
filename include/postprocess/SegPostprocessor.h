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
    ~SegPostprocessor() = default;

    SegPostprocessor(const SegPostprocessor&) = delete;
    SegPostprocessor& operator=(const SegPostprocessor&) = delete;
    SegPostprocessor(SegPostprocessor&&) = delete;
    SegPostprocessor& operator=(SegPostprocessor&&) = delete;

    // 挂接调用方拥有的概率图和二值图显存；本类不申请或释放这些资源。
    bool attachOutputBuffers(
        float* probabilityDevice,
        size_t probabilityBufferBytes,
        std::uint8_t* binaryDevice,
        size_t binaryBufferBytes
    ) noexcept;
    void detachOutputBuffers() noexcept;

    // 输入为 TensorRT output tensor；输出仍保留在 GPU，不执行 D2H。
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

    // Pipeline 路径：将后处理结果写入槽位显存，不申请显存也不修改内部输出状态。
    bool processToBuffers(
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
    ) const;

    // 在同一 stream 上排队下载概率图和二值图；该函数不主动同步。
    bool enqueueDownload(
        cv::Mat& probabilityMask,
        cv::Mat& binaryMask,
        cudaStream_t stream
    ) const;

    // Pipeline 路径：从指定槽位显存下载结果，不依赖内部输出缓冲区。
    bool enqueueDownloadFromBuffers(
        const float* probabilityDevice,
        size_t probabilityBufferBytes,
        const std::uint8_t* binaryDevice,
        size_t binaryBufferBytes,
        int outputWidth,
        int outputHeight,
        cv::Mat& probabilityMask,
        cv::Mat& binaryMask,
        cudaStream_t stream
    ) const;

    const float* probabilityDeviceBuffer() const noexcept;
    const std::uint8_t* binaryDeviceBuffer() const noexcept;
    int outputWidth() const noexcept;
    int outputHeight() const noexcept;

private:
    SegPostprocessConfig config_;
    // 非拥有指针，仅供单图和验证工具的兼容路径使用。
    float* probabilityDevice_ = nullptr;
    std::uint8_t* binaryDevice_ = nullptr;
    size_t probabilityBufferBytes_ = 0;
    size_t binaryBufferBytes_ = 0;
    int outputWidth_ = 0;
    int outputHeight_ = 0;
};
