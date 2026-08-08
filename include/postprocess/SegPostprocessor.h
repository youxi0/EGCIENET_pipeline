#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

// 在 GPU 上对 [1,5,H,W] 输出逐像素执行 argmax，并恢复到原图尺寸。
// 恢复尺寸时先双线性采样每个类别分数，再比较类别，不能插值类别 ID。
class SegPostprocessor {
public:
    SegPostprocessor() = default;
    ~SegPostprocessor() = default;

    SegPostprocessor(const SegPostprocessor&) = delete;
    SegPostprocessor& operator=(const SegPostprocessor&) = delete;
    SegPostprocessor(SegPostprocessor&&) = delete;
    SegPostprocessor& operator=(SegPostprocessor&&) = delete;

    // 挂接调用方拥有的类别图显存；本类不申请或释放该资源。
    bool attachOutputBuffer(
        std::uint8_t* classMaskDevice,
        size_t classMaskBufferBytes
    ) noexcept;
    void detachOutputBuffer() noexcept;

    // 输入为 TensorRT output tensor；输出仍保留在 GPU，不执行 D2H。
    bool process(
        const void* modelMaskDevice,
        size_t modelBufferBytes,
        size_t modelElementSize,
        int modelChannels,
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
        int modelChannels,
        int modelWidth,
        int modelHeight,
        std::uint8_t* classMaskDevice,
        size_t classMaskBufferBytes,
        int outputWidth,
        int outputHeight,
        cudaStream_t stream
    ) const;

    // 在同一 stream 上排队下载类别图；该函数不主动同步。
    bool enqueueDownload(
        cv::Mat& classMask,
        cudaStream_t stream
    ) const;

    // Pipeline 路径：从指定槽位显存下载结果，不依赖内部输出缓冲区。
    bool enqueueDownloadFromBuffers(
        const std::uint8_t* classMaskDevice,
        size_t classMaskBufferBytes,
        int outputWidth,
        int outputHeight,
        cv::Mat& classMask,
        cudaStream_t stream
    ) const;

    const std::uint8_t* classMaskDeviceBuffer() const noexcept;
    int outputWidth() const noexcept;
    int outputHeight() const noexcept;

private:
    // 非拥有指针，仅供单图和验证工具的兼容路径使用。
    std::uint8_t* classMaskDevice_ = nullptr;
    size_t classMaskBufferBytes_ = 0;
    int outputWidth_ = 0;
    int outputHeight_ = 0;
};
