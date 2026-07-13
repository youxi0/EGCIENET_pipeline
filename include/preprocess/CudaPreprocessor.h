#pragma once

#include "common/PreprocessData.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <cstddef>

class CudaPreprocessor {
public:
    CudaPreprocessor(int inputW, int inputH);
    explicit CudaPreprocessor(PreprocessConfig config);
    ~CudaPreprocessor();

    // 将 CPU 端 BGR 图像拷贝到 GPU，resize 到模型尺寸，归一化后写成 NCHW。
    // inputBufferBytes 用于在 kernel 启动前阻止预处理尺寸误配导致的显存越界。
    bool process(
        const cv::Mat& image,
        void* inputDevice,
        size_t inputBufferBytes,
        size_t inputElementSize,
        PreprocessResult& prep,
        cudaStream_t stream
    );

    // 返回最近一帧上传后的紧凑 BGR 设备缓冲区，供 GPU 可视化原地使用。
    unsigned char* imageDeviceBuffer() noexcept;
    size_t imageDeviceStep() const noexcept;
    int imageWidth() const noexcept;
    int imageHeight() const noexcept;

private:
    bool ensureImageBuffer(size_t bytes);
    void release();

private:
    PreprocessConfig config_;
    unsigned char* imageDevice_ = nullptr;
    size_t imageDeviceBytes_ = 0;
    size_t imageDeviceStep_ = 0;
    int imageWidth_ = 0;
    int imageHeight_ = 0;
};
