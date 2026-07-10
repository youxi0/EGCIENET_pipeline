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
    bool process(
        const cv::Mat& image,
        void* inputDevice,
        size_t inputElementSize,
        PreprocessResult& prep,
        cudaStream_t stream
    );

private:
    bool ensureImageBuffer(size_t bytes);
    void release();

private:
    PreprocessConfig config_;
    unsigned char* imageDevice_ = nullptr;
    size_t imageDeviceBytes_ = 0;
};
