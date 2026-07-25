#pragma once

#include "common/PreprocessData.h"

#include <cuda_runtime.h>

#include <cstddef>

class CudaPreprocessor {
public:
    CudaPreprocessor(int inputW, int inputH);
    explicit CudaPreprocessor(PreprocessConfig config);
    ~CudaPreprocessor() = default;

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;
    CudaPreprocessor(CudaPreprocessor&&) = delete;
    CudaPreprocessor& operator=(CudaPreprocessor&&) = delete;

    // 将已上传到 GPU 的 BGR 原图 resize、归一化后写成 NCHW。
    // 原图 H2D 由获取模块负责；本类不申请、释放或同步任何显存。
    bool process(
        const unsigned char* imageDevice,
        size_t imageBufferBytes,
        int imageWidth,
        int imageHeight,
        size_t imageStep,
        void* inputDevice,
        size_t inputBufferBytes,
        size_t inputElementSize,
        PreprocessResult& prep,
        cudaStream_t stream
    );

private:
    PreprocessConfig config_;
};
