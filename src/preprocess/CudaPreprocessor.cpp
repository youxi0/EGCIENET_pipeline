#include "preprocess/CudaPreprocessor.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <iostream>

void launchPreprocessKernel(
    const unsigned char* src,
    int srcW,
    int srcH,
    int srcStep,
    void* dst,
    int dstW,
    int dstH,
    size_t dstElementSize,
    float scale,
    int padX,
    int padY,
    cudaStream_t stream
);

CudaPreprocessor::CudaPreprocessor(int inputW, int inputH)
    : inputW_(inputW),
      inputH_(inputH) {}

CudaPreprocessor::~CudaPreprocessor() {
    release();
}

void CudaPreprocessor::release() {
    if (imageDevice_) {
        cudaFree(imageDevice_);
        imageDevice_ = nullptr;
        imageDeviceBytes_ = 0;
    }
}

bool CudaPreprocessor::ensureImageBuffer(size_t bytes) {
    if (bytes <= imageDeviceBytes_) {
        return true;
    }

    release();

    cudaError_t err = cudaMalloc(&imageDevice_, bytes);

    if (err != cudaSuccess) {
        std::cerr << "[CUDA Preprocess] cudaMalloc image buffer failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }

    imageDeviceBytes_ = bytes;
    return true;
}

bool CudaPreprocessor::process(
    const cv::Mat& image,
    void* inputDevice,
    size_t inputElementSize,
    PreprocessResult& prep,
    cudaStream_t stream
) {
    if (image.empty()) {
        std::cerr << "[CUDA Preprocess] input image is empty" << std::endl;
        return false;
    }

    if (!inputDevice) {
        std::cerr << "[CUDA Preprocess] TensorRT input device buffer is null" << std::endl;
        return false;
    }

    // 修改点: 当前CUDA预处理只支持写FP32或FP16输入，防止INT8输入被误写成float。
    if (inputElementSize != sizeof(float) && inputElementSize != sizeof(__half)) {
        std::cerr << "[CUDA Preprocess] unsupported TensorRT input element size: "
                  << inputElementSize << std::endl;
        return false;
    }

    if (image.type() != CV_8UC3) {
        std::cerr << "[CUDA Preprocess] only CV_8UC3 image is supported" << std::endl;
        return false;
    }

    prep.original_width = image.cols;
    prep.original_height = image.rows;

    float scale = std::min(
        static_cast<float>(inputW_) / static_cast<float>(image.cols),
        static_cast<float>(inputH_) / static_cast<float>(image.rows)
    );

    int resizedW = static_cast<int>(std::round(image.cols * scale));
    int resizedH = static_cast<int>(std::round(image.rows * scale));

    int padX = (inputW_ - resizedW) / 2;
    int padY = (inputH_ - resizedH) / 2;

    prep.scale = scale;
    prep.pad_x = padX;
    prep.pad_y = padY;

    size_t imageBytes = image.step * image.rows;

    if (!ensureImageBuffer(imageBytes)) {
        return false;
    }

    cudaError_t err = cudaMemcpyAsync(
        imageDevice_,
        image.data,
        imageBytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        std::cerr << "[CUDA Preprocess] cudaMemcpyAsync H2D failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }

    launchPreprocessKernel(
        imageDevice_,
        image.cols,
        image.rows,
        static_cast<int>(image.step),
        inputDevice,
        inputW_,
        inputH_,
        inputElementSize,
        scale,
        padX,
        padY,
        stream
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "[CUDA Preprocess] kernel launch failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }

    return true;
}
