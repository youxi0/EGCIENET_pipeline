#include "preprocess/CudaPreprocessor.h"

#include <cuda_fp16.h>

#include <iostream>
#include <limits>

void launchPreprocessKernel(
    const unsigned char* src,
    int srcW,
    int srcH,
    int srcStep,
    void* dst,
    int dstW,
    int dstH,
    size_t dstElementSize,
    float meanB,
    float meanG,
    float meanR,
    float stdB,
    float stdG,
    float stdR,
    cudaStream_t stream
);

CudaPreprocessor::CudaPreprocessor(int inputW, int inputH)
    : config_{inputW, inputH, {140.505f, 157.845f, 135.66f}, {61.455f, 60.18f, 62.22f}} {}

CudaPreprocessor::CudaPreprocessor(PreprocessConfig config)
    : config_(config) {}

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
    size_t inputBufferBytes,
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

    // 当前 CUDA 预处理只支持写 FP32 或 FP16 输入，防止误写 INT8 binding。
    if (inputElementSize != sizeof(float) && inputElementSize != sizeof(__half)) {
        std::cerr << "[CUDA Preprocess] unsupported TensorRT input element size: "
                  << inputElementSize << std::endl;
        return false;
    }

    if (image.type() != CV_8UC3) {
        std::cerr << "[CUDA Preprocess] only CV_8UC3 image is supported" << std::endl;
        return false;
    }

    if (config_.inputWidth <= 0 || config_.inputHeight <= 0) {
        std::cerr << "[CUDA Preprocess] invalid input size" << std::endl;
        return false;
    }

    const size_t inputPixels =
        static_cast<size_t>(config_.inputWidth) * static_cast<size_t>(config_.inputHeight);

    if (inputPixels > std::numeric_limits<size_t>::max() / 3 ||
        inputPixels * 3 > std::numeric_limits<size_t>::max() / inputElementSize) {
        std::cerr << "[CUDA Preprocess] input buffer size overflow" << std::endl;
        return false;
    }

    const size_t requiredInputBytes = inputPixels * 3 * inputElementSize;
    if (requiredInputBytes > inputBufferBytes) {
        std::cerr << "[CUDA Preprocess] TensorRT input buffer is too small: required="
                  << requiredInputBytes << ", capacity=" << inputBufferBytes << std::endl;
        return false;
    }

    if (config_.std[0] == 0.0f || config_.std[1] == 0.0f || config_.std[2] == 0.0f) {
        std::cerr << "[CUDA Preprocess] std values must be non-zero" << std::endl;
        return false;
    }

    prep.originalWidth = image.cols;
    prep.originalHeight = image.rows;
    prep.inputWidth = config_.inputWidth;
    prep.inputHeight = config_.inputHeight;
    prep.scaleX = static_cast<float>(config_.inputWidth) / static_cast<float>(image.cols);
    prep.scaleY = static_cast<float>(config_.inputHeight) / static_cast<float>(image.rows);
    prep.blob.release();

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
        config_.inputWidth,
        config_.inputHeight,
        inputElementSize,
        config_.mean[0],
        config_.mean[1],
        config_.mean[2],
        config_.std[0],
        config_.std[1],
        config_.std[2],
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
