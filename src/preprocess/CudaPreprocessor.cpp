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

bool CudaPreprocessor::process(
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
) {
    if (imageDevice == nullptr || imageWidth <= 0 || imageHeight <= 0 || stream == nullptr) {
        std::cerr << "[CUDA Preprocess] invalid GPU image or CUDA stream" << std::endl;
        return false;
    }

    if (!inputDevice) {
        std::cerr << "[CUDA Preprocess] TensorRT input device buffer is null" << std::endl;
        return false;
    }

    // 当前 CUDA 预处理只支持写 FP32 或 FP16 输入，防止误写 INT8 tensor。
    if (inputElementSize != sizeof(float) && inputElementSize != sizeof(__half)) {
        std::cerr << "[CUDA Preprocess] unsupported TensorRT input element size: "
                  << inputElementSize << std::endl;
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

    if (imageWidth > std::numeric_limits<int>::max() / 3) {
        std::cerr << "[CUDA Preprocess] image row is too large" << std::endl;
        return false;
    }

    const size_t packedStep = static_cast<size_t>(imageWidth) * 3;
    if (imageStep < packedStep ||
        static_cast<size_t>(imageHeight) >
            std::numeric_limits<size_t>::max() / imageStep ||
        imageStep * static_cast<size_t>(imageHeight) > imageBufferBytes ||
        imageStep > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[CUDA Preprocess] GPU image buffer is too small or has invalid step"
                  << std::endl;
        return false;
    }

    prep.originalWidth = imageWidth;
    prep.originalHeight = imageHeight;
    prep.inputWidth = config_.inputWidth;
    prep.inputHeight = config_.inputHeight;
    prep.scaleX = static_cast<float>(config_.inputWidth) / static_cast<float>(imageWidth);
    prep.scaleY = static_cast<float>(config_.inputHeight) / static_cast<float>(imageHeight);
    prep.blob.release();

    launchPreprocessKernel(
        imageDevice,
        imageWidth,
        imageHeight,
        static_cast<int>(imageStep),
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

    const cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "[CUDA Preprocess] kernel launch failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }
    return true;
}
