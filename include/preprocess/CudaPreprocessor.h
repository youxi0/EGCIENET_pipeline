#pragma once

#include "preprocess/Preprocessor.h"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cstddef>

//模块:CUDA前处理
//作用:将CPU原图拷贝到GPU,并在GPU上完成letterbox、BGR转RGB、归一化、HWC转CHW
//输出:直接写入TensorRT输入GPU buffer
class CudaPreprocessor {
public:
    CudaPreprocessor(int inputW, int inputH);
    ~CudaPreprocessor();

    bool process(
        const cv::Mat& image,
        // 修改点: inputDevice可能是float*或half*，由inputElementSize决定写入格式。
        void* inputDevice,
        size_t inputElementSize,
        PreprocessResult& prep,
        cudaStream_t stream
    );

private:
    bool ensureImageBuffer(size_t bytes);
    void release();

private:
    int inputW_ = 352;
    int inputH_ = 352;

   
    unsigned char* imageDevice_ = nullptr;
    size_t imageDeviceBytes_ = 0;
};
