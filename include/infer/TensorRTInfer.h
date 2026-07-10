#pragma once

#include "infer/TensorRTCommon.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// TensorRT 运行时封装：加载序列化 engine 并执行推理。
class TensorRTInfer {
public:
    explicit TensorRTInfer(const std::string& enginePath);
    ~TensorRTInfer();

    // 加载 runtime，反序列化 engine，创建执行上下文并分配 binding buffer。
    bool load();

    // 将 CPU 端 FP32 NCHW blob 拷贝到输入 binding，执行推理并拷回输出。
    std::vector<cv::Mat> forward(const cv::Mat& blob);

    // 输入 binding 已由 GPU 预处理写好时，直接执行推理。
    std::vector<cv::Mat> forwardFromDevice();

    // 返回 TensorRT 输入端 device buffer，供 GPU 预处理直接写入。
    void* inputDeviceBuffer();

    nvinfer1::DataType inputDataType() const;
    size_t inputElementSize() const;
    cudaStream_t stream();

private:
    bool loadEngineFile(std::vector<char>& engineData);
    bool createRuntimeEngineContext(const std::vector<char>& engineData);
    bool parseBindings();
    bool allocateBuffers();
    void releaseBuffers();

    std::vector<cv::Mat> copyOutputsToHost();

    size_t getSizeByDim(const nvinfer1::Dims& dims) const;
    size_t getElementSize(nvinfer1::DataType dataType) const;
    std::string dataTypeName(nvinfer1::DataType dataType) const;
    bool hasDynamicDim(const nvinfer1::Dims& dims) const;
    std::vector<int> dimsToCvShape(const nvinfer1::Dims& dims) const;

private:
    std::string enginePath_;
    TrtLogger logger_;

    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context_;

    cudaStream_t stream_ = nullptr;

    int inputBindingIndex_ = -1;
    std::vector<int> outputBindingIndices_;
    std::vector<std::string> bindingNames_;
    std::vector<void*> deviceBuffers_;
    std::vector<size_t> bufferBytes_;
};
