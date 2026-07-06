#pragma once

#include "infer/TensorRTCommon.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Runtime wrapper for loading a serialized TensorRT engine and executing inference.
class TensorRTInfer {
public:
    explicit TensorRTInfer(const std::string& enginePath);
    ~TensorRTInfer();

    // Loads runtime, deserializes engine, creates execution context, and allocates bindings.
    bool load();

    // Copies a CPU FP32 NCHW blob to the input binding, runs inference, and copies outputs back.
    std::vector<cv::Mat> forward(const cv::Mat& blob);

    // Runs inference when the input binding has already been filled on device.
    std::vector<cv::Mat> forwardFromDevice();

    // Returns the TensorRT input device buffer for GPU-side preprocessing.
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
