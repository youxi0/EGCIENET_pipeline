#pragma once

#include "infer/TensorRTCommon.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct TensorRTInferConfig {
    std::string enginePath;
    std::string inputTensorName = "image";
    std::string outputTensorName = "mask";

    // 动态 engine 使用该形状设置输入 binding；静态 engine 据此校验输入形状。
    nvinfer1::Dims4 inputShape{1, 3, 352, 352};
};

// EGCINET TensorRT 执行封装。
// 本类只拥有 TensorRT runtime、engine、context 和 binding 元数据，
// 不创建 CUDA stream，也不申请或释放输入输出显存。
class TensorRTInfer {
public:
    explicit TensorRTInfer(std::string enginePath);
    explicit TensorRTInfer(TensorRTInferConfig config);
    ~TensorRTInfer();

    TensorRTInfer(const TensorRTInfer&) = delete;
    TensorRTInfer& operator=(const TensorRTInfer&) = delete;
    TensorRTInfer(TensorRTInfer&&) = delete;
    TensorRTInfer& operator=(TensorRTInfer&&) = delete;

    // 反序列化 engine、创建 execution context，并解析输入输出 binding。
    bool load();
    bool isLoaded() const noexcept;

    // 为单图和验证工具挂接调用方拥有的显存与 stream；本类不接管所有权。
    bool attachExecutionResources(
        void* inputDevice,
        size_t inputBufferBytes,
        void* outputDevice,
        size_t outputBufferBytes,
        cudaStream_t stream
    ) noexcept;

    // 清除非拥有资源引用，不同步 stream，也不释放显存。
    void detachExecutionResources() noexcept;

    // 使用已挂接资源异步提交推理，不执行同步或 D2H。
    bool inferFromDevice();

    // Pipeline 路径：使用调用方给定的输入输出显存和 stream 异步提交推理。
    // 调用方负责前置依赖、buffer 生命周期以及销毁 context 前的 stream 同步。
    bool inferFromDeviceBuffers(
        void* inputDevice,
        size_t inputBufferBytes,
        void* outputDevice,
        size_t outputBufferBytes,
        cudaStream_t stream
    );

    // 以下指针和 stream 来自 attachExecutionResources，本类不拥有它们。
    void* inputDeviceBuffer() noexcept;
    void* outputDeviceBuffer() noexcept;
    const void* outputDeviceBuffer() const noexcept;
    cudaStream_t stream() const noexcept;

    size_t inputBufferBytes() const noexcept;
    nvinfer1::DataType inputDataType() const noexcept;
    size_t inputElementSize() const noexcept;
    int inputWidth() const noexcept;
    int inputHeight() const noexcept;

    size_t outputBufferBytes() const noexcept;
    nvinfer1::DataType outputDataType() const noexcept;
    size_t outputElementSize() const noexcept;
    int outputWidth() const noexcept;
    int outputHeight() const noexcept;

private:
    // 保存单个 binding 的运行时形状、类型和所需显存大小。
    struct BindingInfo {
        int index = -1;
        std::string name;
        nvinfer1::Dims dims{};
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        size_t elementCount = 0;
        size_t bytes = 0;
    };

    bool readEngineFile(std::vector<char>& engineData) const;
    bool createRuntimeObjects(const std::vector<char>& engineData);
    bool inspectBindings();

    // 使用栈上的 binding 指针表提交推理，不进行资源申请、同步或 D2H。
    bool enqueue(
        void* inputDevice,
        void* outputDevice,
        cudaStream_t stream
    );

    // 仅释放本类拥有的 TensorRT 对象，不接触调用方 CUDA 资源。
    void release() noexcept;

    static bool hasDynamicDimension(const nvinfer1::Dims& dims) noexcept;
    static bool sameDimensions(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) noexcept;
    static size_t elementCount(const nvinfer1::Dims& dims) noexcept;
    static size_t elementSize(nvinfer1::DataType dataType) noexcept;

private:
    TensorRTInferConfig config_;
    TrtLogger logger_;

    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context_;

    BindingInfo input_;
    BindingInfo output_;

    // 以下资源均为非拥有引用，仅用于兼容单图和验证工具。
    void* attachedInputDevice_ = nullptr;
    void* attachedOutputDevice_ = nullptr;
    cudaStream_t attachedStream_ = nullptr;
};
