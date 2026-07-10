#pragma once

#include "infer/TensorRTCommon.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct TensorRTInferConfig {
    std::string enginePath;
    std::string inputTensorName = "image";
    std::string outputTensorName = "mask";

    // 动态 engine 使用该形状；静态 engine 也会据此校验模型契约。
    nvinfer1::Dims4 inputShape{1, 3, 352, 352};
};

// EGCINET TensorRT 推理封装。
// 一个实例持有一套执行上下文、CUDA 流和复用显存，不支持多线程并发调用。
class TensorRTInfer {
public:
    explicit TensorRTInfer(std::string enginePath);
    explicit TensorRTInfer(TensorRTInferConfig config);
    ~TensorRTInfer();

    TensorRTInfer(const TensorRTInfer&) = delete;
    TensorRTInfer& operator=(const TensorRTInfer&) = delete;
    TensorRTInfer(TensorRTInfer&&) = delete;
    TensorRTInfer& operator=(TensorRTInfer&&) = delete;

    // 反序列化 engine，并一次性分配输入、输出显存。
    bool load();
    bool isLoaded() const noexcept;

    // CPU 路径：输入必须是连续的 FP32 NCHW blob，输出为 HxW FP32 概率图。
    bool infer(const cv::Mat& inputBlob, cv::Mat& modelMask);

    // CUDA 预处理已在本实例的 stream 上写入输入显存时，直接执行推理。
    bool inferFromDevice(cv::Mat& modelMask);

    // CUDA 预处理模块通过以下接口直接写入 TensorRT 输入显存。
    void* inputDeviceBuffer() noexcept;
    size_t inputBufferBytes() const noexcept;
    nvinfer1::DataType inputDataType() const noexcept;
    size_t inputElementSize() const noexcept;
    int inputWidth() const noexcept;
    int inputHeight() const noexcept;
    cudaStream_t stream() const noexcept;

private:
    // 保存单个 binding 的运行时形状、类型和显存大小。
    struct BindingInfo {
        int index = -1;
        std::string name;
        nvinfer1::Dims dims{};
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        size_t elementCount = 0;
        size_t bytes = 0;
    };

    // 从磁盘完整读取序列化 engine。
    bool readEngineFile(std::vector<char>& engineData) const;

    // 创建 TensorRT runtime、engine、context 和非阻塞 CUDA 流。
    bool createRuntimeObjects(const std::vector<char>& engineData);

    // 按名称解析 binding，并校验 EGCINET 的输入输出契约。
    bool inspectBindings();

    // 根据实际 binding 形状一次性申请输入输出显存。
    bool allocateDeviceBuffers();

    // 校验 CPU blob 的数据类型、内存连续性和 NCHW 形状。
    bool validateInputBlob(const cv::Mat& inputBlob) const;

    // 提交 TensorRT 推理，并把单个 mask 输出拷回主机。
    bool enqueueAndCopyOutput(cv::Mat& modelMask);
    bool copyOutputToHost(cv::Mat& modelMask);

    // 按 CUDA、TensorRT 的依赖顺序释放运行时资源。
    void release() noexcept;
    void releaseDeviceBuffers() noexcept;

    // 以下工具函数用于维度和 buffer 大小的安全计算。
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

    cudaStream_t stream_ = nullptr;
    BindingInfo input_;
    BindingInfo output_;
    std::vector<void*> deviceBuffers_;

    // FP16 I/O 转换使用可复用暂存区，避免每帧反复申请内存。
    std::vector<__half> hostInputScratch_;
    std::vector<__half> hostOutputScratch_;
};
