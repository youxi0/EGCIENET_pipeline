#pragma once

#include "common/ScopeTimer.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

//模块:TensorRT日志类,用于接收TensorRT内部构建和推理日志
class TrtLogger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

namespace detail {

// 修改点: TensorRT不同版本的对象释放接口不完全一致。
// 旧接口对象通常提供destroy()，但IOptimizationProfile等新接口对象没有destroy()。
// 这里在编译期检测destroy()是否存在，避免给没有destroy()的类型实例化错误代码。
template <typename T, typename = void>
struct HasTensorRTDestroy : std::false_type {};

template <typename T>
struct HasTensorRTDestroy<T, std::void_t<decltype(std::declval<T*>()->destroy())>> : std::true_type {};

} // namespace detail

//模块:TensorRT对象释放器,用于自动释放runtime、engine、context以及builder阶段对象
template <typename T>
class TrtDestroy{
public:
    void operator()(T* obj) const
    {
        if (!obj) {
            return;
        }

        // 修改点: 有destroy()的TensorRT对象继续调用destroy()；没有destroy()的对象使用delete释放。
        if constexpr (detail::HasTensorRTDestroy<T>::value) {
            obj->destroy();
        } else {
            delete obj;
        }
    }
};

//模块:TensorRT推理类,负责加载engine、申请显存、执行推理、拷贝输出
class TensorRTInfer{
public:
    explicit TensorRTInfer(const std::string& enginePath);
    ~TensorRTInfer();

    bool load();
    //输入CPU上的blob,内部完成H2D拷贝、TensorRT推理、D2H输出拷贝
    std::vector<cv::Mat> forward(const cv::Mat& blob);

    //输入buffer已经在GPU上准备好时使用,不再执行输入H2D拷贝
    std::vector<cv::Mat> forwardFromDevice();
    
    // 修改点: CUDA前处理会直接写TensorRT输入buffer；这里返回void*以兼容FP32/FP16输入binding。
    void* inputDeviceBuffer();

    // 修改点: 暴露输入binding的数据类型和元素字节数，避免FP16 baseline按float错误写入。
    nvinfer1::DataType inputDataType() const;
    size_t inputElementSize() const;

    //作用:CUDA前处理和TensorRT推理共用同一个stream,保证执行顺序
    cudaStream_t stream();

private:
    //读取 .engine 文件到内存
    bool loadEngineFile(std::vector<char>& engineData);

    // 创建 TensorRT Runtime、反序列化 engine、创建 ExecutionContext
    // Runtime 负责加载 engine，Context 负责执行具体推理
    bool createRuntimeEngineContext(const std::vector<char>& engineData);

    // 解析 engine 的输入输出 binding。TensorRT 8 里输入输出都叫 binding
    bool parseBindings();

    // 根据输入输出 shape 在 GPU 上申请显存
    // 输入 blob 要拷贝到输入显存，推理结果会写到输出显存
    bool allocateBuffers();
    void releaseBuffers();

    //从GPU输出buffer拷贝结果到CPU cv::Mat
    std::vector<cv::Mat> copyOutputsToHost();

    // nvinfer1::Dims：nbDims：维度总数（比如一张图片是 NCHW 四个维度，那 nbDims 就是 4）。
    //                 d：一个整型数组，存放着具体每一个维度的长度（比如 d[0]=1, d[1]=3, d[2]=224, d[3]=224）。
    // 根据TensorRT的Dims计算张量元素数量
    size_t getSizeByDim(const nvinfer1::Dims& dims) const;
    // 根据TensorRT的数据类型计算每个元素占多少字节
    size_t getElementSize(nvinfer1::DataType dataType) const;
    std::string dataTypeName(nvinfer1::DataType dataType) const;
    // 判断engine是不是动态shape
    bool hasDynamicDim(const nvinfer1::Dims& dims) const;
    // 把TensorRT的Dims转换成OpenCV能用的shape格式
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
    //在 GPU 显存上为binding提前挖好的坑（通常通过 cudaMalloc 申请）
    std::vector<void*> deviceBuffers_;
    std::vector<size_t> bufferBytes_;
      
};
