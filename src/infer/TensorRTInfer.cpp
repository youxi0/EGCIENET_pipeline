#include "infer/TensorRTInfer.h"
#include "utils/FileLogger.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

//模块:CUDA错误检查宏,用于定位cudaMalloc、cudaMemcpyAsync等CUDA调用错误
#define CHECK_CUDA_BOOL(call) \
    do \
    { \
        cudaError_t status = call; \
        if (status != cudaSuccess) \
        { \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl; \
            return false; \
        } \
    } while (0)

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        const std::string message = std::string("[TensorRT] ") + msg;

        // 新增日志: TensorRT内部warning/error也写入FileLogger，便于回看engine加载问题。
        if (utils::FileLogger::instance().isOpen())
        {
            if (severity <= Severity::kERROR)
            {
                utils::FileLogger::instance().error(message);
            }
            else
            {
                utils::FileLogger::instance().warning(message);
            }
        }
        else
        {
            std::cout << message << std::endl;
        }
    }
}

TensorRTInfer::TensorRTInfer(const std::string& enginePath)
    : enginePath_(enginePath)
{
}

TensorRTInfer::~TensorRTInfer()
{
    releaseBuffers();
}

bool TensorRTInfer::load()
{
    std::vector<char> engineData;

    if (!loadEngineFile(engineData))
    {
        return false;
    }

    if (!createRuntimeEngineContext(engineData))
    {
        return false;
    }

    if (!parseBindings())
    {
        return false;
    }

    if (!allocateBuffers())
    {
        return false;
    }

    return true;
}

bool TensorRTInfer::loadEngineFile(std::vector<char>& engineData)
{
    std::ifstream file(enginePath_, std::ios::binary);

    if (!file.good())
    {
        std::cerr << "Failed to open engine file: " << enginePath_ << std::endl;
        return false;
    }

    // 1. 将文件指针移动到文件末尾
    file.seekg(0, std::ifstream::end);
    // 2. 获取当前指针的位置（刚好就是文件的大小，单位是字节）
    size_t fileSize = file.tellg();
    // 3. 将文件指针重新移回文件开头，准备读取数据
    file.seekg(0, std::ifstream::beg);

    engineData.resize(fileSize);
    // 将文件里的二进制数据，原封不动地全部倒进 vector 的内存里
    file.read(engineData.data(), fileSize);
    file.close();

    std::cout << "Loaded engine file: " << enginePath_ << std::endl;
    std::cout << "Engine file size: " << fileSize << " bytes" << std::endl;

    return true;
}

bool TensorRTInfer::createRuntimeEngineContext(const std::vector<char>& engineData)
{
    //模块:createInferRuntime创建TensorRT运行时对象,用于反序列化engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_)); 

    if (!runtime_)
    {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return false;
    }
    //模块:deserializeCudaEngine把engine二进制反序列化成TensorRT推理引擎
    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size())); 

    if (!engine_)
    {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return false;
    }
    //模块:createExecutionContext创建推理上下文,真正执行推理时需要使用它
    context_.reset(engine_->createExecutionContext()); 

    if (!context_)
    {
        std::cerr << "Failed to create TensorRT execution context." << std::endl;
        return false;
    }
    //模块:cudaStreamCreate创建CUDA流,用于异步数据拷贝和推理执行
    // CPU 根本不会等待数据拷完，它会瞬间执行完这行代码，然后继续往下走（去干别的工作）。
    // 真正的搬砖工作是由 GPU 内部的 DMA（直接内存访问）控制器在后台完成的
    CHECK_CUDA_BOOL(cudaStreamCreate(&stream_)); 

    return true;
}

bool TensorRTInfer::parseBindings()
{
    //模块:getNbBindings获取engine中输入输出binding总数
    int bindingCount = engine_->getNbBindings(); 

    bindingNames_.resize(bindingCount);
    deviceBuffers_.resize(bindingCount, nullptr);
    bufferBytes_.resize(bindingCount, 0);

    std::cout << "Binding count: " << bindingCount << std::endl;

    for (int i = 0; i < bindingCount; ++i)
    {
        //模块:getBindingName获取当前binding名称
        const char* bindingName = engine_->getBindingName(i); 
        bindingNames_[i] = bindingName;

        //模块:bindingIsInput判断当前binding是输入还是输出
        bool isInput = engine_->bindingIsInput(i); 
        //模块:getBindingDimensions获取当前binding维度
        nvinfer1::Dims dims = engine_->getBindingDimensions(i); 
        //模块:getBindingDataType获取当前binding数据类型
        nvinfer1::DataType dataType = engine_->getBindingDataType(i); 

        std::cout << "Binding[" << i << "] name: " << bindingName << " type: " << (isInput ? "input" : "output") << " shape: ";

        for (int d = 0; d < dims.nbDims; ++d)
        {
            std::cout << dims.d[d] << (d + 1 == dims.nbDims ? "" : " x ");
        }

        std::cout << " dataType: " << dataTypeName(dataType)
                  << "(" << static_cast<int>(dataType) << ")" << std::endl;

        if (isInput)
        {
            inputBindingIndex_ = i;
        }
        else
        {
            outputBindingIndices_.push_back(i);
        }
    }

    if (inputBindingIndex_ < 0)
    {
        std::cerr << "No input binding found." << std::endl;
        return false;
    }

    if (outputBindingIndices_.empty())
    {
        std::cerr << "No output binding found." << std::endl;
        return false;
    }

    return true;
}

bool TensorRTInfer::allocateBuffers()
{
    int bindingCount = engine_->getNbBindings();

    for (int i = 0; i < bindingCount; ++i)
    {
        const std::string& bindingName = bindingNames_[i];
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        nvinfer1::DataType dataType = engine_->getBindingDataType(i);

        //模块:如果是输入binding,并且engine是动态shape,需要手动设置输入维度
        if (engine_->bindingIsInput(i))
        {
            if (hasDynamicDim(dims))
            {
                nvinfer1::Dims4 inputDims(1, 3, 640, 640);
                //模块:setBindingDimensions设置动态输入shape,这里固定为1x3x640x640
                bool ok = context_->setBindingDimensions(i, inputDims); 
                if (!ok)
                {
                    std::cerr << "Failed to set input binding dimensions: " << bindingName << std::endl;
                    return false;
                }

                dims = inputDims;
            }
        }

        //模块:如果是输出binding,需要从context中获取实际输出维度
        if (!engine_->bindingIsInput(i))
        {
            //模块:getBindingDimensions从context获取动态shape推导后的实际输出维度
            dims = context_->getBindingDimensions(i); 
        }

        size_t elementCount = getSizeByDim(dims);

        if (elementCount == 0)
        {
            std::cerr << "Invalid binding shape for: " << bindingName << std::endl;
            return false;
        }

        size_t elementSize = getElementSize(dataType);

        if (elementSize == 0)
        {
            std::cerr << "Unsupported binding data type for: " << bindingName << std::endl;
            return false;
        }

        size_t bytes = elementCount * elementSize;
        bufferBytes_[i] = bytes;

        //模块:cudaMalloc在GPU显存上给输入或输出binding申请空间
        CHECK_CUDA_BOOL(cudaMalloc(&deviceBuffers_[i], bytes)); 

        std::cout << "Allocate buffer for " << bindingName << ", bytes: " << bytes << std::endl;
    }

    if (!context_->allInputDimensionsSpecified())
    {
        std::cerr << "Input dimensions are not fully specified." << std::endl;
        return false;
    }

    return true;
}

void* TensorRTInfer::inputDeviceBuffer()
{
    if (inputBindingIndex_ < 0 || inputBindingIndex_ >= static_cast<int>(deviceBuffers_.size()))
    {
        return nullptr;
    }

    return deviceBuffers_[inputBindingIndex_];
}

nvinfer1::DataType TensorRTInfer::inputDataType() const
{
    if (!engine_ || inputBindingIndex_ < 0)
    {
        return nvinfer1::DataType::kFLOAT;
    }

    return engine_->getBindingDataType(inputBindingIndex_);
}

size_t TensorRTInfer::inputElementSize() const
{
    return getElementSize(inputDataType());
}

cudaStream_t TensorRTInfer::stream() {
    return stream_;
}

std::vector<cv::Mat> TensorRTInfer::forward(const cv::Mat& blob)
{
    std::vector<cv::Mat> outputs;

    if (!context_)
    {
        std::cerr << "TensorRT context is not loaded." << std::endl;
        return outputs;
    }

    if (blob.empty())
    {
        std::cerr << "Input blob is empty." << std::endl;
        return outputs;
    }

    if (blob.dims != 4)
    {
        std::cerr << "Input blob dims must be 4." << std::endl;
        return outputs;
    }

    if (blob.type() != CV_32F)
    {
        std::cerr << "Input blob type must be CV_32F." << std::endl;
        return outputs;
    }

    const nvinfer1::DataType inputType = inputDataType();
    const size_t elementCount = blob.total();
    const void* inputHostData = blob.ptr<float>();
    size_t inputBytes = elementCount * sizeof(float);
    std::vector<__half> halfInput;

    if (inputType == nvinfer1::DataType::kHALF)
    {
        // Modified: convert CPU float blob to half for FP16 TensorRT input bindings.
        halfInput.resize(elementCount);
        const float* src = blob.ptr<float>();

        for (size_t i = 0; i < elementCount; ++i)
        {
            halfInput[i] = __float2half_rn(src[i]);
        }

        inputHostData = halfInput.data();
        inputBytes = elementCount * sizeof(__half);
    }
    else if (inputType != nvinfer1::DataType::kFLOAT)
    {
        std::cerr << "Unsupported TensorRT input data type: "
                  << dataTypeName(inputType)
                  << std::endl;
        return outputs;
    }

    if (inputBytes > bufferBytes_[inputBindingIndex_])
    {
        std::cerr << "Input blob is larger than TensorRT input buffer." << std::endl;
        return outputs;
    }
    // 修改点: inputBytes按TensorRT输入binding类型计算，FP32用float字节数，FP16用half字节数。
    //模块:cudaMemcpyAsync把CPU上的blob 异步拷贝到GPU输入显存,Async（异步）
    cudaError_t status = cudaMemcpyAsync(
        deviceBuffers_[inputBindingIndex_],
        inputHostData,
        inputBytes,
        cudaMemcpyHostToDevice,// Host（CPU）到 Device（GPU）
        stream_
    ); 

    if (status != cudaSuccess)
    {
        std::cerr << "CUDA memcpy input error: " << cudaGetErrorString(status) << std::endl;
        return outputs;
    }

    //模块:enqueueV2执行TensorRT异步推理,输入输出显存地址由deviceBuffers提供
    bool inferOk = context_->enqueueV2(deviceBuffers_.data(), stream_, nullptr); 

    if (!inferOk)
    {
        std::cerr << "TensorRT enqueueV2 failed." << std::endl;
        return outputs;
    }
    return copyOutputsToHost();
}

std::vector<cv::Mat> TensorRTInfer::forwardFromDevice()
{
    std::vector<cv::Mat> outputs;

    if (!context_)
    {
        std::cerr << "TensorRT context is not loaded." << std::endl;
        return outputs;
    }

    if (inputBindingIndex_ < 0)
    {
        std::cerr << "Input binding index is invalid." << std::endl;
        return outputs;
    }

    if (!deviceBuffers_[inputBindingIndex_])
    {
        std::cerr << "Input device buffer is null." << std::endl;
        return outputs;
    }

    bool inferOk = context_->enqueueV2(deviceBuffers_.data(), stream_, nullptr);
    if (!inferOk)
    {
        std::cerr << "TensorRT enqueueV2 failed." << std::endl;
        return outputs;
    }

    return copyOutputsToHost();
}

std::vector<cv::Mat> TensorRTInfer::copyOutputsToHost()
{
    std::vector<cv::Mat> outputs;

    for (int outputIndex : outputBindingIndices_)
    {
        nvinfer1::Dims dims = context_->getBindingDimensions(outputIndex);
        nvinfer1::DataType dataType = engine_->getBindingDataType(outputIndex);

        size_t elementCount = getSizeByDim(dims);

        if (elementCount == 0)
        {
            std::cerr << "Invalid output shape: " << bindingNames_[outputIndex] << std::endl;
            continue;
        }

        std::vector<int> cvShape = dimsToCvShape(dims);
        cv::Mat outputMat(static_cast<int>(cvShape.size()), cvShape.data(), CV_32F);

        cudaError_t status = cudaSuccess;

        if (dataType == nvinfer1::DataType::kFLOAT)
        {
            status = cudaMemcpyAsync(
                outputMat.ptr<float>(),
                deviceBuffers_[outputIndex],
                elementCount * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_
            );

            if (status != cudaSuccess)
            {
                std::cerr << "CUDA memcpy output error: "
                          << cudaGetErrorString(status)
                          << std::endl;
                continue;
            }

            outputs.push_back(outputMat);
        }
        else if (dataType == nvinfer1::DataType::kHALF)
        {
            std::vector<__half> halfOutput(elementCount);

            status = cudaMemcpyAsync(
                halfOutput.data(),
                deviceBuffers_[outputIndex],
                elementCount * sizeof(__half),
                cudaMemcpyDeviceToHost,
                stream_
            );

            if (status != cudaSuccess)
            {
                std::cerr << "CUDA memcpy half output error: "
                          << cudaGetErrorString(status)
                          << std::endl;
                continue;
            }

            status = cudaStreamSynchronize(stream_);

            if (status != cudaSuccess)
            {
                std::cerr << "CUDA stream sync error: "
                          << cudaGetErrorString(status)
                          << std::endl;
                continue;
            }

            float* dst = outputMat.ptr<float>();

            for (size_t i = 0; i < elementCount; ++i)
            {
                dst[i] = __half2float(halfOutput[i]);
            }

            outputs.push_back(outputMat);
        }
        else
        {
            std::cerr << "Unsupported output data type: "
                      << bindingNames_[outputIndex]
                      << std::endl;
            continue;
        }
    }

    cudaError_t status = cudaStreamSynchronize(stream_);

    if (status != cudaSuccess)
    {
        std::cerr << "CUDA stream sync error: "
                  << cudaGetErrorString(status)
                  << std::endl;
        outputs.clear();
        return outputs;
    }

    return outputs;
}

void TensorRTInfer::releaseBuffers()
{
    for (void* buffer : deviceBuffers_)
    {
        if (buffer)
        {
            //cudaFree释放GPU显存
            cudaFree(buffer); 
        }
    }

    deviceBuffers_.clear();
    bufferBytes_.clear();

    if (stream_)
    {
        //cudaStreamDestroy释放CUDA流
        cudaStreamDestroy(stream_); 
        stream_ = nullptr;
    }
}

size_t TensorRTInfer::getSizeByDim(const nvinfer1::Dims& dims) const
{
    if (dims.nbDims <= 0)
    {
        return 0;
    }

    size_t size = 1;

    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return 0;
        }

        size *= static_cast<size_t>(dims.d[i]);
    }

    return size;
}

size_t TensorRTInfer::getElementSize(nvinfer1::DataType dataType) const
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 0;
    }
}

std::string TensorRTInfer::dataTypeName(nvinfer1::DataType dataType) const
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FP32";
    case nvinfer1::DataType::kHALF:
        return "FP16";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    default:
        return "UNKNOWN";
    }
}

bool TensorRTInfer::hasDynamicDim(const nvinfer1::Dims& dims) const
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] == -1)
        {
            return true;
        }
    }

    return false;
}

std::vector<int> TensorRTInfer::dimsToCvShape(const nvinfer1::Dims& dims) const
{
    std::vector<int> shape;

    for (int i = 0; i < dims.nbDims; ++i)
    {
        shape.push_back(static_cast<int>(dims.d[i]));
    }

    return shape;
}
