#include "infer/TensorRTInfer.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>

namespace {

const char* dataTypeName(nvinfer1::DataType dataType) noexcept {
    switch (dataType) {
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

void printDimensions(const nvinfer1::Dims& dims) {
    for (int index = 0; index < dims.nbDims; ++index) {
        if (index != 0) {
            std::cout << 'x';
        }
        std::cout << dims.d[index];
    }
}

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "[TensorRT Infer] " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

} // namespace

TensorRTInfer::TensorRTInfer(std::string enginePath)
    : TensorRTInfer(TensorRTInferConfig{std::move(enginePath)}) {}

TensorRTInfer::TensorRTInfer(TensorRTInferConfig config)
    : config_(std::move(config)) {}

TensorRTInfer::~TensorRTInfer() {
    release();
}

bool TensorRTInfer::load() {
    release();

    if (config_.enginePath.empty()) {
        std::cerr << "[TensorRT Infer] engine path is empty" << std::endl;
        return false;
    }

    std::vector<char> engineData;
    if (!readEngineFile(engineData) ||
        !createRuntimeObjects(engineData) ||
        !inspectBindings() ||
        !allocateDeviceBuffers()) {
        release();
        return false;
    }

    std::cout << "[TensorRT Infer] engine loaded: " << config_.enginePath << std::endl;
    std::cout << "[TensorRT Infer] input " << input_.name << " [";
    printDimensions(input_.dims);
    std::cout << "] " << dataTypeName(input_.dataType) << std::endl;
    std::cout << "[TensorRT Infer] output " << output_.name << " [";
    printDimensions(output_.dims);
    std::cout << "] " << dataTypeName(output_.dataType) << std::endl;
    return true;
}

bool TensorRTInfer::isLoaded() const noexcept {
    return context_ != nullptr && stream_ != nullptr &&
           input_.index >= 0 && output_.index >= 0;
}

bool TensorRTInfer::readEngineFile(std::vector<char>& engineData) const {
    std::ifstream file(config_.enginePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[TensorRT Infer] failed to open engine: "
                  << config_.enginePath << std::endl;
        return false;
    }

    const std::streamoff fileSize = file.tellg();
    if (fileSize <= 0 ||
        static_cast<unsigned long long>(fileSize) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        std::cerr << "[TensorRT Infer] invalid engine file size" << std::endl;
        return false;
    }

    engineData.resize(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), static_cast<std::streamsize>(fileSize));

    if (!file) {
        std::cerr << "[TensorRT Infer] failed to read complete engine file" << std::endl;
        engineData.clear();
        return false;
    }

    return true;
}

bool TensorRTInfer::createRuntimeObjects(const std::vector<char>& engineData) {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "[TensorRT Infer] failed to create runtime" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine_) {
        std::cerr << "[TensorRT Infer] failed to deserialize engine" << std::endl;
        return false;
    }

    if (engine_->hasImplicitBatchDimension()) {
        std::cerr << "[TensorRT Infer] implicit-batch engine is not supported" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[TensorRT Infer] failed to create execution context" << std::endl;
        return false;
    }

    return checkCuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags"
    );
}

bool TensorRTInfer::inspectBindings() {
    if (engine_->getNbBindings() != 2) {
        std::cerr << "[TensorRT Infer] expected one input and one output, got "
                  << engine_->getNbBindings() << " bindings" << std::endl;
        return false;
    }

    input_.index = engine_->getBindingIndex(config_.inputTensorName.c_str());
    output_.index = engine_->getBindingIndex(config_.outputTensorName.c_str());

    if (input_.index < 0 || output_.index < 0) {
        std::cerr << "[TensorRT Infer] binding name mismatch, expected input='"
                  << config_.inputTensorName << "', output='"
                  << config_.outputTensorName << "'" << std::endl;
        return false;
    }

    if (!engine_->bindingIsInput(input_.index) ||
        engine_->bindingIsInput(output_.index)) {
        std::cerr << "[TensorRT Infer] input/output binding role mismatch" << std::endl;
        return false;
    }

    nvinfer1::Dims engineInputDims = engine_->getBindingDimensions(input_.index);
    if (hasDynamicDimension(engineInputDims)) {
        if (!context_->setBindingDimensions(input_.index, config_.inputShape)) {
            std::cerr << "[TensorRT Infer] failed to set dynamic input shape" << std::endl;
            return false;
        }
    } else if (!sameDimensions(engineInputDims, config_.inputShape)) {
        std::cerr << "[TensorRT Infer] engine input shape does not match configured shape" << std::endl;
        return false;
    }

    if (!context_->allInputDimensionsSpecified()) {
        std::cerr << "[TensorRT Infer] input dimensions are not fully specified" << std::endl;
        return false;
    }

    input_.name = engine_->getBindingName(input_.index);
    input_.dims = context_->getBindingDimensions(input_.index);
    input_.dataType = engine_->getBindingDataType(input_.index);

    output_.name = engine_->getBindingName(output_.index);
    output_.dims = context_->getBindingDimensions(output_.index);
    output_.dataType = engine_->getBindingDataType(output_.index);

    // 当前预处理和后处理都按固定的 NCHW 单批次分割模型设计。
    if (input_.dims.nbDims != 4 || input_.dims.d[0] != 1 || input_.dims.d[1] != 3) {
        std::cerr << "[TensorRT Infer] input must be [1,3,H,W]" << std::endl;
        return false;
    }

    if (output_.dims.nbDims != 4 || output_.dims.d[0] != 1 || output_.dims.d[1] != 1) {
        std::cerr << "[TensorRT Infer] output must be [1,1,H,W]" << std::endl;
        return false;
    }

    if (input_.dataType != nvinfer1::DataType::kFLOAT &&
        input_.dataType != nvinfer1::DataType::kHALF) {
        std::cerr << "[TensorRT Infer] unsupported input type: "
                  << dataTypeName(input_.dataType) << std::endl;
        return false;
    }

    if (output_.dataType != nvinfer1::DataType::kFLOAT &&
        output_.dataType != nvinfer1::DataType::kHALF) {
        std::cerr << "[TensorRT Infer] unsupported output type: "
                  << dataTypeName(output_.dataType) << std::endl;
        return false;
    }

    input_.elementCount = elementCount(input_.dims);
    output_.elementCount = elementCount(output_.dims);
    input_.bytes = input_.elementCount * elementSize(input_.dataType);
    output_.bytes = output_.elementCount * elementSize(output_.dataType);

    if (input_.bytes == 0 || output_.bytes == 0) {
        std::cerr << "[TensorRT Infer] invalid input or output buffer size" << std::endl;
        return false;
    }

    return true;
}

bool TensorRTInfer::allocateDeviceBuffers() {
    deviceBuffers_.assign(static_cast<size_t>(engine_->getNbBindings()), nullptr);

    if (!checkCuda(
            cudaMalloc(&deviceBuffers_[input_.index], input_.bytes),
            "cudaMalloc input")) {
        return false;
    }

    if (!checkCuda(
            cudaMalloc(&deviceBuffers_[output_.index], output_.bytes),
            "cudaMalloc output")) {
        return false;
    }

    return true;
}

bool TensorRTInfer::infer(const cv::Mat& inputBlob, cv::Mat& modelMask) {
    if (!validateInputBlob(inputBlob)) {
        modelMask.release();
        return false;
    }

    const void* hostInput = inputBlob.ptr<float>();
    if (input_.dataType == nvinfer1::DataType::kHALF) {
        hostInputScratch_.resize(input_.elementCount);
        __half* halfInput = hostInputScratch_.data();
        const float* floatInput = inputBlob.ptr<float>();

        for (size_t index = 0; index < input_.elementCount; ++index) {
            halfInput[index] = __float2half_rn(floatInput[index]);
        }
        hostInput = halfInput;
    }

    if (!checkCuda(
            cudaMemcpyAsync(
                deviceBuffers_[input_.index],
                hostInput,
                input_.bytes,
                cudaMemcpyHostToDevice,
                stream_
            ),
            "cudaMemcpyAsync input H2D")) {
        modelMask.release();
        return false;
    }

    return enqueueAndCopyOutput(modelMask);
}

bool TensorRTInfer::inferFromDevice(cv::Mat& modelMask) {
    if (!isLoaded() || deviceBuffers_[input_.index] == nullptr) {
        std::cerr << "[TensorRT Infer] engine is not loaded" << std::endl;
        modelMask.release();
        return false;
    }

    // 预处理与推理共用同一个 CUDA 流，enqueue 会自然等待前面的预处理完成。
    return enqueueAndCopyOutput(modelMask);
}

bool TensorRTInfer::validateInputBlob(const cv::Mat& inputBlob) const {
    if (!isLoaded()) {
        std::cerr << "[TensorRT Infer] engine is not loaded" << std::endl;
        return false;
    }

    if (inputBlob.empty() || inputBlob.type() != CV_32F ||
        inputBlob.dims != input_.dims.nbDims || !inputBlob.isContinuous()) {
        std::cerr << "[TensorRT Infer] input must be a continuous FP32 NCHW blob" << std::endl;
        return false;
    }

    for (int index = 0; index < input_.dims.nbDims; ++index) {
        if (inputBlob.size[index] != input_.dims.d[index]) {
            std::cerr << "[TensorRT Infer] input blob shape mismatch" << std::endl;
            return false;
        }
    }

    return true;
}

bool TensorRTInfer::enqueueAndCopyOutput(cv::Mat& modelMask) {
    if (!context_->enqueueV2(deviceBuffers_.data(), stream_, nullptr)) {
        std::cerr << "[TensorRT Infer] enqueueV2 failed" << std::endl;
        modelMask.release();
        return false;
    }

    return copyOutputToHost(modelMask);
}

bool TensorRTInfer::copyOutputToHost(cv::Mat& modelMask) {
    const int outputHeight = output_.dims.d[2];
    const int outputWidth = output_.dims.d[3];
    modelMask.create(outputHeight, outputWidth, CV_32F);

    if (output_.dataType == nvinfer1::DataType::kFLOAT) {
        if (!checkCuda(
                cudaMemcpyAsync(
                    modelMask.ptr<float>(),
                    deviceBuffers_[output_.index],
                    output_.bytes,
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "cudaMemcpyAsync output D2H") ||
            !checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) {
            modelMask.release();
            return false;
        }
        return true;
    }

    hostOutputScratch_.resize(output_.elementCount);
    if (!checkCuda(
            cudaMemcpyAsync(
                hostOutputScratch_.data(),
                deviceBuffers_[output_.index],
                output_.bytes,
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "cudaMemcpyAsync FP16 output D2H") ||
        !checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) {
        modelMask.release();
        return false;
    }

    const __half* halfOutput = hostOutputScratch_.data();
    float* floatOutput = modelMask.ptr<float>();
    for (size_t index = 0; index < output_.elementCount; ++index) {
        floatOutput[index] = __half2float(halfOutput[index]);
    }

    return true;
}

void* TensorRTInfer::inputDeviceBuffer() noexcept {
    if (!isLoaded() || input_.index >= static_cast<int>(deviceBuffers_.size())) {
        return nullptr;
    }
    return deviceBuffers_[input_.index];
}

size_t TensorRTInfer::inputBufferBytes() const noexcept {
    return isLoaded() ? input_.bytes : 0;
}

nvinfer1::DataType TensorRTInfer::inputDataType() const noexcept {
    return isLoaded() ? input_.dataType : nvinfer1::DataType::kFLOAT;
}

size_t TensorRTInfer::inputElementSize() const noexcept {
    return isLoaded() ? elementSize(input_.dataType) : 0;
}

int TensorRTInfer::inputWidth() const noexcept {
    return isLoaded() ? input_.dims.d[3] : 0;
}

int TensorRTInfer::inputHeight() const noexcept {
    return isLoaded() ? input_.dims.d[2] : 0;
}

cudaStream_t TensorRTInfer::stream() const noexcept {
    return stream_;
}

void TensorRTInfer::release() noexcept {
    if (stream_ != nullptr) {
        // 等待指定 Stream 里此前提交的工作完成
        cudaStreamSynchronize(stream_);
    }

    releaseDeviceBuffers();

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    context_.reset();
    engine_.reset();
    runtime_.reset();

    input_ = BindingInfo{};
    output_ = BindingInfo{};
    hostInputScratch_.clear();
    hostOutputScratch_.clear();
}

void TensorRTInfer::releaseDeviceBuffers() noexcept {
    for (void* buffer : deviceBuffers_) {
        if (buffer != nullptr) {
            cudaFree(buffer);
        }
    }
    deviceBuffers_.clear();
}

bool TensorRTInfer::hasDynamicDimension(const nvinfer1::Dims& dims) noexcept {
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] < 0) {
            return true;
        }
    }
    return false;
}

bool TensorRTInfer::sameDimensions(
    const nvinfer1::Dims& lhs,
    const nvinfer1::Dims& rhs
) noexcept {
    if (lhs.nbDims != rhs.nbDims) {
        return false;
    }

    for (int index = 0; index < lhs.nbDims; ++index) {
        if (lhs.d[index] != rhs.d[index]) {
            return false;
        }
    }
    return true;
}

size_t TensorRTInfer::elementCount(const nvinfer1::Dims& dims) noexcept {
    if (dims.nbDims <= 0) {
        return 0;
    }

    size_t count = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0 ||
            count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dims.d[index])) {
            return 0;
        }
        count *= static_cast<size_t>(dims.d[index]);
    }
    return count;
}

size_t TensorRTInfer::elementSize(nvinfer1::DataType dataType) noexcept {
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(__half);
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    default:
        return 0;
    }
}
