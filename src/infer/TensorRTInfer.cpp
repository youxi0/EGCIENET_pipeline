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

} // 匿名命名空间

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
        !inspectIOTensors()) {
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
    return context_ != nullptr && !input_.name.empty() && !output_.name.empty();
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

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[TensorRT Infer] failed to create execution context" << std::endl;
        return false;
    }
    return true;
}

bool TensorRTInfer::inspectIOTensors() {
    const int32_t ioTensorCount = engine_->getNbIOTensors();
    if (ioTensorCount != 2) {
        std::cerr << "[TensorRT Infer] expected one input and one output, got "
                  << ioTensorCount << " I/O tensors" << std::endl;
        return false;
    }

    if (engine_->getTensorIOMode(config_.inputTensorName.c_str()) !=
            nvinfer1::TensorIOMode::kINPUT ||
        engine_->getTensorIOMode(config_.outputTensorName.c_str()) !=
            nvinfer1::TensorIOMode::kOUTPUT) {
        std::cerr << "[TensorRT Infer] tensor name or role mismatch, expected input='"
                  << config_.inputTensorName << "', output='"
                  << config_.outputTensorName << "'" << std::endl;
        return false;
    }

    input_.name = config_.inputTensorName;
    output_.name = config_.outputTensorName;

    if (engine_->getTensorLocation(input_.name.c_str()) !=
            nvinfer1::TensorLocation::kDEVICE ||
        engine_->getTensorLocation(output_.name.c_str()) !=
            nvinfer1::TensorLocation::kDEVICE) {
        std::cerr << "[TensorRT Infer] input and output tensors must use device memory"
                  << std::endl;
        return false;
    }

    const nvinfer1::Dims engineInputDims =
        engine_->getTensorShape(input_.name.c_str());
    if (engineInputDims.nbDims <= 0) {
        std::cerr << "[TensorRT Infer] failed to query engine input shape" << std::endl;
        return false;
    }

    if (hasDynamicDimension(engineInputDims)) {
        if (!context_->setInputShape(input_.name.c_str(), config_.inputShape)) {
            std::cerr << "[TensorRT Infer] failed to set dynamic input shape" << std::endl;
            return false;
        }
    } else if (!sameDimensions(engineInputDims, config_.inputShape)) {
        std::cerr << "[TensorRT Infer] engine input shape does not match configured shape" << std::endl;
        return false;
    }

    // TensorRT 10 的名称接口通过 inferShapes 检查动态尺寸是否已完整指定。
    const int32_t shapeStatus = context_->inferShapes(0, nullptr);
    if (shapeStatus < 0) {
        std::cerr << "[TensorRT Infer] shape inference failed" << std::endl;
        return false;
    }
    if (shapeStatus > 0) {
        std::cerr << "[TensorRT Infer] shape inference has "
                  << shapeStatus << " insufficient input tensors" << std::endl;
        return false;
    }

    input_.dims = context_->getTensorShape(input_.name.c_str());
    input_.dataType = engine_->getTensorDataType(input_.name.c_str());
    output_.dims = context_->getTensorShape(output_.name.c_str());
    output_.dataType = engine_->getTensorDataType(output_.name.c_str());

    // 当前预处理和后处理都按固定的 NCHW 单批次分割模型设计。
    if (input_.dims.nbDims != 4 || input_.dims.d[0] != 1 || input_.dims.d[1] != 3) {
        std::cerr << "[TensorRT Infer] input must be [1,3,H,W]" << std::endl;
        return false;
    }

    if (output_.dims.nbDims != 4 || output_.dims.d[0] != 1 || output_.dims.d[1] != 1) {
        std::cerr << "[TensorRT Infer] output must be [1,1,H,W]" << std::endl;
        return false;
    }

    if (hasDynamicDimension(input_.dims) || hasDynamicDimension(output_.dims)) {
        std::cerr << "[TensorRT Infer] unresolved runtime tensor dimensions" << std::endl;
        return false;
    }

    if (input_.dims.d[2] > std::numeric_limits<int>::max() ||
        input_.dims.d[3] > std::numeric_limits<int>::max() ||
        output_.dims.d[2] > std::numeric_limits<int>::max() ||
        output_.dims.d[3] > std::numeric_limits<int>::max()) {
        std::cerr << "[TensorRT Infer] tensor height or width exceeds int range"
                  << std::endl;
        return false;
    }

    if (engine_->getTensorFormat(input_.name.c_str()) !=
            nvinfer1::TensorFormat::kLINEAR ||
        engine_->getTensorFormat(output_.name.c_str()) !=
            nvinfer1::TensorFormat::kLINEAR) {
        std::cerr << "[TensorRT Infer] only linear input/output tensor format is supported"
                  << std::endl;
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
    const size_t inputElementBytes = elementSize(input_.dataType);
    const size_t outputElementBytes = elementSize(output_.dataType);

    if (input_.elementCount == 0 || output_.elementCount == 0 ||
        inputElementBytes == 0 || outputElementBytes == 0 ||
        input_.elementCount > std::numeric_limits<size_t>::max() / inputElementBytes ||
        output_.elementCount > std::numeric_limits<size_t>::max() / outputElementBytes) {
        std::cerr << "[TensorRT Infer] invalid input or output buffer size" << std::endl;
        return false;
    }

    input_.bytes = input_.elementCount * inputElementBytes;
    output_.bytes = output_.elementCount * outputElementBytes;
    return true;
}

bool TensorRTInfer::attachExecutionResources(
    void* inputDevice,
    size_t inputBufferBytes,
    void* outputDevice,
    size_t outputBufferBytes,
    cudaStream_t stream
) noexcept {
    if (!isLoaded() || inputDevice == nullptr || outputDevice == nullptr ||
        stream == nullptr || inputBufferBytes < input_.bytes ||
        outputBufferBytes < output_.bytes) {
        return false;
    }

    attachedInputDevice_ = inputDevice;
    attachedOutputDevice_ = outputDevice;
    attachedStream_ = stream;
    return true;
}

void TensorRTInfer::detachExecutionResources() noexcept {
    attachedInputDevice_ = nullptr;
    attachedOutputDevice_ = nullptr;
    attachedStream_ = nullptr;
}

bool TensorRTInfer::inferFromDevice() {
    if (!isLoaded() || attachedInputDevice_ == nullptr ||
        attachedOutputDevice_ == nullptr || attachedStream_ == nullptr) {
        std::cerr << "[TensorRT Infer] execution resources are not attached" << std::endl;
        return false;
    }
    return enqueue(attachedInputDevice_, attachedOutputDevice_, attachedStream_);
}

bool TensorRTInfer::inferFromDeviceBuffers(
    void* inputDevice,
    size_t inputBufferBytes,
    void* outputDevice,
    size_t outputBufferBytes,
    cudaStream_t stream
) {
    if (!isLoaded() || inputDevice == nullptr || outputDevice == nullptr ||
        stream == nullptr) {
        std::cerr << "[TensorRT Infer] invalid external device buffers or stream"
                  << std::endl;
        return false;
    }
    if (inputBufferBytes < input_.bytes || outputBufferBytes < output_.bytes) {
        std::cerr << "[TensorRT Infer] external device buffer is too small" << std::endl;
        return false;
    }

    return enqueue(inputDevice, outputDevice, stream);
}

bool TensorRTInfer::enqueue(
    void* inputDevice,
    void* outputDevice,
    cudaStream_t stream
) {
    if (!context_->setTensorAddress(input_.name.c_str(), inputDevice) ||
        !context_->setTensorAddress(output_.name.c_str(), outputDevice)) {
        std::cerr << "[TensorRT Infer] failed to set I/O tensor addresses" << std::endl;
        return false;
    }

    if (!context_->enqueueV3(stream)) {
        std::cerr << "[TensorRT Infer] enqueueV3 failed" << std::endl;
        return false;
    }

    return true;
}

void* TensorRTInfer::inputDeviceBuffer() noexcept {
    return attachedInputDevice_;
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
    return isLoaded() ? static_cast<int>(input_.dims.d[3]) : 0;
}

int TensorRTInfer::inputHeight() const noexcept {
    return isLoaded() ? static_cast<int>(input_.dims.d[2]) : 0;
}

void* TensorRTInfer::outputDeviceBuffer() noexcept {
    return attachedOutputDevice_;
}

const void* TensorRTInfer::outputDeviceBuffer() const noexcept {
    return attachedOutputDevice_;
}

size_t TensorRTInfer::outputBufferBytes() const noexcept {
    return isLoaded() ? output_.bytes : 0;
}

nvinfer1::DataType TensorRTInfer::outputDataType() const noexcept {
    return isLoaded() ? output_.dataType : nvinfer1::DataType::kFLOAT;
}

size_t TensorRTInfer::outputElementSize() const noexcept {
    return isLoaded() ? elementSize(output_.dataType) : 0;
}

int TensorRTInfer::outputWidth() const noexcept {
    return isLoaded() ? static_cast<int>(output_.dims.d[3]) : 0;
}

int TensorRTInfer::outputHeight() const noexcept {
    return isLoaded() ? static_cast<int>(output_.dims.d[2]) : 0;
}

cudaStream_t TensorRTInfer::stream() const noexcept {
    return attachedStream_;
}

void TensorRTInfer::release() noexcept {
    detachExecutionResources();
    context_.reset();
    engine_.reset();
    runtime_.reset();

    input_ = TensorInfo{};
    output_ = TensorInfo{};
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
