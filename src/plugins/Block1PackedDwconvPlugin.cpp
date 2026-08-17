#include "plugins/Block1PackedDwconvPlugin.h"

#include "plugins/Block1PackedDwconvKernel.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace egcinet::plugins {

struct Block1PackedDwconvHostParameters {
    std::vector<int32_t> packedWeights;
    std::vector<int32_t> packedBias;
};

struct Block1PackedDwconvDeviceParameters {
    void* packedWeights = nullptr;
    void* packedBias = nullptr;

    ~Block1PackedDwconvDeviceParameters() noexcept {
        if (packedWeights != nullptr) {
            cudaFree(packedWeights);
        }
        if (packedBias != nullptr) {
            cudaFree(packedBias);
        }
    }
};

namespace {

constexpr int32_t kSuccess = 0;
constexpr int32_t kFailure = -1;

bool isHalfLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kHALF &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool readScalarInt32(
    const nvinfer1::PluginField& field,
    int32_t& value
) noexcept {
    if (field.data == nullptr || field.length != 1) {
        return false;
    }
    if (field.type == nvinfer1::PluginFieldType::kINT32) {
        value = *static_cast<const int32_t*>(field.data);
        return true;
    }
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        const int64_t source = *static_cast<const int64_t*>(field.data);
        if (source < std::numeric_limits<int32_t>::min() ||
            source > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        value = static_cast<int32_t>(source);
        return true;
    }
    return false;
}

bool readPackedInt32Words(
    const nvinfer1::PluginField& field,
    std::vector<int32_t>& values
) {
    if (field.data == nullptr || field.length <= 0) {
        return false;
    }

    values.resize(static_cast<size_t>(field.length));
    if (field.type == nvinfer1::PluginFieldType::kINT32) {
        const auto* source = static_cast<const int32_t*>(field.data);
        for (int32_t index = 0; index < field.length; ++index) {
            values[static_cast<size_t>(index)] = source[index];
        }
        return true;
    }
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        const auto* source = static_cast<const int64_t*>(field.data);
        for (int32_t index = 0; index < field.length; ++index) {
            if (source[index] < std::numeric_limits<int32_t>::min() ||
                source[index] > std::numeric_limits<int32_t>::max()) {
                values.clear();
                return false;
            }
            values[static_cast<size_t>(index)] =
                static_cast<int32_t>(source[index]);
        }
        return true;
    }

    values.clear();
    return false;
}

void allocateAndCopy(void** destination, const std::vector<int32_t>& source) {
    const size_t bytes = source.size() * sizeof(int32_t);
    if (cudaMalloc(destination, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for packed DWConv parameters");
    }
    if (cudaMemcpy(*destination, source.data(), bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed for packed DWConv parameters");
    }
}

} // namespace

Block1PackedDwconvPlugin::Block1PackedDwconvPlugin(
    int32_t height,
    int32_t width,
    int32_t channels,
    std::vector<int32_t> packedWeights,
    std::vector<int32_t> packedBias
)
    : height_(height), width_(width), channels_(channels) {
    if (height_ <= 0 || width_ <= 0 || channels_ <= 0 ||
        (channels_ & 1) != 0) {
        throw std::invalid_argument(
            "packed DWConv requires positive H/W and an even channel count"
        );
    }

    const size_t channelPairs = static_cast<size_t>(channels_ / 2);
    if (packedWeights.size() != 9U * channelPairs ||
        packedBias.size() != channelPairs) {
        throw std::invalid_argument("packed DWConv parameter length mismatch");
    }

    // 步骤 1：保存序列化所需的 host 位模式。每个 int32_t 对应一个 half2。
    auto hostParameters = std::make_shared<Block1PackedDwconvHostParameters>();
    hostParameters->packedWeights = std::move(packedWeights);
    hostParameters->packedBias = std::move(packedBias);

    // 步骤 2：Plugin 创建时一次性上传；enqueue 热路径只读取设备指针。
    auto deviceParameters =
        std::make_shared<Block1PackedDwconvDeviceParameters>();
    allocateAndCopy(
        &deviceParameters->packedWeights,
        hostParameters->packedWeights
    );
    allocateAndCopy(&deviceParameters->packedBias, hostParameters->packedBias);

    hostParameters_ = std::move(hostParameters);
    deviceParameters_ = std::move(deviceParameters);
}

Block1PackedDwconvPlugin::Block1PackedDwconvPlugin(
    int32_t height,
    int32_t width,
    int32_t channels,
    std::shared_ptr<const Block1PackedDwconvHostParameters> hostParameters,
    std::shared_ptr<const Block1PackedDwconvDeviceParameters> deviceParameters
) noexcept
    : height_(height),
      width_(width),
      channels_(channels),
      hostParameters_(std::move(hostParameters)),
      deviceParameters_(std::move(deviceParameters)) {}

nvinfer1::IPluginCapability*
Block1PackedDwconvPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type
) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    default:
        return nullptr;
    }
}

nvinfer1::IPluginV3* Block1PackedDwconvPlugin::clone() noexcept {
    auto* plugin = new (std::nothrow) Block1PackedDwconvPlugin(
        height_,
        width_,
        channels_,
        hostParameters_,
        deviceParameters_
    );
    if (plugin != nullptr) {
        plugin->setPluginNamespace(namespace_.c_str());
    }
    return plugin;
}

const char* Block1PackedDwconvPlugin::getPluginName() const noexcept {
    return kBlock1PackedDwconvPluginName;
}

const char* Block1PackedDwconvPlugin::getPluginVersion() const noexcept {
    return kBlock1PackedDwconvPluginVersion;
}

const char* Block1PackedDwconvPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void Block1PackedDwconvPlugin::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    if (pluginNamespace == nullptr) {
        return;
    }
    try {
        namespace_ = pluginNamespace;
    } catch (...) {
    }
}

int32_t Block1PackedDwconvPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t Block1PackedDwconvPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 ||
        nbInputs != 1 || inputTypes[0] != nvinfer1::DataType::kHALF) {
        return kFailure;
    }
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return kSuccess;
}

int32_t Block1PackedDwconvPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* /* shapeInputs */,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& /* exprBuilder */
) noexcept {
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 ||
        nbOutputs != 1 || nbShapeInputs != 0 || inputs[0].nbDims != 3) {
        return kFailure;
    }
    outputs[0] = inputs[0];
    return kSuccess;
}

bool Block1PackedDwconvPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1 ||
        pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    return isHalfLinear(inOut[pos].desc);
}

int32_t Block1PackedDwconvPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 ||
        nbOutputs != 1 || height_ <= 0 || width_ <= 0 || channels_ <= 0 ||
        (channels_ & 1) != 0 || inputs[0].desc.dims.nbDims != 3 ||
        outputs[0].desc.dims.nbDims != 3 || hostParameters_ == nullptr ||
        deviceParameters_ == nullptr) {
        return kFailure;
    }

    const int32_t tokens = inputs[0].desc.dims.d[1];
    const int32_t channels = inputs[0].desc.dims.d[2];
    if ((tokens >= 0 && tokens != height_ * width_) ||
        (channels >= 0 && channels != channels_)) {
        return kFailure;
    }
    return kSuccess;
}

bool Block1PackedDwconvPlugin::validateDescriptors(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 ||
        nbOutputs != 1 || inputs[0].dims.nbDims != 3 ||
        outputs[0].dims.nbDims != 3 || hostParameters_ == nullptr ||
        deviceParameters_ == nullptr || deviceParameters_->packedWeights == nullptr ||
        deviceParameters_->packedBias == nullptr) {
        return false;
    }
    if (!isHalfLinear(inputs[0]) || !isHalfLinear(outputs[0])) {
        return false;
    }
    if (inputs[0].dims.d[0] <= 0 ||
        inputs[0].dims.d[1] != height_ * width_ ||
        inputs[0].dims.d[2] != channels_) {
        return false;
    }
    for (int32_t index = 0; index < 3; ++index) {
        if (outputs[0].dims.d[index] != inputs[0].dims.d[index]) {
            return false;
        }
    }
    return true;
}

int32_t Block1PackedDwconvPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    return validateDescriptors(inputs, nbInputs, outputs, nbOutputs)
        ? kSuccess
        : kFailure;
}

size_t Block1PackedDwconvPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* /* inputs */,
    int32_t /* nbInputs */,
    const nvinfer1::DynamicPluginTensorDesc* /* outputs */,
    int32_t /* nbOutputs */
) const noexcept {
    return 0;
}

int32_t Block1PackedDwconvPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /* workspace */,
    cudaStream_t stream
) noexcept {
    if (!validateDescriptors(inputDesc, 1, outputDesc, 1) ||
        inputs == nullptr || outputs == nullptr || inputs[0] == nullptr ||
        outputs[0] == nullptr) {
        return kFailure;
    }

    return launchBlock1PackedDwconv(
        inputs[0],
        deviceParameters_->packedWeights,
        deviceParameters_->packedBias,
        outputs[0],
        inputDesc[0].dims.d[0],
        height_,
        width_,
        channels_,
        stream
    );
}

nvinfer1::IPluginV3* Block1PackedDwconvPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /* context */
) noexcept {
    // 设备参数只读，所有 execution context 可以安全共享同一份缓冲。
    return clone();
}

const nvinfer1::PluginFieldCollection*
Block1PackedDwconvPlugin::getFieldsToSerialize() noexcept {
    if (hostParameters_ == nullptr) {
        return nullptr;
    }
    try {
        serializedFields_.clear();
        serializedFields_.emplace_back(
            "height", &height_, nvinfer1::PluginFieldType::kINT32, 1
        );
        serializedFields_.emplace_back(
            "width", &width_, nvinfer1::PluginFieldType::kINT32, 1
        );
        serializedFields_.emplace_back(
            "channels", &channels_, nvinfer1::PluginFieldType::kINT32, 1
        );
        serializedFields_.emplace_back(
            "packed_weights",
            hostParameters_->packedWeights.data(),
            nvinfer1::PluginFieldType::kINT32,
            static_cast<int32_t>(hostParameters_->packedWeights.size())
        );
        serializedFields_.emplace_back(
            "packed_bias",
            hostParameters_->packedBias.data(),
            nvinfer1::PluginFieldType::kINT32,
            static_cast<int32_t>(hostParameters_->packedBias.size())
        );
        serializedFieldCollection_.nbFields =
            static_cast<int32_t>(serializedFields_.size());
        serializedFieldCollection_.fields = serializedFields_.data();
        return &serializedFieldCollection_;
    } catch (...) {
        return nullptr;
    }
}

Block1PackedDwconvPluginCreator::Block1PackedDwconvPluginCreator() noexcept {
    try {
        fields_.emplace_back(
            "height", nullptr, nvinfer1::PluginFieldType::kINT32, 1
        );
        fields_.emplace_back(
            "width", nullptr, nvinfer1::PluginFieldType::kINT32, 1
        );
        fields_.emplace_back(
            "channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1
        );
        fields_.emplace_back(
            "packed_weights", nullptr, nvinfer1::PluginFieldType::kINT32, 0
        );
        fields_.emplace_back(
            "packed_bias", nullptr, nvinfer1::PluginFieldType::kINT32, 0
        );
        fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
        fieldCollection_.fields = fields_.data();
    } catch (...) {
        fields_.clear();
        fieldCollection_ = {};
    }
}

const char* Block1PackedDwconvPluginCreator::getPluginName() const noexcept {
    return kBlock1PackedDwconvPluginName;
}

const char* Block1PackedDwconvPluginCreator::getPluginVersion() const noexcept {
    return kBlock1PackedDwconvPluginVersion;
}

const nvinfer1::PluginFieldCollection*
Block1PackedDwconvPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

const char* Block1PackedDwconvPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void Block1PackedDwconvPluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    if (pluginNamespace == nullptr) {
        return;
    }
    try {
        namespace_ = pluginNamespace;
    } catch (...) {
    }
}

nvinfer1::IPluginV3* Block1PackedDwconvPluginCreator::createPlugin(
    const char* /* name */,
    const nvinfer1::PluginFieldCollection* fieldCollection,
    nvinfer1::TensorRTPhase /* phase */
) noexcept {
    if (fieldCollection == nullptr || fieldCollection->fields == nullptr) {
        return nullptr;
    }

    try {
        int32_t height = 0;
        int32_t width = 0;
        int32_t channels = 0;
        std::vector<int32_t> packedWeights;
        std::vector<int32_t> packedBias;

        for (int32_t index = 0; index < fieldCollection->nbFields; ++index) {
            const nvinfer1::PluginField& field = fieldCollection->fields[index];
            if (field.name == nullptr) {
                continue;
            }
            const std::string_view fieldName(field.name);
            if (fieldName == "height") {
                if (!readScalarInt32(field, height)) {
                    return nullptr;
                }
            } else if (fieldName == "width") {
                if (!readScalarInt32(field, width)) {
                    return nullptr;
                }
            } else if (fieldName == "channels") {
                if (!readScalarInt32(field, channels)) {
                    return nullptr;
                }
            } else if (fieldName == "packed_weights") {
                if (!readPackedInt32Words(field, packedWeights)) {
                    return nullptr;
                }
            } else if (fieldName == "packed_bias") {
                if (!readPackedInt32Words(field, packedBias)) {
                    return nullptr;
                }
            }
        }

        auto* plugin = new Block1PackedDwconvPlugin(
            height,
            width,
            channels,
            std::move(packedWeights),
            std::move(packedBias)
        );
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

REGISTER_TENSORRT_PLUGIN(Block1PackedDwconvPluginCreator);

} // 命名空间 egcinet::plugins
