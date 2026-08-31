#include "plugins/FusedSpatialReductionPlugin.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

namespace egcinet::plugins {

int32_t launchFusedInt8SpatialReduction(
    int32_t stage,
    int32_t int8Mode,
    bool floatInput,
    float activationScale,
    const void* input,
    const void* weight,
    const void* bias,
    const void* dequantScale,
    void* output,
    cudaStream_t stream
) noexcept;

namespace {

constexpr int32_t kSuccess = 0;
constexpr int32_t kFailure = -1;
constexpr char kInt8InputLayout[] = "BNC_INT8";
constexpr char kFusedQuantizeInputLayout[] = "BNC_FP_PACK_INT8";
constexpr char kOutputLayout[] = "BNC_FP16";
constexpr char kInt8PackedWeightLayout[] =
    "CO_KHKW_CI_ROW_MAJOR_INT8";

bool isSupportedParameters(
    const FusedSpatialReductionParameters& parameters
) noexcept {
    if (parameters.groups != 1 || parameters.fuseLayerNorm != 0 ||
        parameters.outputHeight != 11 || parameters.outputWidth != 11 ||
        parameters.inputChannels != parameters.outputChannels ||
        parameters.packedN != parameters.outputChannels ||
        parameters.packedK !=
            parameters.kernelHeight * parameters.kernelWidth *
            parameters.inputChannels ||
        parameters.layerIndex < 0 || parameters.int8Mode < 1 ||
        parameters.int8Mode > 2 ||
        (parameters.int8Mode == 2 &&
         (!std::isfinite(parameters.activationScale) ||
          parameters.activationScale <= 0.0F))) {
        return false;
    }

    switch (parameters.stage) {
    case 1:
        return parameters.int8Mode == 2 &&
               parameters.inputHeight == 88 &&
               parameters.inputWidth == 88 &&
               parameters.inputChannels == 64 &&
               parameters.kernelHeight == 8 &&
               parameters.kernelWidth == 8 &&
               parameters.strideHeight == 8 &&
               parameters.strideWidth == 8 &&
               parameters.packedK == 4096;
    case 2:
        return parameters.int8Mode == 1 &&
               parameters.inputHeight == 44 &&
               parameters.inputWidth == 44 &&
               parameters.inputChannels == 128 &&
               parameters.kernelHeight == 4 &&
               parameters.kernelWidth == 4 &&
               parameters.strideHeight == 4 &&
               parameters.strideWidth == 4 &&
               parameters.packedK == 2048;
    case 3:
        return parameters.int8Mode == 1 &&
               parameters.inputHeight == 22 &&
               parameters.inputWidth == 22 &&
               parameters.inputChannels == 320 &&
               parameters.kernelHeight == 2 &&
               parameters.kernelWidth == 2 &&
               parameters.strideHeight == 2 &&
               parameters.strideWidth == 2 &&
               parameters.packedK == 1280;
    default:
        return false;
    }
}

bool isHalfLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kHALF &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isInt8Linear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kINT8 &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isFloatLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kFLOAT &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isHalfOrFloatLinear(
    const nvinfer1::PluginTensorDesc& descriptor
) noexcept {
    return descriptor.format == nvinfer1::PluginFormat::kLINEAR &&
           (descriptor.type == nvinfer1::DataType::kHALF ||
            descriptor.type == nvinfer1::DataType::kFLOAT);
}

bool matchesDimension(
    int32_t actual,
    int32_t expected,
    bool allowDynamic
) noexcept {
    return actual == expected || (allowDynamic && actual == -1);
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

bool readScalarFloat(
    const nvinfer1::PluginField& field,
    float& value
) noexcept {
    if (field.data == nullptr || field.length != 1 ||
        field.type != nvinfer1::PluginFieldType::kFLOAT32) {
        return false;
    }
    value = *static_cast<const float*>(field.data);
    return true;
}

bool readString(
    const nvinfer1::PluginField& field,
    std::string& value
) {
    if (field.data == nullptr || field.length < 0 ||
        field.type != nvinfer1::PluginFieldType::kCHAR) {
        return false;
    }
    const auto* source = static_cast<const char*>(field.data);
    value.assign(source, source + field.length);
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return true;
}

} // namespace

FusedSpatialReductionPlugin::FusedSpatialReductionPlugin(
    FusedSpatialReductionParameters parameters
)
    : parameters_(parameters) {
    if (!isSupportedParameters(parameters_)) {
        throw std::invalid_argument(
            "unsupported fused spatial reduction parameters"
        );
    }
}

FusedSpatialReductionPlugin::~FusedSpatialReductionPlugin() noexcept = default;

nvinfer1::IPluginCapability*
FusedSpatialReductionPlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* FusedSpatialReductionPlugin::clone() noexcept {
    try {
        auto* plugin = new FusedSpatialReductionPlugin(parameters_);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] clone failed: %s\n",
            error.what()
        );
        return nullptr;
    } catch (...) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] clone failed: "
            "unknown exception\n"
        );
        return nullptr;
    }
}

const char* FusedSpatialReductionPlugin::getPluginName() const noexcept {
    return kFusedSpatialReductionPluginName;
}

const char* FusedSpatialReductionPlugin::getPluginVersion() const noexcept {
    return kFusedSpatialReductionPluginVersion;
}

const char* FusedSpatialReductionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void FusedSpatialReductionPlugin::setPluginNamespace(
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

int32_t FusedSpatialReductionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t FusedSpatialReductionPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    constexpr int32_t expectedInputs = 4;
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 ||
        nbInputs != expectedInputs) {
        return kFailure;
    }
    const bool inputTypeValid = parameters_.int8Mode == 1
        ? inputTypes[0] == nvinfer1::DataType::kINT8
        : inputTypes[0] == nvinfer1::DataType::kHALF ||
              inputTypes[0] == nvinfer1::DataType::kFLOAT;
    if (!inputTypeValid || inputTypes[1] != nvinfer1::DataType::kINT8 ||
        inputTypes[2] != nvinfer1::DataType::kHALF ||
        inputTypes[3] != nvinfer1::DataType::kFLOAT) {
        return kFailure;
    }
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return kSuccess;
}

int32_t FusedSpatialReductionPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* /* shapeInputs */,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    constexpr int32_t expectedInputs = 4;
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedInputs ||
        nbOutputs != 1 || nbShapeInputs != 0 || inputs[0].nbDims != 3 ||
        inputs[1].nbDims != 2 || inputs[2].nbDims != 1 ||
        inputs[3].nbDims != 1) {
        return kFailure;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(
        parameters_.outputHeight * parameters_.outputWidth
    );
    outputs[0].d[2] = exprBuilder.constant(parameters_.outputChannels);
    return kSuccess;
}

bool FusedSpatialReductionPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    constexpr int32_t expectedInputs = 4;
    if (inOut == nullptr || nbInputs != expectedInputs || nbOutputs != 1 ||
        pos < 0 ||
        pos >= nbInputs + nbOutputs) {
        return false;
    }
    switch (pos) {
    case 0:
        return parameters_.int8Mode == 1
            ? isInt8Linear(inOut[pos].desc)
            : isHalfOrFloatLinear(inOut[pos].desc);
    case 1:
        return isInt8Linear(inOut[pos].desc);
    case 2:
    case 4:
        return isHalfLinear(inOut[pos].desc);
    case 3:
        return isFloatLinear(inOut[pos].desc);
    default:
        return false;
    }
}

bool FusedSpatialReductionPlugin::validateDescriptors(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs,
    bool allowDynamic
) const noexcept {
    constexpr int32_t expectedInputs = 4;
    if (inputs == nullptr || outputs == nullptr ||
        nbInputs != expectedInputs ||
        nbOutputs != 1 ||
        inputs[0].dims.nbDims != 3 || inputs[1].dims.nbDims != 2 ||
        inputs[2].dims.nbDims != 1 || outputs[0].dims.nbDims != 3 ||
        inputs[3].dims.nbDims != 1) {
        return false;
    }
    const bool int8InputValid = parameters_.int8Mode == 1
        ? isInt8Linear(inputs[0])
        : isHalfOrFloatLinear(inputs[0]);
    const bool formatsValid = int8InputValid && isInt8Linear(inputs[1]) &&
        isHalfLinear(inputs[2]) && isFloatLinear(inputs[3]);
    if (!formatsValid || !isHalfLinear(outputs[0])) {
        return false;
    }

    return
        matchesDimension(inputs[0].dims.d[0], 1, allowDynamic) &&
        matchesDimension(
            inputs[0].dims.d[1],
            parameters_.inputHeight * parameters_.inputWidth,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[0].dims.d[2], parameters_.inputChannels, allowDynamic
        ) &&
        matchesDimension(
            inputs[1].dims.d[0],
            parameters_.packedN,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[1].dims.d[1],
            parameters_.packedK,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[2].dims.d[0], parameters_.packedN, allowDynamic
        ) &&
        matchesDimension(
            inputs[3].dims.d[0], parameters_.packedN, allowDynamic
        ) &&
        matchesDimension(outputs[0].dims.d[0], 1, allowDynamic) &&
        matchesDimension(
            outputs[0].dims.d[1],
            parameters_.outputHeight * parameters_.outputWidth,
            allowDynamic
        ) &&
        matchesDimension(
            outputs[0].dims.d[2], parameters_.outputChannels, allowDynamic
        );
}

int32_t FusedSpatialReductionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    if (inputs == nullptr || outputs == nullptr) {
        return kFailure;
    }
    nvinfer1::PluginTensorDesc inputDescriptors[4]{
        inputs[0].desc,
        inputs[1].desc,
        inputs[2].desc,
        {},
    };
    if (nbInputs == 4) {
        inputDescriptors[3] = inputs[3].desc;
    }
    return validateDescriptors(
               inputDescriptors,
               nbInputs,
               &outputs[0].desc,
               nbOutputs,
               true
           )
        ? kSuccess
        : kFailure;
}

int32_t FusedSpatialReductionPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    return validateDescriptors(inputs, nbInputs, outputs, nbOutputs, false)
        ? kSuccess
        : kFailure;
}

size_t FusedSpatialReductionPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* /* inputs */,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* /* outputs */,
    int32_t nbOutputs
) const noexcept {
    (void)nbInputs;
    (void)nbOutputs;
    return 0;
}

int32_t FusedSpatialReductionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /* workspace */,
    cudaStream_t stream
) noexcept {
    constexpr int32_t expectedInputs = 4;
    if (!validateDescriptors(
            inputDesc, expectedInputs, outputDesc, 1, false
        ) ||
        inputs == nullptr || outputs == nullptr ||
        inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr ||
        inputs[3] == nullptr ||
        outputs[0] == nullptr) {
        return kFailure;
    }

    return launchFusedInt8SpatialReduction(
        parameters_.stage,
        parameters_.int8Mode,
        inputDesc[0].type == nvinfer1::DataType::kFLOAT,
        parameters_.activationScale,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        outputs[0],
        stream
    );
}

nvinfer1::IPluginV3* FusedSpatialReductionPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /* context */
) noexcept {
    return clone();
}

const nvinfer1::PluginFieldCollection*
FusedSpatialReductionPlugin::getFieldsToSerialize() noexcept {
    try {
        serializedFields_.clear();
        auto addInt = [this](const char* name, int32_t* value) {
            serializedFields_.emplace_back(
                name, value, nvinfer1::PluginFieldType::kINT32, 1
            );
        };
        addInt("stage", &parameters_.stage);
        addInt("layer_index", &parameters_.layerIndex);
        addInt("input_height", &parameters_.inputHeight);
        addInt("input_width", &parameters_.inputWidth);
        addInt("input_channels", &parameters_.inputChannels);
        addInt("output_height", &parameters_.outputHeight);
        addInt("output_width", &parameters_.outputWidth);
        addInt("output_channels", &parameters_.outputChannels);
        addInt("kernel_height", &parameters_.kernelHeight);
        addInt("kernel_width", &parameters_.kernelWidth);
        addInt("stride_height", &parameters_.strideHeight);
        addInt("stride_width", &parameters_.strideWidth);
        addInt("groups", &parameters_.groups);
        addInt("packed_k", &parameters_.packedK);
        addInt("packed_n", &parameters_.packedN);
        addInt("fuse_layernorm", &parameters_.fuseLayerNorm);
        addInt("int8_mode", &parameters_.int8Mode);
        serializedFields_.emplace_back(
            "activation_scale",
            &parameters_.activationScale,
            nvinfer1::PluginFieldType::kFLOAT32,
            1
        );
        const char* inputLayout = parameters_.int8Mode == 1
            ? kInt8InputLayout
            : kFusedQuantizeInputLayout;
        serializedFields_.emplace_back(
            "input_layout",
            inputLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(std::string_view(inputLayout).size())
        );
        serializedFields_.emplace_back(
            "output_layout",
            kOutputLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(sizeof(kOutputLayout) - 1)
        );
        serializedFields_.emplace_back(
            "packed_weight_layout",
            kInt8PackedWeightLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(
                std::string_view(kInt8PackedWeightLayout).size()
            )
        );
        serializedFieldCollection_.nbFields =
            static_cast<int32_t>(serializedFields_.size());
        serializedFieldCollection_.fields = serializedFields_.data();
        return &serializedFieldCollection_;
    } catch (...) {
        return nullptr;
    }
}

FusedSpatialReductionPluginCreator::
FusedSpatialReductionPluginCreator() noexcept {
    try {
        constexpr const char* kIntegerFields[] = {
            "stage",
            "layer_index",
            "input_height",
            "input_width",
            "input_channels",
            "output_height",
            "output_width",
            "output_channels",
            "kernel_height",
            "kernel_width",
            "stride_height",
            "stride_width",
            "groups",
            "packed_k",
            "packed_n",
            "fuse_layernorm",
            "int8_mode",
        };
        for (const char* name : kIntegerFields) {
            fields_.emplace_back(
                name, nullptr, nvinfer1::PluginFieldType::kINT32, 1
            );
        }
        fields_.emplace_back(
            "activation_scale",
            nullptr,
            nvinfer1::PluginFieldType::kFLOAT32,
            1
        );
        constexpr const char* kStringFields[] = {
            "input_layout",
            "output_layout",
            "packed_weight_layout",
        };
        for (const char* name : kStringFields) {
            fields_.emplace_back(
                name, nullptr, nvinfer1::PluginFieldType::kCHAR, 0
            );
        }
        fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
        fieldCollection_.fields = fields_.data();
    } catch (...) {
        fields_.clear();
        fieldCollection_ = {};
    }
}

const char* FusedSpatialReductionPluginCreator::getPluginName() const noexcept {
    return kFusedSpatialReductionPluginName;
}

const char* FusedSpatialReductionPluginCreator::getPluginVersion() const noexcept {
    return kFusedSpatialReductionPluginVersion;
}

const nvinfer1::PluginFieldCollection*
FusedSpatialReductionPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

const char*
FusedSpatialReductionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void FusedSpatialReductionPluginCreator::setPluginNamespace(
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

nvinfer1::IPluginV3* FusedSpatialReductionPluginCreator::createPlugin(
    const char* /* name */,
    const nvinfer1::PluginFieldCollection* fieldCollection,
    nvinfer1::TensorRTPhase /* phase */
) noexcept {
    if (fieldCollection == nullptr || fieldCollection->fields == nullptr) {
        return nullptr;
    }

    try {
        FusedSpatialReductionParameters parameters{};
        std::string inputLayout;
        std::string outputLayout;
        std::string weightLayout;
        uint32_t seen = 0;

        for (int32_t index = 0; index < fieldCollection->nbFields; ++index) {
            const nvinfer1::PluginField& field = fieldCollection->fields[index];
            if (field.name == nullptr) {
                continue;
            }
            const std::string_view name(field.name);
            int32_t* destination = nullptr;
            uint32_t bit = 0;
            if (name == "stage") {
                destination = &parameters.stage; bit = 1U << 0;
            } else if (name == "layer_index") {
                destination = &parameters.layerIndex; bit = 1U << 1;
            } else if (name == "input_height") {
                destination = &parameters.inputHeight; bit = 1U << 2;
            } else if (name == "input_width") {
                destination = &parameters.inputWidth; bit = 1U << 3;
            } else if (name == "input_channels") {
                destination = &parameters.inputChannels; bit = 1U << 4;
            } else if (name == "output_height") {
                destination = &parameters.outputHeight; bit = 1U << 5;
            } else if (name == "output_width") {
                destination = &parameters.outputWidth; bit = 1U << 6;
            } else if (name == "output_channels") {
                destination = &parameters.outputChannels; bit = 1U << 7;
            } else if (name == "kernel_height") {
                destination = &parameters.kernelHeight; bit = 1U << 8;
            } else if (name == "kernel_width") {
                destination = &parameters.kernelWidth; bit = 1U << 9;
            } else if (name == "stride_height") {
                destination = &parameters.strideHeight; bit = 1U << 10;
            } else if (name == "stride_width") {
                destination = &parameters.strideWidth; bit = 1U << 11;
            } else if (name == "groups") {
                destination = &parameters.groups; bit = 1U << 12;
            } else if (name == "packed_k") {
                destination = &parameters.packedK; bit = 1U << 13;
            } else if (name == "packed_n") {
                destination = &parameters.packedN; bit = 1U << 14;
            } else if (name == "fuse_layernorm") {
                destination = &parameters.fuseLayerNorm; bit = 1U << 15;
            } else if (name == "int8_mode") {
                destination = &parameters.int8Mode; bit = 1U << 16;
            }

            if (destination != nullptr) {
                if (!readScalarInt32(field, *destination)) {
                    return nullptr;
                }
                seen |= bit;
            } else if (name == "activation_scale") {
                if (!readScalarFloat(field, parameters.activationScale)) {
                    return nullptr;
                }
            } else if (name == "input_layout") {
                if (!readString(field, inputLayout)) {
                    return nullptr;
                }
            } else if (name == "output_layout") {
                if (!readString(field, outputLayout)) {
                    return nullptr;
                }
            } else if (name == "packed_weight_layout") {
                if (!readString(field, weightLayout)) {
                    return nullptr;
                }
            }
        }

        constexpr uint32_t kRequiredIntegerFields = (1U << 16) - 1U;
        const char* expectedInputLayout = parameters.int8Mode == 1
            ? kInt8InputLayout
            : kFusedQuantizeInputLayout;
        if ((seen & kRequiredIntegerFields) != kRequiredIntegerFields ||
            (!inputLayout.empty() && inputLayout != expectedInputLayout) ||
            (!outputLayout.empty() && outputLayout != kOutputLayout) ||
            (!weightLayout.empty() &&
             weightLayout != kInt8PackedWeightLayout)) {
            return nullptr;
        }

        auto* plugin = new FusedSpatialReductionPlugin(parameters);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] createPlugin failed: %s\n",
            error.what()
        );
        return nullptr;
    } catch (...) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] createPlugin failed: "
            "unknown exception\n"
        );
        return nullptr;
    }
}

REGISTER_TENSORRT_PLUGIN(FusedSpatialReductionPluginCreator);

} // namespace egcinet::plugins
