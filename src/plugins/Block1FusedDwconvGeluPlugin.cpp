#include "plugins/Block1FusedDwconvGeluPlugin.h"

#include "plugins/Block1FusedDwconvGeluKernel.h"

#include <new>
#include <string_view>

namespace egcinet::plugins {
namespace {

constexpr int32_t kSuccess = 0;
constexpr int32_t kFailure = -1;

// 本插件只接受线性 FP16 张量，避免 TensorRT 为插件内部再选择其他格式。
bool isHalfLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kHALF &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

} // 匿名命名空间

Block1FusedDwconvGeluPlugin::Block1FusedDwconvGeluPlugin(
    int32_t height,
    int32_t width
) noexcept
    : height_(height), width_(width) {}

nvinfer1::IPluginCapability*
Block1FusedDwconvGeluPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type
) noexcept {
    // 根据 TensorRT 当前所处阶段，返回同一插件对象对应的能力接口。
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

nvinfer1::IPluginV3* Block1FusedDwconvGeluPlugin::clone() noexcept {
    // 因为该接口noexcept，如果内存分配失败，不抛出异常，而是返回 nullptr
    auto* plugin = new (std::nothrow) Block1FusedDwconvGeluPlugin(height_, width_);
    if (plugin != nullptr) {
        plugin->setPluginNamespace(namespace_.c_str());
    }
    return plugin;
}

const char* Block1FusedDwconvGeluPlugin::getPluginName() const noexcept {
    return kBlock1FusedPluginName;
}

const char* Block1FusedDwconvGeluPlugin::getPluginVersion() const noexcept {
    return kBlock1FusedPluginVersion;
}

const char* Block1FusedDwconvGeluPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void Block1FusedDwconvGeluPlugin::setPluginNamespace(
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

int32_t Block1FusedDwconvGeluPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t Block1FusedDwconvGeluPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    // 三个输入依次为input feature、DWConv weight和 bias；输出固定保持 FP16。
    if (outputTypes == nullptr || inputTypes == nullptr ||
        nbOutputs != 1 || nbInputs != 3) {
        return kFailure;
    }
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return kSuccess;
}

int32_t Block1FusedDwconvGeluPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* /* shapeInputs */,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& /* exprBuilder */
) noexcept {
    // 插件从 token-major（B,T,C） 读入并仍按 token-major 输出，因此 shape 完全不变。
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 ||
        nbOutputs != 1 || nbShapeInputs != 0 || inputs[0].nbDims != 3) {
        return kFailure;
    }
    outputs[0] = inputs[0];
    return kSuccess;
}

bool Block1FusedDwconvGeluPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    // TensorRT 会逐个询问 3 个输入和 1 个输出是否支持当前 dtype/format 组合。
    if (inOut == nullptr || nbInputs != 3 || nbOutputs != 1 ||
        pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }
    return isHalfLinear(inOut[pos].desc);
}

int32_t Block1FusedDwconvGeluPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    // 构建阶段可能仍含动态维度，这里只检查张量数量、rank 和插件固定参数。
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 ||
        nbOutputs != 1 || height_ <= 0 || width_ <= 0 ||
        inputs[0].desc.dims.nbDims != 3 ||
        inputs[1].desc.dims.nbDims != 4 ||
        inputs[2].desc.dims.nbDims != 1 ||
        outputs[0].desc.dims.nbDims != 3) {
        return kFailure;
    }
    return kSuccess;
}

bool Block1FusedDwconvGeluPlugin::validateDescriptors(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    // 步骤 1：检查接口传入的张量数量、rank 和保存的 H/W 是否有效。
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 || nbOutputs != 1 ||
        height_ <= 0 || width_ <= 0 || inputs[0].dims.nbDims != 3 ||
        inputs[1].dims.nbDims != 4 || inputs[2].dims.nbDims != 1 ||
        outputs[0].dims.nbDims != 3) {
        return false;
    }

    // 步骤 2：确认input feature、DWConv weight、bias和输出全部为线性 FP16。
    if (!isHalfLinear(inputs[0]) || !isHalfLinear(inputs[1]) ||
        !isHalfLinear(inputs[2]) || !isHalfLinear(outputs[0])) {
        return false;
    }

    const int32_t tokens = inputs[0].dims.d[1];
    const int32_t channels = inputs[0].dims.d[2];
    // 步骤 3：input feature必须是 [B,H*W,C]，DWConv weight必须是 [C,1,3,3]。
    if (inputs[0].dims.d[0] <= 0 || tokens != height_ * width_ || channels <= 0) {
        return false;
    }
    if (inputs[1].dims.d[0] != channels || inputs[1].dims.d[1] != 1 ||
        inputs[1].dims.d[2] != 3 || inputs[1].dims.d[3] != 3 ||
        inputs[2].dims.d[0] != channels) {
        return false;
    }

    // 步骤 4：插件前后不改变 token-major shape，逐维检查输出与激活一致。
    for (int32_t index = 0; index < 3; ++index) {
        if (outputs[0].dims.d[index] != inputs[0].dims.d[index]) {
            return false;
        }
    }
    return true;
}

int32_t Block1FusedDwconvGeluPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    // execution context 获得确定 shape 后，再进行一次完整运行时校验。
    return validateDescriptors(inputs, nbInputs, outputs, nbOutputs)
        ? kSuccess
        : kFailure;
}

size_t Block1FusedDwconvGeluPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* /* inputs */,
    int32_t /* nbInputs */,
    const nvinfer1::DynamicPluginTensorDesc* /* outputs */,
    int32_t /* nbOutputs */
) const noexcept {
    // 融合 kernel 不需要 TensorRT 额外分配临时 workspace。
    return 0;
}

int32_t Block1FusedDwconvGeluPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /* workspace */,
    cudaStream_t stream
) noexcept {
    // 步骤 1：enqueue 位于执行热路径，只做必要的描述符和指针校验。
    if (!validateDescriptors(inputDesc, 3, outputDesc, 1) ||
        inputs == nullptr || outputs == nullptr) {
        return kFailure;
    }

    // 步骤 2：把 TensorRT 的三个设备输入直接交给 CUDA 启动函数；
    // 启动函数在传入 stream 上异步执行，不做 cudaStreamSynchronize。
    return launchBlock1FusedDwconvGelu(
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[0],
        inputDesc[0].dims.d[0],
        height_,
        width_,
        inputDesc[0].dims.d[2],
        stream
    );
}

nvinfer1::IPluginV3* Block1FusedDwconvGeluPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /* context */
) noexcept {
    // 当前实现没有 context 私有资源，返回普通 clone 即可。
    return clone();
}

const nvinfer1::PluginFieldCollection*
Block1FusedDwconvGeluPlugin::getFieldsToSerialize() noexcept {
    try {
        // Engine 反序列化时只需恢复 block1 的固定 H/W；权重仍由 engine 张量保存。
        serializedFields_.clear();
        serializedFields_.emplace_back(
            "height", &height_, nvinfer1::PluginFieldType::kINT32, 1
        );
        serializedFields_.emplace_back(
            "width", &width_, nvinfer1::PluginFieldType::kINT32, 1
        );
        serializedFieldCollection_.nbFields =
            static_cast<int32_t>(serializedFields_.size());
        serializedFieldCollection_.fields = serializedFields_.data();
        return &serializedFieldCollection_;
    } catch (...) {
        return nullptr;
    }
}

Block1FusedDwconvGeluPluginCreator::Block1FusedDwconvGeluPluginCreator() noexcept {
    try {
        // 向 ONNX parser 声明插件节点允许接收的两个整型属性。
        fields_.emplace_back(
            "height", nullptr, nvinfer1::PluginFieldType::kINT32, 1
        );
        fields_.emplace_back(
            "width", nullptr, nvinfer1::PluginFieldType::kINT32, 1
        );
        fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
        fieldCollection_.fields = fields_.data();
    } catch (...) {
        fields_.clear();
        fieldCollection_ = {};
    }
}

const char* Block1FusedDwconvGeluPluginCreator::getPluginName() const noexcept {
    return kBlock1FusedPluginName;
}

const char* Block1FusedDwconvGeluPluginCreator::getPluginVersion() const noexcept {
    return kBlock1FusedPluginVersion;
}

const nvinfer1::PluginFieldCollection*
Block1FusedDwconvGeluPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

const char* Block1FusedDwconvGeluPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void Block1FusedDwconvGeluPluginCreator::setPluginNamespace(
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

nvinfer1::IPluginV3* Block1FusedDwconvGeluPluginCreator::createPlugin(
    const char* /* name */,
    const nvinfer1::PluginFieldCollection* fieldCollection,
    nvinfer1::TensorRTPhase /* phase */
) noexcept {
    // 步骤 1：从构建期 ONNX 属性或运行期序列化字段中读取 H/W。
    if (fieldCollection == nullptr || fieldCollection->fields == nullptr) {
        return nullptr;
    }

    int32_t height = 0;
    int32_t width = 0;
    for (int32_t index = 0; index < fieldCollection->nbFields; ++index) {
        const nvinfer1::PluginField& field = fieldCollection->fields[index];
        if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
            field.type != nvinfer1::PluginFieldType::kINT32) {
            continue;
        }

        const auto value = *static_cast<const int32_t*>(field.data);
        const std::string_view fieldName(field.name);
        if (fieldName == "height") {
            height = value;
        } else if (fieldName == "width") {
            width = value;
        }
    }

    if (height <= 0 || width <= 0) {
        return nullptr;
    }

    // 步骤 2：创建完整的 IPluginV3 对象，并保持 Creator 的命名空间一致。
    auto* plugin = new (std::nothrow) Block1FusedDwconvGeluPlugin(height, width);
    if (plugin != nullptr) {
        plugin->setPluginNamespace(namespace_.c_str());
    }
    return plugin;
}

// 动态库被 dlopen 或 trtexec --dynamicPlugins 加载时，静态注册 Creator。
REGISTER_TENSORRT_PLUGIN(Block1FusedDwconvGeluPluginCreator);

} // namespace egcinet::plugins
