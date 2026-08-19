#pragma once

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#if NV_TENSORRT_MAJOR < 10
#error "Block1PackedDwconvPlugin requires TensorRT 10 or newer."
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace egcinet::plugins {

inline constexpr char kBlock1PackedDwconvPluginName[] =
    "EGCINET_Block1PackedDwconv";
inline constexpr char kBlock1PackedDwconvPluginVersion[] = "1";
inline constexpr char kBlock1PackedDwconvPluginNamespace[] = "";

struct Block1PackedDwconvHostParameters;
struct Block1PackedDwconvDeviceParameters;

// 单输入 IPluginV3：计算 token-major DWConv，并可按 ONNX 的 fuse_gelu 字段
// 选择融合 FastGELU。V11 未提供该字段，默认保持仅 DWConv 的兼容行为。
// 权重和 bias 已经由 ONNX 改图脚本按 half2 访问顺序打包，插件创建时一次性
// 上传到设备；clone 和 execution context 只共享不可变参数，不再复制或重排。
class Block1PackedDwconvPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
public:
    Block1PackedDwconvPlugin(
        int32_t height,
        int32_t width,
        int32_t channels,
        int32_t fuseGelu,
        std::vector<int32_t> packedWeights,
        std::vector<int32_t> packedBias
    );
    ~Block1PackedDwconvPlugin() noexcept override = default;

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type
    ) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes,
        int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs
    ) const noexcept override;
    int32_t getOutputShapes(
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        const nvinfer1::DimsExprs* shapeInputs,
        int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;
    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs
    ) noexcept override;
    int32_t configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs,
        int32_t nbOutputs
    ) noexcept override;
    int32_t onShapeChange(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs
    ) noexcept override;
    size_t getWorkspaceSize(
        const nvinfer1::DynamicPluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs,
        int32_t nbOutputs
    ) const noexcept override;
    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context
    ) noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override;

private:
    // 接收两个 shared_ptr，用于 clone() 直接共享已经上传的不可变参数
    Block1PackedDwconvPlugin(
        int32_t height,
        int32_t width,
        int32_t channels,
        int32_t fuseGelu,
        std::shared_ptr<const Block1PackedDwconvHostParameters> hostParameters,
        std::shared_ptr<const Block1PackedDwconvDeviceParameters> deviceParameters
    ) noexcept;

    bool validateDescriptors(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs
    ) const noexcept;

    int32_t height_ = 0;
    int32_t width_ = 0;
    int32_t channels_ = 0;
    int32_t fuseGelu_ = 0;
    std::string namespace_ = kBlock1PackedDwconvPluginNamespace;
    std::shared_ptr<const Block1PackedDwconvHostParameters> hostParameters_;
    std::shared_ptr<const Block1PackedDwconvDeviceParameters> deviceParameters_;
    std::vector<nvinfer1::PluginField> serializedFields_;
    nvinfer1::PluginFieldCollection serializedFieldCollection_{};
};

class Block1PackedDwconvPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
public:
    Block1PackedDwconvPluginCreator() noexcept;
    ~Block1PackedDwconvPluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fieldCollection,
        nvinfer1::TensorRTPhase phase
    ) noexcept override;

private:
    std::string namespace_ = kBlock1PackedDwconvPluginNamespace;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fieldCollection_{};
};

} // 命名空间 egcinet::plugins
