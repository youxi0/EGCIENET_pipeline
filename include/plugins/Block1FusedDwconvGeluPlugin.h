#pragma once

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#if NV_TENSORRT_MAJOR < 10
#error "Block1FusedDwconvGeluPlugin requires TensorRT 10 or newer."
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace egcinet::plugins {

inline constexpr char kBlock1FusedPluginName[] =
    "EGCINET_Block1FusedDwconvGelu";
inline constexpr char kBlock1FusedPluginVersion[] = "1";
// REGISTER_TENSORRT_PLUGIN 只能把 Creator 注册到 TensorRT 默认命名空间。
inline constexpr char kBlock1FusedPluginNamespace[] = "";

// TensorRT 10 IPluginV3 实现：build 描述输入输出 shape/format，runtime调度融合 CUDA kernel。
class Block1FusedDwconvGeluPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
public:
    Block1FusedDwconvGeluPlugin(int32_t height, int32_t width) noexcept;
    ~Block1FusedDwconvGeluPlugin() noexcept override = default;

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
    bool validateDescriptors(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs
    ) const noexcept;

    int32_t height_ = 0;
    int32_t width_ = 0;
    std::string namespace_ = kBlock1FusedPluginNamespace;
    std::vector<nvinfer1::PluginField> serializedFields_;
    nvinfer1::PluginFieldCollection serializedFieldCollection_{};
};

// Creator 负责接收 ONNX 节点属性，并在构建或反序列化阶段创建插件对象。
class Block1FusedDwconvGeluPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
public:
    Block1FusedDwconvGeluPluginCreator() noexcept;
    ~Block1FusedDwconvGeluPluginCreator() noexcept override = default;

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
    std::string namespace_ = kBlock1FusedPluginNamespace;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fieldCollection_{};
};

} // 命名空间 egcinet::plugins
