#pragma once

#include <NvInferRuntime.h>
#include <NvInferVersion.h>

#if NV_TENSORRT_MAJOR < 10
#error "FusedSpatialReductionPlugin requires TensorRT 10 or newer."
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace egcinet::plugins {

inline constexpr char kFusedSpatialReductionPluginName[] =
    "EGCINET_FusedSpatialReduction";
inline constexpr char kFusedSpatialReductionPluginVersion[] = "1";
inline constexpr char kFusedSpatialReductionPluginNamespace[] = "";

struct FusedSpatialReductionParameters {
    int32_t stage = 0;
    int32_t layerIndex = 0;
    int32_t inputHeight = 0;
    int32_t inputWidth = 0;
    int32_t inputChannels = 0;
    int32_t outputHeight = 0;
    int32_t outputWidth = 0;
    int32_t outputChannels = 0;
    int32_t kernelHeight = 0;
    int32_t kernelWidth = 0;
    int32_t strideHeight = 0;
    int32_t strideWidth = 0;
    int32_t groups = 0;
    int32_t packedK = 0;
    int32_t packedN = 0;
    int32_t fuseLayerNorm = 0;
    int32_t int8Mode = 0;
    float activationScale = 0.0F;
};

struct FusedSpatialReductionRuntime;

// TensorRT V3 plugin，支持三种静态接口：
//   FP16: input[0] BNC FP16, input[1] FP16 weight [K,N],
//         input[2] FP16 bias [N].
//   INT8: input[0] BNC INT8, input[1] INT8 weight [N,K],
//         input[2] FP16 bias [N], input[3] FP32 dequant scale [N].
//   Fused-Q INT8: 与 INT8 路径接口相同，但 input[0] 为 FP16/FP32；
//         window pack 在重排过程中按 activationScale 直接量化为 INT8。
// INT8 的 scale 已在构图时合并为 activation_scale * weight_scale[N]，
// GEMM 使用 INT32 累加。两条路径都输出 token-major BNC FP16。
class FusedSpatialReductionPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
public:
    explicit FusedSpatialReductionPlugin(
        FusedSpatialReductionParameters parameters
    );
    ~FusedSpatialReductionPlugin() noexcept override;

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
        int32_t nbOutputs,
        bool allowDynamic
    ) const noexcept;

    FusedSpatialReductionParameters parameters_{};
    std::string namespace_ = kFusedSpatialReductionPluginNamespace;
    std::unique_ptr<FusedSpatialReductionRuntime> runtime_;
    std::vector<nvinfer1::PluginField> serializedFields_;
    nvinfer1::PluginFieldCollection serializedFieldCollection_{};
};

class FusedSpatialReductionPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
public:
    FusedSpatialReductionPluginCreator() noexcept;
    ~FusedSpatialReductionPluginCreator() noexcept override = default;

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
    std::string namespace_ = kFusedSpatialReductionPluginNamespace;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fieldCollection_{};
};

} // namespace egcinet::plugins
