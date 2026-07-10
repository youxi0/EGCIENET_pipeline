#pragma once

#include "infer/TensorRTCommon.h"

#include <NvInfer.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Int8Calibrator;

namespace nvonnxparser {
class IParser;
} // 命名空间 nvonnxparser

enum class EnginePrecision {
    kFP32,
    kFP16,
    kINT8
};

struct EngineBuilderConfig {
    std::string onnxPath;
    std::string enginePath;

    EnginePrecision precision = EnginePrecision::kFP16;

    // EGCINET 输入约定：image，NCHW，BGR，1x3x352x352。
    std::string inputTensorName = "image";
    int inputWidth = 352;
    int inputHeight = 352;
    int minBatch = 1;
    int optBatch = 1;
    int maxBatch = 1;

    // BGR 顺序，数值基于原始 0-255 像素尺度。
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};

    // INT8 PTQ 校准配置。
    std::string calibrateImageDir;
    std::string calibrateCacheFile = "models/egcinet_352_int8.cache";
    int calibrateBatchSize = 1;
    size_t calibrateMaxImages = 500;
    bool readCalibrationCache = true;
    bool allowFp16Fallback = true;
    bool forceLayerNormFp32 = true;

    size_t workspaceSizeMiB = 2048;
    bool verbose = false;
};

class EngineBuilder {
public:
    explicit EngineBuilder(EngineBuilderConfig config);

    // 解析 ONNX，配置 TensorRT 精度/优化配置，并写出 engine plan。
    bool build();

    // build() 失败后返回最近一次可读错误信息。
    const std::string& lastError() const noexcept;

private:
    // 检查必要文件、输入尺寸、校准配置和 batch 优化配置顺序。
    bool validateConfig();

    // 执行 ONNX 解析；失败时收集 parser 诊断信息。
    bool parseOnnx(nvonnxparser::IParser& parser);

    // 开启 FP16/INT8 构建标志；INT8 模式下挂载校准器。
    bool configurePrecision(
        nvinfer1::IBuilder& builder,
        nvinfer1::IBuilderConfig& builderConfig,
        std::unique_ptr<Int8Calibrator>& calibrator
    );

    // 在 FP16/INT8 engine 中强制 LayerNorm 类层使用 FP32，其余层保持低精度。
    bool configureLayerPrecisions(
        nvinfer1::INetworkDefinition& network,
        nvinfer1::IBuilderConfig& builderConfig
    );

    // 仅在 ONNX 输入存在动态维度时添加优化配置。
    bool configureOptimizationProfile(
        nvinfer1::IBuilder& builder,
        nvinfer1::INetworkDefinition& network,
        nvinfer1::IBuilderConfig& builderConfig,
        std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>>>& profiles
    );

    // 将构建好的 engine 序列化到磁盘，必要时创建父目录。
    bool writeEnginePlan(const nvinfer1::IHostMemory& plan);

    bool hasDynamicDim(const nvinfer1::Dims& dims) const;
    nvinfer1::Dims makeProfileDims(
        const nvinfer1::Dims& originalDims,
        int batch,
        int height,
        int width
    ) const;
    const char* precisionName() const;
    void setError(const std::string& message);

private:
    EngineBuilderConfig config_;
    TrtLogger logger_;
    std::string lastError_;
};
