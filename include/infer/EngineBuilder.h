#pragma once

#include "infer/TensorRTInfer.h"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Int8Calibrator;

namespace nvonnxparser {
class IParser;
} // namespace nvonnxparser

enum class EnginePrecision {
    kFP32,
    kFP16,
    kINT8
};

struct EngineBuilderConfig {
    std::string onnxPath;
    std::string enginePath;

    EnginePrecision precision = EnginePrecision::kFP16;

    // 构建动态shape engine时使用的输入尺寸；静态ONNX会保留原始shape。
    int inputWidth = 640;
    int inputHeight = 640;
    int minBatch = 1;
    int optBatch = 1;
    int maxBatch = 1;

    // INT8 PTQ校准配置；precision=int8时会创建Int8Calibrator并挂到builder config上。
    std::string calibrateImageDir;
    std::string calibrateCacheFile = "models/best_int8.cache";
    std::string inputTensorName;
    int calibrateBatchSize = 1;
    size_t calibrateMaxImages = 500;
    bool readCalibrationCache = true;
    bool allowFp16Fallback = true;

    size_t workspaceSizeMiB = 2048;
    bool verbose = false;
};

// TensorRT engine构建器：
// 1. 解析ONNX网络。
// 2. 按fp32/fp16/int8配置builder。
// 3. INT8模式下使用Int8Calibrator生成或复用校准cache。
// 4. 序列化输出.engine文件，供TensorRTInfer运行时加载。
class EngineBuilder {
public:
    explicit EngineBuilder(EngineBuilderConfig config);

    bool build();
    const std::string& lastError() const noexcept;

private:
    bool validateConfig();
    bool parseOnnx(nvonnxparser::IParser& parser);
    bool configurePrecision(
        nvinfer1::IBuilder& builder,
        nvinfer1::IBuilderConfig& builderConfig,
        std::unique_ptr<Int8Calibrator>& calibrator
    );
    bool configureOptimizationProfile(
        nvinfer1::IBuilder& builder,
        nvinfer1::INetworkDefinition& network,
        nvinfer1::IBuilderConfig& builderConfig,
        std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>>>& profiles
    );
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
