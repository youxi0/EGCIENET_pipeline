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

    // Model input contract for EGCINET: image, NCHW, BGR, 1x3x352x352.
    std::string inputTensorName = "image";
    int inputWidth = 352;
    int inputHeight = 352;
    int minBatch = 1;
    int optBatch = 1;
    int maxBatch = 1;

    // BGR order, raw 0-255 pixel scale.
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};

    // INT8 PTQ calibration settings.
    std::string calibrateImageDir;
    std::string calibrateCacheFile = "models/egcinet_352_int8.cache";
    int calibrateBatchSize = 1;
    size_t calibrateMaxImages = 500;
    bool readCalibrationCache = true;
    bool allowFp16Fallback = true;

    size_t workspaceSizeMiB = 2048;
    bool verbose = false;
};

class EngineBuilder {
public:
    explicit EngineBuilder(EngineBuilderConfig config);

    // Parses ONNX, configures TensorRT precision/profile, and writes the engine plan.
    bool build();

    // Returns the latest human-readable error after build() fails.
    const std::string& lastError() const noexcept;

private:
    // Checks required files, dimensions, calibration settings, and batch profile order.
    bool validateConfig();

    // Runs the ONNX parser and captures parser diagnostics on failure.
    bool parseOnnx(nvonnxparser::IParser& parser);

    // Enables FP16/INT8 flags and attaches the INT8 calibrator when requested.
    bool configurePrecision(
        nvinfer1::IBuilder& builder,
        nvinfer1::IBuilderConfig& builderConfig,
        std::unique_ptr<Int8Calibrator>& calibrator
    );

    // Adds an optimization profile only when the ONNX input has dynamic dimensions.
    bool configureOptimizationProfile(
        nvinfer1::IBuilder& builder,
        nvinfer1::INetworkDefinition& network,
        nvinfer1::IBuilderConfig& builderConfig,
        std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>>>& profiles
    );

    // Serializes the built engine to disk, creating the parent directory if needed.
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
