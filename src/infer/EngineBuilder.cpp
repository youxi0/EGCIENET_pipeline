#include "infer/EngineBuilder.h"

#include "infer/Int8Calibrator.h"
#include "utils/FileLogger.h"

#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

namespace {

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        oss << dims.d[i];
        if (i + 1 < dims.nbDims) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void setWorkspaceSize(nvinfer1::IBuilderConfig& builderConfig, size_t workspaceSizeMiB) {
    const size_t workspaceBytes = workspaceSizeMiB * 1024ULL * 1024ULL;

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
    builderConfig.setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes);
#else
    builderConfig.setMaxWorkspaceSize(workspaceBytes);
#endif
}

} // namespace

EngineBuilder::EngineBuilder(EngineBuilderConfig config)
    : config_(std::move(config)) {}

bool EngineBuilder::build() {
    if (!validateConfig()) {
        return false;
    }

    utils::FileLogger::instance().info(
        std::string("[EngineBuilder] build start, precision=") + precisionName()
    );
    utils::FileLogger::instance().info("[EngineBuilder] onnx: " + config_.onnxPath);
    utils::FileLogger::instance().info("[EngineBuilder] engine: " + config_.enginePath);

    std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>> builder(
        nvinfer1::createInferBuilder(logger_)
    );
    if (!builder) {
        setError("failed to create TensorRT builder");
        return false;
    }

    const uint32_t explicitBatchFlag =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network(
        builder->createNetworkV2(explicitBatchFlag)
    );
    if (!network) {
        setError("failed to create TensorRT network");
        return false;
    }

    std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>> parser(
        nvonnxparser::createParser(*network, logger_)
    );
    if (!parser) {
        setError("failed to create ONNX parser");
        return false;
    }

    if (!parseOnnx(*parser)) {
        return false;
    }

    for (int i = 0; i < network->getNbInputs(); ++i) {
        const nvinfer1::ITensor* input = network->getInput(i);
        if (input != nullptr) {
            utils::FileLogger::instance().info(
                std::string("[EngineBuilder] input ") + input->getName() +
                " shape=" + dimsToString(input->getDimensions())
            );
        }
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        const nvinfer1::ITensor* output = network->getOutput(i);
        if (output != nullptr) {
            utils::FileLogger::instance().info(
                std::string("[EngineBuilder] output ") + output->getName() +
                " shape=" + dimsToString(output->getDimensions())
            );
        }
    }

    std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>> builderConfig(
        builder->createBuilderConfig()
    );
    if (!builderConfig) {
        setError("failed to create TensorRT builder config");
        return false;
    }

    setWorkspaceSize(*builderConfig, config_.workspaceSizeMiB);

    std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>>> profiles;
    if (!configureOptimizationProfile(*builder, *network, *builderConfig, profiles)) {
        return false;
    }

    std::unique_ptr<Int8Calibrator> calibrator;
    if (!configurePrecision(*builder, *builderConfig, calibrator)) {
        return false;
    }

    std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>> plan(
        builder->buildSerializedNetwork(*network, *builderConfig)
    );
    if (!plan) {
        setError("TensorRT buildSerializedNetwork failed");
        return false;
    }

    if (!writeEnginePlan(*plan)) {
        return false;
    }

    utils::FileLogger::instance().info("[EngineBuilder] build finished");
    return true;
}

const std::string& EngineBuilder::lastError() const noexcept {
    return lastError_;
}

bool EngineBuilder::validateConfig() {
    if (config_.onnxPath.empty()) {
        setError("onnx path is empty");
        return false;
    }

    if (config_.enginePath.empty()) {
        setError("engine path is empty");
        return false;
    }

    if (!fs::is_regular_file(config_.onnxPath)) {
        setError("onnx file does not exist: " + config_.onnxPath);
        return false;
    }

    if (config_.inputWidth <= 0 || config_.inputHeight <= 0) {
        setError("input width and height must be positive");
        return false;
    }

    if (config_.minBatch <= 0 || config_.optBatch <= 0 || config_.maxBatch <= 0) {
        setError("min/opt/max batch must be positive");
        return false;
    }

    if (config_.minBatch > config_.optBatch || config_.optBatch > config_.maxBatch) {
        setError("batch profile must satisfy min <= opt <= max");
        return false;
    }

    if (config_.calibrateBatchSize <= 0) {
        setError("calibration batch size must be positive");
        return false;
    }

    if (std::any_of(config_.std.begin(), config_.std.end(), [](float value) {
            return value == 0.0f;
        })) {
        setError("preprocess std values must be non-zero");
        return false;
    }

    return true;
}

bool EngineBuilder::parseOnnx(nvonnxparser::IParser& parser) {
    const int verbosity = static_cast<int>(
        config_.verbose ? nvinfer1::ILogger::Severity::kINFO
                        : nvinfer1::ILogger::Severity::kWARNING
    );

    if (parser.parseFromFile(config_.onnxPath.c_str(), verbosity)) {
        return true;
    }

    std::ostringstream oss;
    oss << "failed to parse ONNX: " << config_.onnxPath;
    const int errorCount = parser.getNbErrors();
    for (int i = 0; i < errorCount; ++i) {
        const auto* error = parser.getError(i);
        if (error != nullptr) {
            oss << "\n  [" << i << "] " << error->desc();
        }
    }

    setError(oss.str());
    return false;
}

bool EngineBuilder::configurePrecision(
    nvinfer1::IBuilder& builder,
    nvinfer1::IBuilderConfig& builderConfig,
    std::unique_ptr<Int8Calibrator>& calibrator
) {
    if (config_.precision == EnginePrecision::kFP32) {
        utils::FileLogger::instance().info("[EngineBuilder] use FP32 precision");
        return true;
    }

    if (config_.precision == EnginePrecision::kFP16) {
        if (!builder.platformHasFastFp16()) {
            utils::FileLogger::instance().warning(
                "[EngineBuilder] platformHasFastFp16 is false, FP16 build may not be faster"
            );
        }
        builderConfig.setFlag(nvinfer1::BuilderFlag::kFP16);
        utils::FileLogger::instance().info("[EngineBuilder] enable FP16");
        return true;
    }

    if (!builder.platformHasFastInt8()) {
        utils::FileLogger::instance().warning(
            "[EngineBuilder] platformHasFastInt8 is false, INT8 build may not be faster"
        );
    }

    builderConfig.setFlag(nvinfer1::BuilderFlag::kINT8);

    if (config_.allowFp16Fallback) {
        builderConfig.setFlag(nvinfer1::BuilderFlag::kFP16);
        utils::FileLogger::instance().info("[EngineBuilder] enable FP16 fallback for INT8 build");
    }

    Int8CalibratorConfig calibratorConfig;
    calibratorConfig.imageDirectory = config_.calibrateImageDir;
    calibratorConfig.cacheFile = config_.calibrateCacheFile;
    calibratorConfig.inputTensorName = config_.inputTensorName;
    calibratorConfig.batchSize = config_.calibrateBatchSize;
    calibratorConfig.inputWidth = config_.inputWidth;
    calibratorConfig.inputHeight = config_.inputHeight;
    calibratorConfig.mean = config_.mean;
    calibratorConfig.std = config_.std;
    calibratorConfig.maxImages = config_.calibrateMaxImages;
    calibratorConfig.readCache = config_.readCalibrationCache;

    calibrator = std::make_unique<Int8Calibrator>(calibratorConfig);
    if (!calibrator->isValid()) {
        setError("INT8 calibrator is invalid: " + calibrator->lastError());
        return false;
    }

    builderConfig.setInt8Calibrator(calibrator.get());

    std::ostringstream oss;
    oss << "[EngineBuilder] enable INT8, calibration_images=" << calibrator->imageCount()
        << ", cache=" << config_.calibrateCacheFile;
    utils::FileLogger::instance().info(oss.str());
    return true;
}

bool EngineBuilder::configureOptimizationProfile(
    nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network,
    nvinfer1::IBuilderConfig& builderConfig,
    std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>>>& profiles
) {
    bool needsProfile = false;
    for (int i = 0; i < network.getNbInputs(); ++i) {
        const nvinfer1::ITensor* input = network.getInput(i);
        if (input != nullptr && hasDynamicDim(input->getDimensions())) {
            needsProfile = true;
            break;
        }
    }

    if (!needsProfile) {
        utils::FileLogger::instance().info("[EngineBuilder] ONNX input shape is static, no profile needed");
        return true;
    }

    std::unique_ptr<nvinfer1::IOptimizationProfile, TrtDestroy<nvinfer1::IOptimizationProfile>> profile(
        builder.createOptimizationProfile()
    );
    if (!profile) {
        setError("failed to create optimization profile");
        return false;
    }

    for (int i = 0; i < network.getNbInputs(); ++i) {
        nvinfer1::ITensor* input = network.getInput(i);
        if (input == nullptr) {
            continue;
        }

        const nvinfer1::Dims inputDims = input->getDimensions();
        if (!hasDynamicDim(inputDims)) {
            continue;
        }

        const std::string inputName = input->getName();
        const nvinfer1::Dims minDims = makeProfileDims(
            inputDims,
            config_.minBatch,
            config_.inputHeight,
            config_.inputWidth
        );
        const nvinfer1::Dims optDims = makeProfileDims(
            inputDims,
            config_.optBatch,
            config_.inputHeight,
            config_.inputWidth
        );
        const nvinfer1::Dims maxDims = makeProfileDims(
            inputDims,
            config_.maxBatch,
            config_.inputHeight,
            config_.inputWidth
        );

        if (!profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, minDims) ||
            !profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, optDims) ||
            !profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDims)) {
            setError("failed to set optimization profile for input: " + inputName);
            return false;
        }

        std::ostringstream oss;
        oss << "[EngineBuilder] dynamic profile for input=" << inputName
            << ", min=" << dimsToString(minDims)
            << ", opt=" << dimsToString(optDims)
            << ", max=" << dimsToString(maxDims);
        utils::FileLogger::instance().info(oss.str());
    }

    const int profileIndex = builderConfig.addOptimizationProfile(profile.get());
    if (profileIndex < 0) {
        setError("failed to add optimization profile to builder config");
        return false;
    }

    profiles.push_back(std::move(profile));
    return true;
}

bool EngineBuilder::writeEnginePlan(const nvinfer1::IHostMemory& plan) {
    const fs::path enginePath(config_.enginePath);
    const fs::path parent = enginePath.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            setError("failed to create engine directory: " + ec.message());
            return false;
        }
    }

    std::ofstream file(config_.enginePath, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        setError("failed to open engine for writing: " + config_.enginePath);
        return false;
    }

    file.write(
        static_cast<const char*>(plan.data()),
        static_cast<std::streamsize>(plan.size())
    );

    if (!file.good()) {
        setError("failed to write engine: " + config_.enginePath);
        return false;
    }

    std::ostringstream oss;
    oss << "[EngineBuilder] wrote engine: " << config_.enginePath
        << ", bytes=" << plan.size();
    utils::FileLogger::instance().info(oss.str());
    return true;
}

bool EngineBuilder::hasDynamicDim(const nvinfer1::Dims& dims) const {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }

    return false;
}

nvinfer1::Dims EngineBuilder::makeProfileDims(
    const nvinfer1::Dims& originalDims,
    int batch,
    int height,
    int width
) const {
    nvinfer1::Dims dims = originalDims;

    if (dims.nbDims == 4) {
        if (dims.d[0] < 0) {
            dims.d[0] = batch;
        }
        if (dims.d[1] < 0) {
            dims.d[1] = 3;
        }
        if (dims.d[2] < 0) {
            dims.d[2] = height;
        }
        if (dims.d[3] < 0) {
            dims.d[3] = width;
        }
        return dims;
    }

    if (dims.nbDims == 3) {
        if (dims.d[0] < 0) {
            dims.d[0] = 3;
        }
        if (dims.d[1] < 0) {
            dims.d[1] = height;
        }
        if (dims.d[2] < 0) {
            dims.d[2] = width;
        }
        return dims;
    }

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            dims.d[i] = 1;
        }
    }

    return dims;
}

const char* EngineBuilder::precisionName() const {
    switch (config_.precision) {
    case EnginePrecision::kFP32:
        return "fp32";
    case EnginePrecision::kFP16:
        return "fp16";
    case EnginePrecision::kINT8:
        return "int8";
    default:
        return "unknown";
    }
}

void EngineBuilder::setError(const std::string& message) {
    lastError_ = message;
    utils::FileLogger::instance().error("[EngineBuilder] " + message);
}
