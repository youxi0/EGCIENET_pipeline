#include "infer/Int8Calibrator.h"
#include "infer/TensorRTCommon.h"
#include "utils/FileLogger.h"

#include <NvOnnxParser.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

// 返回命令行参数名后面的值；不存在时返回默认值。
std::string getArgument(
    int argc,
    char** argv,
    const std::string& key,
    const std::string& defaultValue = ""
) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == key) {
            return argv[index + 1];
        }
    }
    return defaultValue;
}

// 检查命令行中是否存在不带值的开关。
bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == key) {
            return true;
        }
    }
    return false;
}

// 解析形如 b,g,r 的三个浮点数。
bool parseFloatTriplet(const std::string& text, std::array<float, 3>& values) {
    if (text.empty()) {
        return true;
    }

    std::stringstream stream(text);
    std::string item;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::getline(stream, item, ',')) {
            return false;
        }
        values[index] = std::stof(item);
    }
    return !std::getline(stream, item, ',');
}

// 打印校准工具参数；该工具只生成 cache，不保存构建阶段产生的临时 plan。
void printUsage(const char* application) {
    std::cout
        << "Usage:\n"
        << "  " << application
        << " --onnx models/egcinet_352.onnx"
        << " --calib_dir datasets/images/calibration"
        << " --calib_cache models/egcinet_352_int8.cache\n\n"
        << "Options:\n"
        << "  --input_name       input tensor name, default image\n"
        << "  --input_w          input width, default 352\n"
        << "  --input_h          input height, default 352\n"
        << "  --batch            calibration batch, default 1\n"
        << "  --max_images       max calibration images, 0 means all, default 500\n"
        << "  --mean             BGR mean, default 140.505,157.845,135.66\n"
        << "  --std              BGR std, default 61.455,60.18,62.22\n"
        << "  --workspace_mib    temporary build workspace, default 2048\n"
        << "  --log_dir          log directory, default results/logs\n";
}

// 输出 ONNX parser 收集到的详细错误。
void printParserErrors(const nvonnxparser::IParser& parser) {
    for (int index = 0; index < parser.getNbErrors(); ++index) {
        const nvonnxparser::IParserError* error = parser.getError(index);
        if (error != nullptr) {
            std::cerr << "[INT8 Calibration] ONNX parser: " << error->desc() << std::endl;
        }
    }
}

// 当前模型使用静态 NCHW 输入；校准 batch 和尺寸必须与 ONNX 完全一致。
bool validateInputShape(
    const nvinfer1::INetworkDefinition& network,
    const Int8CalibratorConfig& config
) {
    if (network.getNbInputs() != 1 || network.getInput(0) == nullptr) {
        std::cerr << "[INT8 Calibration] expected exactly one ONNX input" << std::endl;
        return false;
    }

    const nvinfer1::ITensor& input = *network.getInput(0);
    const std::string inputName = input.getName() == nullptr ? "" : input.getName();
    if (inputName != config.inputTensorName) {
        std::cerr << "[INT8 Calibration] input name mismatch: ONNX=" << inputName
                  << ", config=" << config.inputTensorName << std::endl;
        return false;
    }

    const nvinfer1::Dims dims = input.getDimensions();
    if (dims.nbDims != 4 || dims.d[0] != config.batchSize || dims.d[1] != 3 ||
        dims.d[2] != config.inputHeight || dims.d[3] != config.inputWidth) {
        std::cerr << "[INT8 Calibration] expected static input ["
                  << config.batchSize << ",3," << config.inputHeight << ','
                  << config.inputWidth << "]" << std::endl;
        return false;
    }
    return true;
}

// 检查 TensorRT 是否已经写出非空校准 cache。
bool isNonEmptyFile(const std::string& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error) {
        return false;
    }

    const std::uintmax_t fileSize = fs::file_size(path, error);
    return !error && fileSize > 0;
}

} // namespace

int main(int argc, char** argv) {
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    try {
        const std::string onnxPath = getArgument(argc, argv, "--onnx");
        Int8CalibratorConfig calibratorConfig;
        calibratorConfig.imageDirectory = getArgument(argc, argv, "--calib_dir");
        calibratorConfig.cacheFile = getArgument(argc, argv, "--calib_cache");
        calibratorConfig.inputTensorName = getArgument(argc, argv, "--input_name", "image");
        calibratorConfig.inputWidth = std::stoi(getArgument(argc, argv, "--input_w", "352"));
        calibratorConfig.inputHeight = std::stoi(getArgument(argc, argv, "--input_h", "352"));
        calibratorConfig.batchSize = std::stoi(getArgument(argc, argv, "--batch", "1"));
        calibratorConfig.maxImages = static_cast<size_t>(
            std::stoull(getArgument(argc, argv, "--max_images", "500"))
        );
        calibratorConfig.readCache = false;

        const size_t workspaceMiB = static_cast<size_t>(
            std::stoull(getArgument(argc, argv, "--workspace_mib", "2048"))
        );
        const std::string logDirectory =
            getArgument(argc, argv, "--log_dir", "results/logs");

        if (onnxPath.empty() || calibratorConfig.imageDirectory.empty() ||
            calibratorConfig.cacheFile.empty()) {
            printUsage(argv[0]);
            return 1;
        }

        if (!parseFloatTriplet(getArgument(argc, argv, "--mean"), calibratorConfig.mean) ||
            !parseFloatTriplet(getArgument(argc, argv, "--std"), calibratorConfig.std)) {
            throw std::invalid_argument("--mean/--std must use b,g,r format");
        }

        if (workspaceMiB == 0 ||
            workspaceMiB > std::numeric_limits<size_t>::max() / (1024U * 1024U)) {
            throw std::invalid_argument("--workspace_mib is out of range");
        }

        if (!utils::FileLogger::instance().open(logDirectory, "egcinet_int8_calibration")) {
            return 2;
        }

        TrtLogger logger;
        std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>> builder(
            nvinfer1::createInferBuilder(logger)
        );
        if (!builder) {
            throw std::runtime_error("failed to create TensorRT builder");
        }

        const uint32_t explicitBatchFlag =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network(
            builder->createNetworkV2(explicitBatchFlag)
        );
        if (!network) {
            throw std::runtime_error("failed to create TensorRT network");
        }

        std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>> parser(
            nvonnxparser::createParser(*network, logger)
        );
        if (!parser) {
            throw std::runtime_error("failed to create ONNX parser");
        }

        if (!parser->parseFromFile(
                onnxPath.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            printParserErrors(*parser);
            throw std::runtime_error("failed to parse ONNX model");
        }

        if (!validateInputShape(*network, calibratorConfig)) {
            utils::FileLogger::instance().close();
            return 3;
        }

        Int8Calibrator calibrator(calibratorConfig);
        if (!calibrator.isValid()) {
            throw std::runtime_error("invalid calibrator: " + calibrator.lastError());
        }

        std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>> buildConfig(
            builder->createBuilderConfig()
        );
        if (!buildConfig) {
            throw std::runtime_error("failed to create TensorRT builder config");
        }

        buildConfig->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            workspaceMiB * 1024U * 1024U
        );
        buildConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
        buildConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        buildConfig->setInt8Calibrator(&calibrator);

        utils::FileLogger::instance().info(
            "[INT8 Calibration] start calibration; temporary engine plan will be discarded"
        );
        std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>> temporaryPlan(
            builder->buildSerializedNetwork(*network, *buildConfig)
        );
        if (!temporaryPlan) {
            throw std::runtime_error("TensorRT calibration build failed");
        }

        if (!calibrator.lastError().empty()) {
            throw std::runtime_error("calibration failed: " + calibrator.lastError());
        }

        if (!isNonEmptyFile(calibratorConfig.cacheFile)) {
            throw std::runtime_error("TensorRT did not write a non-empty calibration cache");
        }

        std::cout << "[PASS] INT8 calibration cache: "
                  << calibratorConfig.cacheFile << std::endl;
        utils::FileLogger::instance().close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[INT8 Calibration] " << error.what() << std::endl;
        utils::FileLogger::instance().close();
        return 4;
    }
}
