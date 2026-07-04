#include "infer/EngineBuilder.h"
#include "utils/FileLogger.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

namespace {

std::string getArg(
    int argc,
    char** argv,
    const std::string& key,
    const std::string& defaultValue = ""
) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }

    return defaultValue;
}

bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == key) {
            return true;
        }
    }

    return false;
}

std::string toLower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

bool parseBool(const std::string& value, bool defaultValue) {
    if (value.empty()) {
        return defaultValue;
    }

    const std::string lower = toLower(value);
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

EnginePrecision parsePrecision(const std::string& value) {
    const std::string lower = toLower(value);
    if (lower == "fp32") {
        return EnginePrecision::kFP32;
    }
    if (lower == "int8") {
        return EnginePrecision::kINT8;
    }

    return EnginePrecision::kFP16;
}

void printUsage(const char* app) {
    std::cout
        << "Usage:\n"
        << "  " << app << " --onnx models/best.onnx --engine models/best_fp16.engine --precision fp16\n"
        << "  " << app << " --onnx models/best.onnx --engine models/best_int8.engine --precision int8 --calib_dir datasets/images/calibration\n\n"
        << "Options:\n"
        << "  --onnx                 ONNX model path\n"
        << "  --engine               output TensorRT engine path\n"
        << "  --precision            fp32 | fp16 | int8, default fp16\n"
        << "  --input_name           ONNX input tensor name, optional\n"
        << "  --input_w              input width, default 640\n"
        << "  --input_h              input height, default 640\n"
        << "  --min_batch            dynamic profile min batch, default 1\n"
        << "  --opt_batch            dynamic profile opt batch, default 1\n"
        << "  --max_batch            dynamic profile max batch, default 1\n"
        << "  --workspace_mib        TensorRT workspace size MiB, default 2048\n"
        << "  --calib_dir            calibration image directory for INT8\n"
        << "  --calib_cache          calibration cache path, default models/best_int8.cache\n"
        << "  --calib_batch          calibration batch size, default 1\n"
        << "  --calib_max_images     max calibration images, 0 means all, default 500\n"
        << "  --read_calib_cache     0 | 1, default 1\n"
        << "  --fp16_fallback        0 | 1, default 1 for INT8\n"
        << "  --log_dir              log output dir, default results/logs\n"
        << "  --verbose              print more TensorRT parser logs\n";
}

} // namespace

int main(int argc, char** argv) {
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    EngineBuilderConfig config;
    config.onnxPath = getArg(argc, argv, "--onnx");
    config.enginePath = getArg(argc, argv, "--engine");
    config.precision = parsePrecision(getArg(argc, argv, "--precision", "fp16"));
    config.inputTensorName = getArg(argc, argv, "--input_name");
    config.inputWidth = std::stoi(getArg(argc, argv, "--input_w", "640"));
    config.inputHeight = std::stoi(getArg(argc, argv, "--input_h", "640"));
    config.minBatch = std::stoi(getArg(argc, argv, "--min_batch", "1"));
    config.optBatch = std::stoi(getArg(argc, argv, "--opt_batch", "1"));
    config.maxBatch = std::stoi(getArg(argc, argv, "--max_batch", "1"));
    config.workspaceSizeMiB = static_cast<size_t>(
        std::stoull(getArg(argc, argv, "--workspace_mib", "2048"))
    );
    config.calibrateImageDir = getArg(argc, argv, "--calib_dir");
    config.calibrateCacheFile = getArg(argc, argv, "--calib_cache", "models/best_int8.cache");
    config.calibrateBatchSize = std::stoi(getArg(argc, argv, "--calib_batch", "1"));
    config.calibrateMaxImages = static_cast<size_t>(
        std::stoull(getArg(argc, argv, "--calib_max_images", "500"))
    );
    config.readCalibrationCache = parseBool(getArg(argc, argv, "--read_calib_cache", "1"), true);
    config.allowFp16Fallback = parseBool(getArg(argc, argv, "--fp16_fallback", "1"), true);
    config.verbose = hasFlag(argc, argv, "--verbose");

    const std::string logDir = getArg(argc, argv, "--log_dir", "results/logs");
    utils::FileLogger::instance().open(logDir, "blade_build_engine");

    if (config.onnxPath.empty() || config.enginePath.empty()) {
        printUsage(argv[0]);
        utils::FileLogger::instance().close();
        return 1;
    }

    EngineBuilder builder(config);
    if (!builder.build()) {
        std::cerr << "Failed to build engine: " << builder.lastError() << std::endl;
        utils::FileLogger::instance().close();
        return 1;
    }

    std::cout << "Engine build finished: " << config.enginePath << std::endl;
    utils::FileLogger::instance().close();
    return 0;
}
