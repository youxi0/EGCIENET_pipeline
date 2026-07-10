#include "infer/EngineBuilder.h"
#include "utils/FileLogger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>

namespace {

// 返回命令行参数名后面的值；缺省时返回默认值。
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

// 检查命令行里是否出现某个标志位。
bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == key) {
            return true;
        }
    }

    return false;
}

// 将短选项字符串转成小写，便于解析布尔值和精度类型。
std::string toLower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

// 解析用户传入的真假字符串；空字符串沿用默认值。
bool parseBool(const std::string& value, bool defaultValue) {
    if (value.empty()) {
        return defaultValue;
    }

    const std::string lower = toLower(value);
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

// 将命令行精度字符串转换为强类型枚举。
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

// 解析形如 "v0,v1,v2" 的 BGR 三元组。
bool parseFloatTriplet(const std::string& text, std::array<float, 3>& values) {
    if (text.empty()) {
        return true;
    }

    std::stringstream ss(text);
    std::string item;

    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::getline(ss, item, ',')) {
            return false;
        }
        values[i] = std::stof(item);
    }

    return !std::getline(ss, item, ',');
}

// 打印构建示例和所有支持的命令行参数。
void printUsage(const char* app) {
    std::cout
        << "Usage:\n"
        << "  " << app << " --onnx models/egcinet_352.onnx --engine models/egcinet_352_fp16.engine --precision fp16\n"
        << "  " << app << " --onnx models/egcinet_352.onnx --engine models/egcinet_352_int8.engine --precision int8 --calib_dir datasets/images/calibration\n\n"
        << "Options:\n"
        << "  --onnx                 ONNX model path\n"
        << "  --engine               output TensorRT engine path\n"
        << "  --precision            fp32 | fp16 | int8, default fp16\n"
        << "  --input_name           ONNX input tensor name, default image\n"
        << "  --input_w              input width, default 352\n"
        << "  --input_h              input height, default 352\n"
        << "  --mean                 BGR mean triplet, default 140.505,157.845,135.66\n"
        << "  --std                  BGR std triplet, default 61.455,60.18,62.22\n"
        << "  --min_batch            dynamic profile min batch, default 1\n"
        << "  --opt_batch            dynamic profile opt batch, default 1\n"
        << "  --max_batch            dynamic profile max batch, default 1\n"
        << "  --workspace_mib        TensorRT workspace size MiB, default 2048\n"
        << "  --calib_dir            calibration image directory for INT8\n"
        << "  --calib_cache          calibration cache path, default models/egcinet_352_int8.cache\n"
        << "  --calib_batch          calibration batch size, default 1\n"
        << "  --calib_max_images     max calibration images, 0 means all, default 500\n"
        << "  --read_calib_cache     0 | 1, default 1\n"
        << "  --fp16_fallback        0 | 1, default 1 for INT8\n"
        << "  --force_layernorm_fp32 0 | 1, default 1 for FP16/INT8\n"
        << "  --log_dir              log output dir, default results/logs\n"
        << "  --verbose              print more TensorRT parser logs\n";
}

} // 匿名命名空间

int main(int argc, char** argv) {
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    EngineBuilderConfig config;
    config.onnxPath = getArg(argc, argv, "--onnx");
    config.enginePath = getArg(argc, argv, "--engine");
    config.precision = parsePrecision(getArg(argc, argv, "--precision", "fp16"));
    config.inputTensorName = getArg(argc, argv, "--input_name", "image");
    config.inputWidth = std::stoi(getArg(argc, argv, "--input_w", "352"));
    config.inputHeight = std::stoi(getArg(argc, argv, "--input_h", "352"));
    config.minBatch = std::stoi(getArg(argc, argv, "--min_batch", "1"));
    config.optBatch = std::stoi(getArg(argc, argv, "--opt_batch", "1"));
    config.maxBatch = std::stoi(getArg(argc, argv, "--max_batch", "1"));
    config.workspaceSizeMiB = static_cast<size_t>(
        std::stoull(getArg(argc, argv, "--workspace_mib", "2048"))
    );
    config.calibrateImageDir = getArg(argc, argv, "--calib_dir");
    config.calibrateCacheFile = getArg(
        argc,
        argv,
        "--calib_cache",
        "models/egcinet_352_int8.cache"
    );
    config.calibrateBatchSize = std::stoi(getArg(argc, argv, "--calib_batch", "1"));
    config.calibrateMaxImages = static_cast<size_t>(
        std::stoull(getArg(argc, argv, "--calib_max_images", "500"))
    );
    config.readCalibrationCache = parseBool(getArg(argc, argv, "--read_calib_cache", "1"), true);
    config.allowFp16Fallback = parseBool(getArg(argc, argv, "--fp16_fallback", "1"), true);
    config.forceLayerNormFp32 = parseBool(getArg(argc, argv, "--force_layernorm_fp32", "1"), true);
    config.verbose = hasFlag(argc, argv, "--verbose");

    if (!parseFloatTriplet(getArg(argc, argv, "--mean"), config.mean) ||
        !parseFloatTriplet(getArg(argc, argv, "--std"), config.std)) {
        std::cerr << "Failed to parse --mean/--std. Expected format: b,g,r" << std::endl;
        return 1;
    }

    const std::string logDir = getArg(argc, argv, "--log_dir", "results/logs");
    utils::FileLogger::instance().open(logDir, "egcinet_build_engine");

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
