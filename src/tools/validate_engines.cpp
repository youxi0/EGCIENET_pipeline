#include "evaluation/EngineValidator.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct CommandLineArguments {
    ValidationConfig config;
    std::string fp32Engine;
    std::string fp16Engine;
    std::string int8Engine;
};

// 打印验证工具支持的参数。
void printUsage(const char* application) {
    std::cout
        << "Usage:\n"
        << "  " << application << " --images datasets/images/val"
        << " --labels datasets/labels/val"
        << " [--fp32 models/egcinet_352_fp32.engine]"
        << " [--fp16 models/egcinet_352_fp16.engine]"
        << " [--int8 models/egcinet_352_int8.engine]\n"
        << "Options:\n"
        << "  --output PATH       CSV output path\n"
        << "  --warmup N          Warmup iterations, default 20\n"
        << "  --iterations N      Timed iterations, default 200\n"
        << "  --max-images N      Accuracy images, 0 means all\n"
        << "  --threshold VALUE   Binary mask threshold, default 0.5\n"
        << "  --mean B,G,R        Raw-pixel BGR mean\n"
        << "  --std B,G,R         Raw-pixel BGR std\n"
        << "  --visualize 0|1     Include visualization and its D2H, default 1\n";
}

// 解析 BGR 三通道浮点参数。
std::array<float, 3> parseChannels(const std::string& text, const std::string& option) {
    std::array<float, 3> values{};
    std::stringstream stream(text);
    std::string token;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::getline(stream, token, ',') || token.empty()) {
            throw std::invalid_argument(option + " requires three comma-separated values");
        }
        values[index] = std::stof(token);
    }
    if (std::getline(stream, token, ',')) {
        throw std::invalid_argument(option + " requires exactly three values");
    }
    return values;
}

// 解析脚本传入的布尔开关。
bool parseBoolean(const std::string& text, const std::string& option) {
    if (text == "1" || text == "true" || text == "TRUE") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE") {
        return false;
    }
    throw std::invalid_argument(option + " must be 0 or 1");
}

// 解析非负迭代次数或样本上限。
std::size_t parseSize(const std::string& text, const std::string& option) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(option + " must be a non-negative integer");
    }
    std::size_t parsedCharacters = 0;
    const unsigned long long value = std::stoull(text, &parsedCharacters);
    if (parsedCharacters != text.size()) {
        throw std::invalid_argument(option + " must be an integer");
    }
    return static_cast<std::size_t>(value);
}

// 严格解析参数，避免拼错参数时静默使用默认值。
bool parseArguments(
    int argc,
    char** argv,
    CommandLineArguments& arguments,
    std::string& error
) {
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--help" || option == "-h") {
                return false;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + option);
            }
            const std::string value = argv[++index];

            if (option == "--images") {
                arguments.config.imageDirectory = value;
            } else if (option == "--labels") {
                arguments.config.labelDirectory = value;
            } else if (option == "--fp32") {
                arguments.fp32Engine = value;
            } else if (option == "--fp16") {
                arguments.fp16Engine = value;
            } else if (option == "--int8") {
                arguments.int8Engine = value;
            } else if (option == "--output") {
                arguments.config.outputCsv = value;
            } else if (option == "--warmup") {
                arguments.config.warmupIterations = parseSize(value, option);
            } else if (option == "--iterations") {
                arguments.config.benchmarkIterations = parseSize(value, option);
            } else if (option == "--max-images") {
                arguments.config.maxImages = parseSize(value, option);
            } else if (option == "--threshold") {
                arguments.config.maskThreshold = std::stof(value);
            } else if (option == "--mean") {
                arguments.config.mean = parseChannels(value, option);
            } else if (option == "--std") {
                arguments.config.std = parseChannels(value, option);
            } else if (option == "--visualize") {
                arguments.config.includeVisualization = parseBoolean(value, option);
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }

    if (arguments.config.imageDirectory.empty() || arguments.config.labelDirectory.empty()) {
        error = "--images and --labels are required";
        return false;
    }
    if (arguments.fp32Engine.empty() && arguments.fp16Engine.empty() &&
        arguments.int8Engine.empty()) {
        error = "at least one of --fp32, --fp16 and --int8 is required";
        return false;
    }
    return true;
}

// 输出单个 engine 最常用的精度和速度摘要。
void printResult(const EngineValidationResult& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "[Result] " << result.name
              << " dice=" << result.accuracy.globalDice
              << " iou=" << result.accuracy.globalIou
              << " mae=" << result.accuracy.mae
              << " infer_mean_ms=" << result.performance.inference.meanMs
              << " e2e_mean_ms=" << result.performance.endToEnd.meanMs
              << " e2e_p95_ms=" << result.performance.endToEnd.p95Ms
              << " fps=" << result.performance.fps
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    CommandLineArguments arguments;
    std::string error;
    if (!parseArguments(argc, argv, arguments, error)) {
        if (!error.empty()) {
            std::cerr << "[Validate] " << error << std::endl;
        }
        printUsage(argv[0]);
        return error.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::vector<EngineValidationTarget> targets;
    if (!arguments.fp32Engine.empty()) {
        targets.push_back({"fp32", arguments.fp32Engine});
    }
    if (!arguments.fp16Engine.empty()) {
        targets.push_back({"fp16", arguments.fp16Engine});
    }
    if (!arguments.int8Engine.empty()) {
        targets.push_back({"int8", arguments.int8Engine});
    }

    EngineValidator validator(std::move(arguments.config));
    std::vector<EngineValidationResult> results;
    if (!validator.run(targets, results)) {
        std::cerr << "[Validate] " << validator.lastError() << std::endl;
        return EXIT_FAILURE;
    }

    for (const EngineValidationResult& result : results) {
        printResult(result);
    }
    std::cout << "[PASS] engine validation finished" << std::endl;
    return EXIT_SUCCESS;
}
