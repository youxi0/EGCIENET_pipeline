#include "acquisition/ImageFolderSource.h"
#include "acquisition/ImageSource.h"
#include "acquisition/VideoSource.h"
#include "pipeline/InspectionPipeline.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// getArg() 负责取参数
//    ↓
// splitClasses() 负责拆类别
//    ↓
// main() 根据这些参数创建：
//    - TensorRTInfer
//    - ImageFolderSource / VideoSource / CameraSource
//    - TCP Server
//      while(gRunning)
//      - 读图
//      - 推理
//      - 后处理
//      - 发送结果

namespace {

std::atomic<bool> gRunning{true};

void handleSignal(int) {
    gRunning = false;
}

// 把命令行里传进来的类别字符串拆开
std::vector<std::string> splitClasses(const std::string& text) {
    std::vector<std::string> classes;
    std::stringstream ss(text);
    std::string item;
    // 从字符串流 ss 里面读内容，每次遇到逗号 , 就切一段
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            classes.push_back(item);
        }
    }

    if (classes.empty()) {
        classes.push_back("defect");
    }

    return classes;
}

void printUsage(const char* app) {
    std::cout << "Usage:\n"
              << "  " << app << " --engine best.engine --source ./images --type folder --port 9000 --classes scratch,crack\n"
              << "  " << app << " --engine best.engine --source ./test.mp4 --type video --port 9000 --classes scratch,crack\n"
              << "  " << app << " --engine best.engine --source 0 --type camera --port 9000 --classes scratch,crack\n\n"
              << "Options:\n"
              << "  --engine   TensorRT engine path\n"
              << "  --source   image folder / video path / camera id\n"
              << "  --type     folder | video | camera\n"
              << "  --port     TCP listen port, default 9000\n"
              << "  --classes  comma separated class names, default defect\n"
              << "  --save_baseline  0 | 1, default 1\n"
              << "  --baseline_dir  baseline output dir, default results/baseline_fp16\n"
              << "  --log_dir       log output dir, default results/logs\n"
              << "  --enable_int8_compare  0 | 1, default 0\n"
              << "  --int8_engine          INT8 TensorRT engine path\n"
              << "  --compare_dir          FP16 vs INT8 output dir, default results/compare_fp16_int8\n"
              << "  --compare_iou          box IoU threshold, default 0.5\n"
              << "  --compare_min_match    minimum match rate, default 0.95\n"
              << "  --compare_max_score_diff  maximum mean score diff, default 0.10\n";
}

std::string getArg(int argc, char** argv, const std::string& key, const std::string& defaultValue = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }
    return defaultValue;
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    std::string enginePath = getArg(argc, argv, "--engine");
    std::string sourcePath = getArg(argc, argv, "--source");
    std::string sourceType = getArg(argc, argv, "--type", "folder");
    std::string classText = getArg(argc, argv, "--classes", "defect");
    bool saveBaseline = getArg(argc, argv, "--save_baseline", "1") == "1";
    std::string baselineDir = getArg(argc, argv, "--baseline_dir", "results/baseline_fp16");
    std::string logDir = getArg(argc, argv, "--log_dir", "results/logs");
    bool enableInt8Compare = getArg(argc, argv, "--enable_int8_compare", "0") == "1";
    std::string int8EnginePath = getArg(argc, argv, "--int8_engine");
    std::string compareDir = getArg(argc, argv, "--compare_dir", "results/compare_fp16_int8");
    float compareIouThreshold = std::stof(getArg(argc, argv, "--compare_iou", "0.5"));
    float compareMinMatchRate = std::stof(getArg(argc, argv, "--compare_min_match", "0.95"));
    float compareMaxMeanScoreDiff = std::stof(getArg(argc, argv, "--compare_max_score_diff", "0.10"));
    uint16_t port = static_cast<uint16_t>(std::stoi(getArg(argc, argv, "--port", "9000")));

    if (enginePath.empty() || sourcePath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    std::unique_ptr<ImageSource> source;

    if (sourceType == "folder") {
        source = std::make_unique<ImageFolderSource>(sourcePath);
    }
    else if (sourceType == "video") {
        source = std::make_unique<VideoSource>(sourcePath);
    }
    else if (sourceType == "camera") {
        source = std::make_unique<VideoSource>(std::stoi(sourcePath));
    }
    else {
        std::cerr << "Unknown source type: " << sourceType << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    blade::pipeline::PipelineConfig config;
    config.enginePath = enginePath;
    config.listenPort = port;
    config.classNames = splitClasses(classText);
    config.inputWidth = 640;
    config.inputHeight = 640;
    config.confThreshold = 0.25f;
    config.nmsThreshold = 0.45f;
    config.maskThreshold = 0.5f;
    config.jpegQuality = 85;
    config.queueSize = 4;
    config.heartbeatIntervalMs = 2000;
    config.heartbeatTimeoutMs = 8000;
    config.saveBaseline = saveBaseline;
    config.baselineDir = baselineDir;
    config.logDir = logDir;
    config.enableInt8Compare = enableInt8Compare;
    config.int8EnginePath = int8EnginePath;
    config.compareDir = compareDir;
    config.compareIouThreshold = compareIouThreshold;
    config.compareMinMatchRate = compareMinMatchRate;
    config.compareMaxMeanScoreDiff = compareMaxMeanScoreDiff;

    blade::pipeline::InspectionPipeline pipeline(std::move(source), config);

    if (!pipeline.start()) {
        std::cerr << "Failed to start inspection pipeline." << std::endl;
        return 1;
    }

    std::cout << "Inspection pipeline is running. Press Ctrl+C to exit." << std::endl;

    while (gRunning && pipeline.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    pipeline.stop();
    return 0;
}
