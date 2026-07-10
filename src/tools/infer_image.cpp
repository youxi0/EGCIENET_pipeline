#include "common/PreprocessData.h"
#include "common/Timer.h"
#include "infer/TensorRTInfer.h"
#include "preprocess/CudaPreprocessor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

namespace {

// 返回指定命令行选项后面的值，不存在时返回默认值。
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

// 打印单图推理工具的最小使用说明。
void printUsage(const char* application) {
    std::cout << "Usage:\n"
              << "  " << application
              << " --engine models/egcinet_fp16.engine"
              << " --image data/test.jpg"
              << " [--output mask.png]\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::string enginePath = getArgument(argc, argv, "--engine");
    const std::string imagePath = getArgument(argc, argv, "--image");
    const std::string outputPath = getArgument(argc, argv, "--output", "mask.png");

    if (enginePath.empty() || imagePath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    TensorRTInfer infer(enginePath);
    if (!infer.load()) {
        return 2;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[Infer Image] failed to read image: " << imagePath << std::endl;
        return 3;
    }

    // 预处理尺寸直接取自 engine，避免配置值与 binding 尺寸不一致。
    CudaPreprocessor preprocessor(infer.inputWidth(), infer.inputHeight());
    CudaEventTimer preprocessTimer(infer.stream());
    CudaEventTimer inferTimer(infer.stream());
    if (!preprocessTimer.isValid() || !inferTimer.isValid()) {
        return 4;
    }

    PreprocessResult preprocessResult;
    if (!preprocessTimer.start()) {
        return 4;
    }

    const bool preprocessOk = preprocessor.process(
            image,
            infer.inputDeviceBuffer(),
            infer.inputBufferBytes(),
            infer.inputElementSize(),
            preprocessResult,
            infer.stream());

    const bool preprocessTimerStopped = preprocessTimer.stop();
    if (!preprocessTimerStopped || !preprocessOk) {
        if (preprocessTimerStopped) {
            preprocessTimer.elapsedMs();
        }
        return 5;
    }

    if (!inferTimer.start()) {
        return 5;
    }

    cv::Mat modelMask;
    const bool inferOk = infer.inferFromDevice(modelMask);
    const bool inferTimerStopped = inferTimer.stop();
    if (!inferTimerStopped || !inferOk) {
        if (inferTimerStopped) {
            inferTimer.elapsedMs();
        }
        return 6;
    }

    const double preprocessGpuMs = preprocessTimer.elapsedMs();
    const double inferGpuMs = inferTimer.elapsedMs();
    if (preprocessGpuMs < 0.0 || inferGpuMs < 0.0) {
        return 6;
    }

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(modelMask, &minimum, &maximum);

    // PNG 使用 8 位灰度保存；convertTo 会把范围外的值饱和到 0 或 255。
    cv::Mat maskImage;
    modelMask.convertTo(maskImage, CV_8U, 255.0);
    if (!cv::imwrite(outputPath, maskImage)) {
        std::cerr << "[Infer Image] failed to write mask: " << outputPath << std::endl;
        return 7;
    }

    std::cout << "[Infer Image] mask shape: "
              << modelMask.cols << 'x' << modelMask.rows << std::endl;
    std::cout << "[Infer Image] probability range: ["
              << minimum << ", " << maximum << "]" << std::endl;
    std::cout << "[Infer Image] preprocess_gpu_ms: "
              << preprocessGpuMs << std::endl;
    std::cout << "[Infer Image] infer_gpu_ms: "
              << inferGpuMs << std::endl;
    std::cout << "[Infer Image] saved: " << outputPath << std::endl;
    return 0;
}
