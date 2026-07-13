#include "common/PreprocessData.h"
#include "common/Timer.h"
#include "infer/TensorRTInfer.h"
#include "postprocess/SegPostprocessor.h"
#include "preprocess/CudaPreprocessor.h"
#include "visualize/Visualizer.h"

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
              << " [--output mask.png]"
              << " [--probability probability.png]"
              << " [--visualized visualized.png]\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::string enginePath = getArgument(argc, argv, "--engine");
    const std::string imagePath = getArgument(argc, argv, "--image");
    const std::string outputPath = getArgument(argc, argv, "--output", "mask.png");
    const std::string probabilityPath =
        getArgument(argc, argv, "--probability", "probability.png");
    const std::string visualizedPath =
        getArgument(argc, argv, "--visualized", "visualized.png");

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
    SegPostprocessor postprocessor;
    Visualizer visualizer;
    CudaEventTimer preprocessTimer(infer.stream());
    CudaEventTimer inferTimer(infer.stream());
    CudaEventTimer postprocessTimer(infer.stream());
    CudaEventTimer visualizeTimer(infer.stream());
    CudaEventTimer d2hTimer(infer.stream());
    if (!preprocessTimer.isValid() || !inferTimer.isValid() ||
        !postprocessTimer.isValid() || !visualizeTimer.isValid() ||
        !d2hTimer.isValid()) {
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

    const bool inferOk = infer.inferFromDevice();
    const bool inferTimerStopped = inferTimer.stop();
    if (!inferTimerStopped || !inferOk) {
        if (inferTimerStopped) {
            inferTimer.elapsedMs();
        }
        return 6;
    }

    if (!postprocessTimer.start()) {
        return 6;
    }

    const bool postprocessOk = postprocessor.process(
        infer.outputDeviceBuffer(),
        infer.outputBufferBytes(),
        infer.outputElementSize(),
        infer.outputWidth(),
        infer.outputHeight(),
        image.cols,
        image.rows,
        infer.stream()
    );
    const bool postprocessTimerStopped = postprocessTimer.stop();
    if (!postprocessTimerStopped || !postprocessOk) {
        return 7;
    }

    if (!visualizeTimer.start()) {
        return 7;
    }

    const bool visualizeOk = visualizer.process(
        preprocessor.imageDeviceBuffer(),
        preprocessor.imageDeviceStep(),
        preprocessor.imageWidth(),
        preprocessor.imageHeight(),
        postprocessor.probabilityDeviceBuffer(),
        postprocessor.binaryDeviceBuffer(),
        infer.stream()
    );
    const bool visualizeTimerStopped = visualizeTimer.stop();
    if (!visualizeTimerStopped || !visualizeOk) {
        return 8;
    }

    // 所有 GPU 计算都已排入同一 stream，最后统一排队 D2H 并同步一次。
    if (!d2hTimer.start()) {
        return 8;
    }

    cv::Mat probabilityMask;
    cv::Mat binaryMask;
    cv::Mat visualizedImage;
    const bool masksDownloadOk = postprocessor.enqueueDownload(
        probabilityMask,
        binaryMask,
        infer.stream()
    );
    const bool imageDownloadOk = visualizer.enqueueDownload(
        preprocessor.imageDeviceBuffer(),
        preprocessor.imageDeviceStep(),
        preprocessor.imageWidth(),
        preprocessor.imageHeight(),
        visualizedImage,
        infer.stream()
    );
    const bool d2hTimerStopped = d2hTimer.stop();
    const cudaError_t syncStatus = cudaStreamSynchronize(infer.stream());

    if (!masksDownloadOk || !imageDownloadOk || !d2hTimerStopped ||
        syncStatus != cudaSuccess) {
        std::cerr << "[Infer Image] final D2H failed";
        if (syncStatus != cudaSuccess) {
            std::cerr << ": " << cudaGetErrorString(syncStatus);
        }
        std::cerr << std::endl;
        return 9;
    }

    const double preprocessGpuMs = preprocessTimer.elapsedMs();
    const double inferGpuMs = inferTimer.elapsedMs();
    const double postprocessGpuMs = postprocessTimer.elapsedMs();
    const double visualizeGpuMs = visualizeTimer.elapsedMs();
    const double d2hMs = d2hTimer.elapsedMs();
    if (preprocessGpuMs < 0.0 || inferGpuMs < 0.0 ||
        postprocessGpuMs < 0.0 || visualizeGpuMs < 0.0 || d2hMs < 0.0) {
        return 9;
    }

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(probabilityMask, &minimum, &maximum);

    cv::Mat probabilityImage;
    probabilityMask.convertTo(probabilityImage, CV_8U, 255.0);
    if (!cv::imwrite(outputPath, binaryMask) ||
        !cv::imwrite(probabilityPath, probabilityImage) ||
        !cv::imwrite(visualizedPath, visualizedImage)) {
        std::cerr << "[Infer Image] failed to write mask: " << outputPath << std::endl;
        return 10;
    }

    std::cout << "[Infer Image] mask shape: "
              << probabilityMask.cols << 'x' << probabilityMask.rows << std::endl;
    std::cout << "[Infer Image] probability range: ["
              << minimum << ", " << maximum << "]" << std::endl;
    std::cout << "[Infer Image] preprocess_gpu_ms: "
              << preprocessGpuMs << std::endl;
    std::cout << "[Infer Image] infer_gpu_ms: "
              << inferGpuMs << std::endl;
    std::cout << "[Infer Image] postprocess_gpu_ms: "
              << postprocessGpuMs << std::endl;
    std::cout << "[Infer Image] visualize_gpu_ms: "
              << visualizeGpuMs << std::endl;
    std::cout << "[Infer Image] d2h_ms: " << d2hMs << std::endl;
    std::cout << "[Infer Image] saved binary mask: " << outputPath << std::endl;
    std::cout << "[Infer Image] saved probability mask: " << probabilityPath << std::endl;
    std::cout << "[Infer Image] saved visualization: " << visualizedPath << std::endl;
    return 0;
}
