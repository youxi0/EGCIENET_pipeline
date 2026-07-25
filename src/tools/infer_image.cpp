#include "common/PreprocessData.h"
#include "common/Timer.h"
#include "infer/TensorRTInfer.h"
#include "postprocess/SegPostprocessor.h"
#include "preprocess/CudaPreprocessor.h"
#include "visualize/Visualizer.h"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace {

// 单图工具自行持有执行显存和 stream，TensorRTInfer 只保存非拥有引用。
class ExecutionResources {
public:
    ~ExecutionResources() {
        release();
    }

    ExecutionResources(const ExecutionResources&) = delete;
    ExecutionResources& operator=(const ExecutionResources&) = delete;

    ExecutionResources() = default;

    bool initialize(
        TensorRTInfer& infer,
        SegPostprocessor& postprocessor,
        int imageWidth,
        int imageHeight
    ) {
        if (imageWidth <= 0 || imageHeight <= 0) {
            return false;
        }
        const size_t width = static_cast<size_t>(imageWidth);
        const size_t height = static_cast<size_t>(imageHeight);
        if (height > std::numeric_limits<size_t>::max() / width) {
            return false;
        }
        const size_t pixels = width * height;
        if (pixels > std::numeric_limits<size_t>::max() / 5) {
            return false;
        }
        const size_t imageBytes = pixels * 3;
        const size_t probabilityBytes = pixels * sizeof(float);
        const size_t binaryBytes = pixels * sizeof(std::uint8_t);
        const size_t postprocessBytes = probabilityBytes + binaryBytes;

        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc(&inputDevice_, infer.inputBufferBytes()) != cudaSuccess ||
            cudaMalloc(&outputDevice_, infer.outputBufferBytes()) != cudaSuccess ||
            cudaMalloc(
                reinterpret_cast<void**>(&imageDevice_), imageBytes) != cudaSuccess ||
            cudaMalloc(&postprocessDevice_, postprocessBytes) != cudaSuccess) {
            std::cerr << "[Infer Image] failed to allocate CUDA execution resources"
                      << std::endl;
            release();
            return false;
        }
        if (!infer.attachExecutionResources(
                inputDevice_,
                infer.inputBufferBytes(),
                outputDevice_,
                infer.outputBufferBytes(),
                stream_)) {
            release();
            return false;
        }
        infer_ = &infer;
        imageBufferBytes_ = imageBytes;

        float* probabilityDevice = static_cast<float*>(postprocessDevice_);
        auto* binaryDevice = static_cast<std::uint8_t*>(postprocessDevice_) +
                             probabilityBytes;
        if (!postprocessor.attachOutputBuffers(
                probabilityDevice,
                probabilityBytes,
                binaryDevice,
                binaryBytes)) {
            release();
            return false;
        }
        postprocessor_ = &postprocessor;
        return true;
    }

    unsigned char* imageDeviceBuffer() noexcept {
        return imageDevice_;
    }

    size_t imageBufferBytes() const noexcept {
        return imageBufferBytes_;
    }

    // 单图工具在预处理前直接执行 H2D，预处理器只接收设备指针。
    bool uploadImage(
        const cv::Mat& image,
        size_t& imageStep,
        std::string& error
    ) {
        imageStep = 0;
        if (image.empty() || image.type() != CV_8UC3 || image.cols <= 0 || image.rows <= 0 ||
            image.cols > std::numeric_limits<int>::max() / 3) {
            error = "image must be a non-empty CV_8UC3 BGR image";
            return false;
        }
        imageStep = static_cast<size_t>(image.cols) * 3;
        if (image.step < imageStep ||
            static_cast<size_t>(image.rows) >
                std::numeric_limits<size_t>::max() / imageStep) {
            error = "invalid BGR image step or dimensions";
            return false;
        }
        const size_t imageBytes = imageStep * static_cast<size_t>(image.rows);
        if (imageDevice_ == nullptr || stream_ == nullptr || imageBytes > imageBufferBytes_) {
            error = "image exceeds initialized CUDA resource capacity";
            return false;
        }

        const cudaError_t status = cudaMemcpy2DAsync(
            imageDevice_,
            imageStep,
            image.data,
            image.step,
            imageStep,
            static_cast<size_t>(image.rows),
            cudaMemcpyHostToDevice,
            stream_
        );
        if (status != cudaSuccess) {
            error = std::string("cudaMemcpy2DAsync H2D failed: ")
                + cudaGetErrorString(status);
            return false;
        }
        return true;
    }

private:
    void release() noexcept {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
        if (infer_ != nullptr) {
            infer_->detachExecutionResources();
            infer_ = nullptr;
        }
        if (postprocessor_ != nullptr) {
            postprocessor_->detachOutputBuffers();
            postprocessor_ = nullptr;
        }
        if (postprocessDevice_ != nullptr) {
            cudaFree(postprocessDevice_);
            postprocessDevice_ = nullptr;
        }
        if (imageDevice_ != nullptr) {
            cudaFree(imageDevice_);
            imageDevice_ = nullptr;
        }
        imageBufferBytes_ = 0;
        if (outputDevice_ != nullptr) {
            cudaFree(outputDevice_);
            outputDevice_ = nullptr;
        }
        if (inputDevice_ != nullptr) {
            cudaFree(inputDevice_);
            inputDevice_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

private:
    TensorRTInfer* infer_ = nullptr;
    SegPostprocessor* postprocessor_ = nullptr;
    void* inputDevice_ = nullptr;
    void* outputDevice_ = nullptr;
    unsigned char* imageDevice_ = nullptr;
    size_t imageBufferBytes_ = 0;
    void* postprocessDevice_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

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

    // 预处理尺寸直接取自 engine，避免配置值与输入 tensor 尺寸不一致。
    CudaPreprocessor preprocessor(infer.inputWidth(), infer.inputHeight());
    SegPostprocessor postprocessor;
    Visualizer visualizer;
    ExecutionResources executionResources;
    if (!executionResources.initialize(
            infer, postprocessor, image.cols, image.rows)) {
        return 4;
    }

    size_t imageStep = 0;
    std::string h2dError;

    CudaEventTimer h2dTimer(infer.stream());
    CudaEventTimer preprocessTimer(infer.stream());
    CudaEventTimer inferTimer(infer.stream());
    CudaEventTimer postprocessTimer(infer.stream());
    CudaEventTimer visualizeTimer(infer.stream());
    CudaEventTimer d2hTimer(infer.stream());
    if (!h2dTimer.isValid() || !preprocessTimer.isValid() || !inferTimer.isValid() ||
        !postprocessTimer.isValid() || !visualizeTimer.isValid() ||
        !d2hTimer.isValid()) {
        return 4;
    }

    PreprocessResult preprocessResult;
    if (!h2dTimer.start() ||
        !executionResources.uploadImage(image, imageStep, h2dError) ||
        !h2dTimer.stop()) {
        std::cerr << "[Infer Image] H2D failed: " << h2dError << std::endl;
        return 5;
    }
    if (!preprocessTimer.start()) {
        return 5;
    }

    const bool preprocessOk = preprocessor.process(
            executionResources.imageDeviceBuffer(),
            executionResources.imageBufferBytes(),
            image.cols,
            image.rows,
            imageStep,
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
        executionResources.imageDeviceBuffer(),
        imageStep,
        image.cols,
        image.rows,
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
        executionResources.imageDeviceBuffer(),
        imageStep,
        image.cols,
        image.rows,
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

    const double h2dGpuMs = h2dTimer.elapsedMs();
    const double preprocessGpuMs = preprocessTimer.elapsedMs();
    const double inferGpuMs = inferTimer.elapsedMs();
    const double postprocessGpuMs = postprocessTimer.elapsedMs();
    const double visualizeGpuMs = visualizeTimer.elapsedMs();
    const double d2hMs = d2hTimer.elapsedMs();
    if (h2dGpuMs < 0.0 || preprocessGpuMs < 0.0 || inferGpuMs < 0.0 ||
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
    std::cout << "[Infer Image] h2d_gpu_ms: " << h2dGpuMs << std::endl;
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
