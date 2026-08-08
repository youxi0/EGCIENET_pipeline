#include "evaluation/EngineValidator.h"

#include "common/PreprocessData.h"
#include "common/Timer.h"
#include "infer/TensorRTInfer.h"
#include "postprocess/SegPostprocessor.h"
#include "preprocess/CudaPreprocessor.h"
#include "visualize/Visualizer.h"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

namespace fs = std::filesystem;

struct DatasetSample {
    fs::path imagePath;
    fs::path labelPath;
};

struct AccuracyAccumulator {
    // 行是真值类别，列是预测类别。
    std::array<
        std::array<std::uint64_t, egcinet::segmentation::kClassCount>,
        egcinet::segmentation::kClassCount> confusion{};
    std::uint64_t pixelCount = 0;
};

struct TimingSamples {
    std::vector<double> h2d;
    std::vector<double> preprocess;
    std::vector<double> inference;
    std::vector<double> postprocess;
    std::vector<double> visualization;
    std::vector<double> d2h;
    std::vector<double> endToEnd;
};

// 验证器在开始运行前一次性申请全部 CUDA 显存和 stream。
class InferenceExecutionResources {
public:
    InferenceExecutionResources() = default;
    ~InferenceExecutionResources() {
        release();
    }

    InferenceExecutionResources(const InferenceExecutionResources&) = delete;
    InferenceExecutionResources& operator=(const InferenceExecutionResources&) = delete;

    bool initialize(
        TensorRTInfer& infer,
        SegPostprocessor& postprocessor,
        size_t imageBufferBytes,
        size_t maximumPixels,
        std::string& error
    ) {
        if (imageBufferBytes == 0 || maximumPixels == 0) {
            error = "invalid validation image buffer capacity";
            return false;
        }
        const size_t classMaskBytes = maximumPixels * sizeof(std::uint8_t);

        cudaError_t status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (status == cudaSuccess) {
            status = cudaMalloc(&inputDevice_, infer.inputBufferBytes());
        }
        if (status == cudaSuccess) {
            status = cudaMalloc(&outputDevice_, infer.outputBufferBytes());
        }
        if (status == cudaSuccess) {
            status = cudaMalloc(
                reinterpret_cast<void**>(&imageDevice_), imageBufferBytes);
        }
        if (status == cudaSuccess) {
            status = cudaMalloc(&postprocessDevice_, classMaskBytes);
        }
        if (status != cudaSuccess) {
            error = std::string("failed to initialize validation CUDA resources: ") +
                    cudaGetErrorString(status);
            release();
            return false;
        }
        if (!infer.attachExecutionResources(
                inputDevice_,
                infer.inputBufferBytes(),
                outputDevice_,
                infer.outputBufferBytes(),
                stream_)) {
            error = "failed to attach validation CUDA resources";
            release();
            return false;
        }
        infer_ = &infer;
        imageBufferBytes_ = imageBufferBytes;

        if (!postprocessor.attachOutputBuffer(
                static_cast<std::uint8_t*>(postprocessDevice_),
                classMaskBytes)) {
            error = "failed to attach validation postprocessing buffer";
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

    // 验证工具在预处理前直接上传原图，避免预处理器承担 H2D。
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
            error = "image exceeds initialized validation CUDA capacity";
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
            postprocessor_->detachOutputBuffer();
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

// 统一转成小写，供扩展名和 engine 名称比较使用。
std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

// 判断文件扩展名是否属于 OpenCV 常见图像格式。
bool isSupportedImage(const fs::path& path) {
    static const std::unordered_set<std::string> extensions{
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };
    return extensions.count(toLower(path.extension().string())) != 0;
}

// 相对路径去掉扩展名后作为样本键，允许图像和标签使用不同扩展名。
std::string sampleKey(const fs::path& path, const fs::path& root) {
    fs::path relative = path.lexically_relative(root);
    relative.replace_extension();
    return relative.generic_string();
}

// 递归扫描图像和标签目录，并严格检查一一对应关系。
bool collectDataset(
    const ValidationConfig& config,
    std::vector<DatasetSample>& samples,
    std::string& error
) {
    const fs::path imageRoot(config.imageDirectory);
    const fs::path labelRoot(config.labelDirectory);
    if (!fs::is_directory(imageRoot)) {
        error = "image directory does not exist: " + imageRoot.string();
        return false;
    }
    if (!fs::is_directory(labelRoot)) {
        error = "label directory does not exist: " + labelRoot.string();
        return false;
    }

    std::unordered_map<std::string, fs::path> labels;
    try {
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(labelRoot)) {
            if (!entry.is_regular_file() || !isSupportedImage(entry.path())) {
                continue;
            }
            const std::string key = sampleKey(entry.path(), labelRoot);
            if (!labels.emplace(key, entry.path()).second) {
                error = "duplicate label key: " + key;
                return false;
            }
        }

        std::unordered_set<std::string> imageKeys;
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(imageRoot)) {
            if (!entry.is_regular_file() || !isSupportedImage(entry.path())) {
                continue;
            }

            const std::string key = sampleKey(entry.path(), imageRoot);
            if (!imageKeys.emplace(key).second) {
                error = "duplicate image key: " + key;
                return false;
            }

            const auto label = labels.find(key);
            if (label == labels.end()) {
                error = "label not found for image: " + entry.path().string();
                return false;
            }
            samples.push_back({entry.path(), label->second});
        }
    } catch (const fs::filesystem_error& exception) {
        error = exception.what();
        return false;
    }

    std::sort(samples.begin(), samples.end(), [](const DatasetSample& lhs, const DatasetSample& rhs) {
        return lhs.imagePath.generic_string() < rhs.imagePath.generic_string();
    });

    if (samples.empty()) {
        error = "no image-label pairs found";
        return false;
    }
    if (config.maxImages > 0 && samples.size() > config.maxImages) {
        samples.resize(config.maxImages);
    }
    return true;
}

// 在正式验证前扫描图片尺寸，确保后续循环中不会发生任何显存扩容。
bool inspectDatasetCapacity(
    const std::vector<DatasetSample>& samples,
    size_t& maximumImageBytes,
    size_t& maximumPixels,
    std::string& error
) {
    maximumImageBytes = 0;
    maximumPixels = 0;
    for (const DatasetSample& sample : samples) {
        const cv::Mat image = cv::imread(sample.imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty() || image.type() != CV_8UC3 || image.cols <= 0 || image.rows <= 0 ||
            image.cols > std::numeric_limits<int>::max() / 3) {
            error = "failed to inspect validation image: " + sample.imagePath.string();
            return false;
        }
        const size_t imageStep = static_cast<size_t>(image.cols) * 3;
        if (image.step < imageStep ||
            static_cast<size_t>(image.rows) >
                std::numeric_limits<size_t>::max() / imageStep) {
            error = "validation image size overflow: " + sample.imagePath.string();
            return false;
        }
        const size_t imageBytes = imageStep * static_cast<size_t>(image.rows);

        const size_t pixels = (imageBytes / imageStep) * (imageStep / 3);
        maximumPixels = std::max(maximumPixels, pixels);
        maximumImageBytes = std::max(maximumImageBytes, imageBytes);
    }
    return maximumImageBytes > 0 && maximumPixels > 0;
}

// 读取一个 BGR 图像及其 0~4 灰度类别标签，并检查原始尺寸一致性。
bool readSample(
    const DatasetSample& sample,
    cv::Mat& image,
    cv::Mat& label,
    std::string& error
) {
    image = cv::imread(sample.imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        error = "failed to read image: " + sample.imagePath.string();
        return false;
    }

    label = cv::imread(sample.labelPath.string(), cv::IMREAD_GRAYSCALE);
    if (label.empty()) {
        error = "failed to read label: " + sample.labelPath.string();
        return false;
    }
    if (image.size() != label.size()) {
        std::ostringstream stream;
        stream << "image and label dimensions differ: "
               << sample.imagePath.string() << " is " << image.cols << 'x' << image.rows
               << ", " << sample.labelPath.string() << " is "
               << label.cols << 'x' << label.rows;
        error = stream.str();
        return false;
    }
    double labelMaximum = 0.0;
    cv::minMaxLoc(label, nullptr, &labelMaximum);
    if (labelMaximum >= egcinet::segmentation::kClassCount) {
        error = "label contains class id outside [0, 4]: " + sample.labelPath.string();
        return false;
    }
    return true;
}

// 在同一 CUDA stream 中依次提交 H2D、预处理、TensorRT 推理和后处理。
bool enqueueInference(
    TensorRTInfer& infer,
    InferenceExecutionResources& resources,
    CudaPreprocessor& preprocessor,
    SegPostprocessor& postprocessor,
    const cv::Mat& image,
    std::string& error
) {
    size_t imageStep = 0;
    if (!resources.uploadImage(image, imageStep, error)) {
        return false;
    }

    PreprocessResult preprocessResult;
    if (!preprocessor.process(
            resources.imageDeviceBuffer(),
            resources.imageBufferBytes(),
            image.cols,
            image.rows,
            imageStep,
            infer.inputDeviceBuffer(),
            infer.inputBufferBytes(),
            infer.inputElementSize(),
            preprocessResult,
            infer.stream())) {
        error = "CUDA preprocessing failed";
        return false;
    }
    if (!infer.inferFromDevice()) {
        error = "TensorRT inference failed";
        return false;
    }
    if (!postprocessor.process(
            infer.outputDeviceBuffer(),
            infer.outputBufferBytes(),
            infer.outputElementSize(),
            infer.outputChannels(),
            infer.outputWidth(),
            infer.outputHeight(),
            image.cols,
            image.rows,
            infer.stream())) {
        error = "CUDA postprocessing failed";
        return false;
    }
    return true;
}

// 同步指定 stream，并把 CUDA 错误转换成可读文本。
bool synchronize(cudaStream_t stream, std::string& error) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status == cudaSuccess) {
        return true;
    }
    error = std::string("CUDA stream synchronization failed: ") + cudaGetErrorString(status);
    return false;
}

// 累加多类别混淆矩阵；类别图和标签都直接使用训练定义的 0~4 ID。
void accumulateAccuracy(
    const cv::Mat& classMask,
    const cv::Mat& label,
    AccuracyAccumulator& accumulator
) {
    for (int row = 0; row < label.rows; ++row) {
        const std::uint8_t* prediction = classMask.ptr<std::uint8_t>(row);
        const std::uint8_t* groundTruth = label.ptr<std::uint8_t>(row);
        for (int column = 0; column < label.cols; ++column) {
            accumulator.confusion[groundTruth[column]][prediction[column]]++;
        }
    }
    accumulator.pixelCount += static_cast<std::uint64_t>(label.total());
}

// 将整个验证集的累计值转换成最终精度指标。
AccuracySummary finishAccuracy(
    const AccuracyAccumulator& accumulator,
    std::size_t imageCount
) {
    AccuracySummary summary;
    summary.imageCount = imageCount;
    summary.pixelCount = accumulator.pixelCount;

    std::uint64_t correctPixels = 0;
    std::size_t activeClasses = 0;
    for (int classId = 0; classId < egcinet::segmentation::kClassCount; ++classId) {
        std::uint64_t truthPixels = 0;
        std::uint64_t predictedPixels = 0;
        for (int other = 0; other < egcinet::segmentation::kClassCount; ++other) {
            truthPixels += accumulator.confusion[classId][other];
            predictedPixels += accumulator.confusion[other][classId];
        }

        const std::uint64_t truePositive = accumulator.confusion[classId][classId];
        correctPixels += truePositive;
        const std::uint64_t falsePositive = predictedPixels - truePositive;
        const std::uint64_t falseNegative = truthPixels - truePositive;
        const std::uint64_t unionPixels =
            truePositive + falsePositive + falseNegative;
        if (unionPixels == 0) {
            continue;
        }

        const double tp = static_cast<double>(truePositive);
        const double precisionDenominator = static_cast<double>(predictedPixels);
        const double recallDenominator = static_cast<double>(truthPixels);
        const double diceDenominator =
            2.0 * tp + static_cast<double>(falsePositive + falseNegative);
        summary.classDice[classId] = diceDenominator == 0.0
            ? 0.0 : 2.0 * tp / diceDenominator;
        summary.classIou[classId] = tp / static_cast<double>(unionPixels);
        summary.meanDice += summary.classDice[classId];
        summary.meanIou += summary.classIou[classId];
        summary.macroPrecision += precisionDenominator == 0.0
            ? 0.0 : tp / precisionDenominator;
        summary.macroRecall += recallDenominator == 0.0
            ? 0.0 : tp / recallDenominator;
        ++activeClasses;
    }

    if (accumulator.pixelCount > 0) {
        summary.pixelAccuracy = static_cast<double>(correctPixels) /
            static_cast<double>(accumulator.pixelCount);
    }
    if (activeClasses > 0) {
        const double divisor = static_cast<double>(activeClasses);
        summary.meanDice /= divisor;
        summary.meanIou /= divisor;
        summary.macroPrecision /= divisor;
        summary.macroRecall /= divisor;
    }
    return summary;
}

// 对全部配对样本执行推理，并与 5 类标签计算像素精度和宏平均指标。
bool validateAccuracy(
    TensorRTInfer& infer,
    InferenceExecutionResources& resources,
    CudaPreprocessor& preprocessor,
    SegPostprocessor& postprocessor,
    const std::vector<DatasetSample>& samples,
    AccuracySummary& summary,
    std::string& error
) {
    AccuracyAccumulator accumulator;
    cv::Mat image;
    cv::Mat label;
    cv::Mat classMask;

    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (!readSample(samples[index], image, label, error)) {
            return false;
        }
        if (!enqueueInference(
                infer, resources, preprocessor, postprocessor, image, error)) {
            error += ": " + samples[index].imagePath.string();
            return false;
        }
        if (!postprocessor.enqueueDownload(classMask, infer.stream())) {
            error = "failed to enqueue mask D2H: " + samples[index].imagePath.string();
            return false;
        }
        if (!synchronize(infer.stream(), error)) {
            return false;
        }
        if (classMask.size() != label.size() || classMask.type() != CV_8UC1) {
            error = "postprocess output does not match label: " + samples[index].imagePath.string();
            return false;
        }

        accumulateAccuracy(classMask, label, accumulator);
        if ((index + 1) % 50 == 0 || index + 1 == samples.size()) {
            std::cout << "[Validate] accuracy " << (index + 1)
                      << '/' << samples.size() << std::endl;
        }
    }

    summary = finishAccuracy(accumulator, samples.size());
    return true;
}

// 正式计时前运行完整链路，使显存、CUDA 上下文和 TensorRT 状态稳定。
bool runWarmup(
    TensorRTInfer& infer,
    InferenceExecutionResources& resources,
    CudaPreprocessor& preprocessor,
    SegPostprocessor& postprocessor,
    Visualizer& visualizer,
    const std::vector<DatasetSample>& samples,
    const ValidationConfig& config,
    std::string& error
) {
    cv::Mat image;
    cv::Mat classMask;
    cv::Mat visualizedImage;

    for (std::size_t index = 0; index < config.warmupIterations; ++index) {
        image = cv::imread(
            samples[index % samples.size()].imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            error = "failed to read warmup image: "
                + samples[index % samples.size()].imagePath.string();
            return false;
        }
        if (!enqueueInference(
                infer, resources, preprocessor, postprocessor, image, error)) {
            return false;
        }
        const size_t imageStep = static_cast<size_t>(image.cols) * 3;
        if (config.includeVisualization && !visualizer.process(
                resources.imageDeviceBuffer(),
                imageStep,
                image.cols,
                image.rows,
                postprocessor.classMaskDeviceBuffer(),
                infer.stream())) {
            error = "CUDA visualization failed during warmup";
            return false;
        }
        if (!postprocessor.enqueueDownload(classMask, infer.stream())) {
            error = "failed to enqueue warmup mask D2H";
            return false;
        }
        if (config.includeVisualization && !visualizer.enqueueDownload(
                resources.imageDeviceBuffer(),
                imageStep,
                image.cols,
                image.rows,
                visualizedImage,
                infer.stream())) {
            error = "failed to enqueue warmup visualization D2H";
            return false;
        }
        if (!synchronize(infer.stream(), error)) {
            return false;
        }
    }
    return true;
}

// 读取一次 CUDA Event 耗时，并统一处理计时错误。
bool timerValue(CudaEventTimer& timer, double& value, std::string& error) {
    value = timer.elapsedMs();
    if (value >= 0.0) {
        return true;
    }
    error = "failed to read CUDA event timing";
    return false;
}

// 分阶段记录 GPU 时间，并记录不含图像解码的串行端到端时间。
bool benchmarkPerformance(
    TensorRTInfer& infer,
    InferenceExecutionResources& resources,
    CudaPreprocessor& preprocessor,
    SegPostprocessor& postprocessor,
    Visualizer& visualizer,
    const std::vector<DatasetSample>& samples,
    const ValidationConfig& config,
    TimingSamples& timings,
    std::string& error
) {
    if (!runWarmup(
            infer, resources, preprocessor, postprocessor, visualizer, samples, config, error)) {
        return false;
    }

    CudaEventTimer h2dTimer(infer.stream());
    CudaEventTimer preprocessTimer(infer.stream());
    CudaEventTimer inferenceTimer(infer.stream());
    CudaEventTimer postprocessTimer(infer.stream());
    CudaEventTimer visualizationTimer(infer.stream());
    CudaEventTimer d2hTimer(infer.stream());
    if (!h2dTimer.isValid() || !preprocessTimer.isValid() || !inferenceTimer.isValid() ||
        !postprocessTimer.isValid() || !visualizationTimer.isValid() ||
        !d2hTimer.isValid()) {
        error = "failed to create CUDA event timers";
        return false;
    }

    timings.h2d.reserve(config.benchmarkIterations);
    timings.preprocess.reserve(config.benchmarkIterations);
    timings.inference.reserve(config.benchmarkIterations);
    timings.postprocess.reserve(config.benchmarkIterations);
    timings.visualization.reserve(config.benchmarkIterations);
    timings.d2h.reserve(config.benchmarkIterations);
    timings.endToEnd.reserve(config.benchmarkIterations);

    cv::Mat image;
    cv::Mat classMask;
    cv::Mat visualizedImage;
    for (std::size_t index = 0; index < config.benchmarkIterations; ++index) {
        image = cv::imread(
            samples[index % samples.size()].imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            error = "failed to read benchmark image: "
                + samples[index % samples.size()].imagePath.string();
            return false;
        }

        CpuTimer endToEndTimer;
        PreprocessResult preprocessResult;
        size_t imageStep = 0;
        if (!h2dTimer.start() ||
            !resources.uploadImage(image, imageStep, error) ||
            !h2dTimer.stop()) {
            error = error.empty() ? "CUDA H2D failed during benchmark" : error;
            return false;
        }
        if (!preprocessTimer.start()) {
            error = "failed to start preprocessing timer";
            return false;
        }
        const bool preprocessOk = preprocessor.process(
            resources.imageDeviceBuffer(),
            resources.imageBufferBytes(),
            image.cols,
            image.rows,
            imageStep,
            infer.inputDeviceBuffer(),
            infer.inputBufferBytes(),
            infer.inputElementSize(),
            preprocessResult,
            infer.stream());
        if (!preprocessTimer.stop() || !preprocessOk) {
            error = "CUDA preprocessing failed during benchmark";
            return false;
        }

        if (!inferenceTimer.start()) {
            error = "failed to start inference timer";
            return false;
        }
        const bool inferenceOk = infer.inferFromDevice();
        if (!inferenceTimer.stop() || !inferenceOk) {
            error = "TensorRT inference failed during benchmark";
            return false;
        }

        if (!postprocessTimer.start()) {
            error = "failed to start postprocessing timer";
            return false;
        }
        const bool postprocessOk = postprocessor.process(
            infer.outputDeviceBuffer(),
            infer.outputBufferBytes(),
            infer.outputElementSize(),
            infer.outputChannels(),
            infer.outputWidth(),
            infer.outputHeight(),
            image.cols,
            image.rows,
            infer.stream());
        if (!postprocessTimer.stop() || !postprocessOk) {
            error = "CUDA postprocessing failed during benchmark";
            return false;
        }

        if (config.includeVisualization) {
            if (!visualizationTimer.start()) {
                error = "failed to start visualization timer";
                return false;
            }
            const bool visualizationOk = visualizer.process(
                resources.imageDeviceBuffer(),
                imageStep,
                image.cols,
                image.rows,
                postprocessor.classMaskDeviceBuffer(),
                infer.stream());
            if (!visualizationTimer.stop() || !visualizationOk) {
                error = "CUDA visualization failed during benchmark";
                return false;
            }
        }

        if (!d2hTimer.start()) {
            error = "failed to start D2H timer";
            return false;
        }
        const bool maskDownloadOk = postprocessor.enqueueDownload(classMask, infer.stream());
        const bool visualizationDownloadOk = !config.includeVisualization ||
            visualizer.enqueueDownload(
                resources.imageDeviceBuffer(),
                imageStep,
                image.cols,
                image.rows,
                visualizedImage,
                infer.stream());
        if (!d2hTimer.stop() || !maskDownloadOk || !visualizationDownloadOk) {
            error = "failed to enqueue benchmark D2H";
            return false;
        }
        if (!synchronize(infer.stream(), error)) {
            return false;
        }
        timings.endToEnd.push_back(endToEndTimer.elapsedMs());

        double value = 0.0;
        if (!timerValue(h2dTimer, value, error)) {
            return false;
        }
        timings.h2d.push_back(value);
        if (!timerValue(preprocessTimer, value, error)) {
            return false;
        }
        timings.preprocess.push_back(value);
        if (!timerValue(inferenceTimer, value, error)) {
            return false;
        }
        timings.inference.push_back(value);
        if (!timerValue(postprocessTimer, value, error)) {
            return false;
        }
        timings.postprocess.push_back(value);
        if (config.includeVisualization) {
            if (!timerValue(visualizationTimer, value, error)) {
                return false;
            }
            timings.visualization.push_back(value);
        }
        if (!timerValue(d2hTimer, value, error)) {
            return false;
        }
        timings.d2h.push_back(value);
    }
    return true;
}

// 在线性插值下计算已排序样本的指定分位数。
double percentile(const std::vector<double>& sortedValues, double quantile) {
    if (sortedValues.empty()) {
        return 0.0;
    }
    const double position = quantile * static_cast<double>(sortedValues.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sortedValues[lower] * (1.0 - weight) + sortedValues[upper] * weight;
}

// 从原始耗时样本生成平均值及 P50、P95、P99。
TimingDistribution summarizeTiming(std::vector<double> values) {
    TimingDistribution summary;
    if (values.empty()) {
        return summary;
    }
    summary.meanMs = std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    summary.p50Ms = percentile(values, 0.50);
    summary.p95Ms = percentile(values, 0.95);
    summary.p99Ms = percentile(values, 0.99);
    return summary;
}

// 汇总所有阶段耗时，并计算单 stream 串行 FPS。
PerformanceSummary finishPerformance(TimingSamples timings) {
    PerformanceSummary summary;
    summary.h2d = summarizeTiming(std::move(timings.h2d));
    summary.preprocess = summarizeTiming(std::move(timings.preprocess));
    summary.inference = summarizeTiming(std::move(timings.inference));
    summary.postprocess = summarizeTiming(std::move(timings.postprocess));
    summary.visualization = summarizeTiming(std::move(timings.visualization));
    summary.d2h = summarizeTiming(std::move(timings.d2h));
    summary.endToEnd = summarizeTiming(std::move(timings.endToEnd));
    if (summary.endToEnd.meanMs > 0.0) {
        summary.fps = 1000.0 / summary.endToEnd.meanMs;
    }
    return summary;
}

// 按 CSV 规则转义字符串字段。
std::string csvString(const std::string& value) {
    std::string escaped = value;
    std::size_t position = 0;
    while ((position = escaped.find('"', position)) != std::string::npos) {
        escaped.insert(position, 1, '"');
        position += 2;
    }
    return '"' + escaped + '"';
}

// 依次写入一个阶段的 mean、P50、P95 和 P99。
void writeTiming(std::ostream& output, const TimingDistribution& timing) {
    output << ',' << timing.meanMs
           << ',' << timing.p50Ms
           << ',' << timing.p95Ms
           << ',' << timing.p99Ms;
}

// 输出每个 engine 的完整结果及其相对 FP32 的精度变化。
bool writeCsv(
    const std::string& outputPath,
    const std::vector<EngineValidationResult>& results,
    std::string& error
) {
    const fs::path path(outputPath);
    try {
        if (!path.parent_path().empty()) {
            fs::create_directories(path.parent_path());
        }
    } catch (const fs::filesystem_error& exception) {
        error = exception.what();
        return false;
    }

    std::ofstream output(path);
    if (!output) {
        error = "failed to open CSV output: " + path.string();
        return false;
    }

    output << "engine,engine_path,images,pixels,pixel_accuracy,mean_dice,"
              "mean_iou,macro_precision,macro_recall,"
              "background_dice,burn_dice,crack_tear_dice,material_loss_dice,deformation_dice,"
              "background_iou,burn_iou,crack_tear_iou,material_loss_iou,deformation_iou,"
              "h2d_mean_ms,h2d_p50_ms,h2d_p95_ms,h2d_p99_ms,"
              "pre_mean_ms,pre_p50_ms,pre_p95_ms,pre_p99_ms,"
              "infer_mean_ms,infer_p50_ms,infer_p95_ms,infer_p99_ms,"
              "post_mean_ms,post_p50_ms,post_p95_ms,post_p99_ms,"
              "visualize_mean_ms,visualize_p50_ms,visualize_p95_ms,visualize_p99_ms,"
              "d2h_mean_ms,d2h_p50_ms,d2h_p95_ms,d2h_p99_ms,"
              "e2e_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_p99_ms,fps,"
              "pixel_accuracy_delta_vs_fp32,dice_delta_vs_fp32,iou_delta_vs_fp32\n";
    output << std::fixed << std::setprecision(8);

    const EngineValidationResult* fp32 = nullptr;
    for (const EngineValidationResult& result : results) {
        if (toLower(result.name) == "fp32") {
            fp32 = &result;
            break;
        }
    }

    for (const EngineValidationResult& result : results) {
        output << csvString(result.name)
               << ',' << csvString(result.enginePath)
               << ',' << result.accuracy.imageCount
               << ',' << result.accuracy.pixelCount
               << ',' << result.accuracy.pixelAccuracy
               << ',' << result.accuracy.meanDice
               << ',' << result.accuracy.meanIou
               << ',' << result.accuracy.macroPrecision
               << ',' << result.accuracy.macroRecall;
        for (const double value : result.accuracy.classDice) {
            output << ',' << value;
        }
        for (const double value : result.accuracy.classIou) {
            output << ',' << value;
        }
        writeTiming(output, result.performance.h2d);
        writeTiming(output, result.performance.preprocess);
        writeTiming(output, result.performance.inference);
        writeTiming(output, result.performance.postprocess);
        writeTiming(output, result.performance.visualization);
        writeTiming(output, result.performance.d2h);
        writeTiming(output, result.performance.endToEnd);
        output << ',' << result.performance.fps;
        if (fp32 != nullptr) {
            output << ',' << result.accuracy.pixelAccuracy - fp32->accuracy.pixelAccuracy
                   << ',' << result.accuracy.meanDice - fp32->accuracy.meanDice
                   << ',' << result.accuracy.meanIou - fp32->accuracy.meanIou;
        } else {
            output << ",,,";
        }
        output << '\n';
    }
    return true;
}

// 在创建 CUDA/TensorRT 对象前检查公共配置。
bool validateConfig(const ValidationConfig& config, std::string& error) {
    if (config.imageDirectory.empty() || config.labelDirectory.empty()) {
        error = "image and label directories are required";
        return false;
    }
    if (config.outputCsv.empty()) {
        error = "CSV output path is empty";
        return false;
    }
    if (config.benchmarkIterations == 0) {
        error = "benchmark iterations must be greater than zero";
        return false;
    }
    for (float value : config.std) {
        if (value <= std::numeric_limits<float>::epsilon()) {
            error = "all std values must be greater than zero";
            return false;
        }
    }
    return true;
}

} // namespace

EngineValidator::EngineValidator(ValidationConfig config)
    : config_(std::move(config)) {
}

bool EngineValidator::run(
    const std::vector<EngineValidationTarget>& targets,
    std::vector<EngineValidationResult>& results
) {
    results.clear();
    lastError_.clear();
    if (!validateConfig(config_, lastError_)) {
        return false;
    }
    if (targets.empty()) {
        lastError_ = "at least one engine is required";
        return false;
    }

    std::vector<DatasetSample> samples;
    if (!collectDataset(config_, samples, lastError_)) {
        return false;
    }
    size_t maximumImageBytes = 0;
    size_t maximumPixels = 0;
    if (!inspectDatasetCapacity(
            samples, maximumImageBytes, maximumPixels, lastError_)) {
        return false;
    }
    std::cout << "[Validate] paired samples: " << samples.size() << std::endl;

    for (const EngineValidationTarget& target : targets) {
        if (target.name.empty() || target.enginePath.empty()) {
            lastError_ = "engine name and path cannot be empty";
            return false;
        }
        if (!fs::is_regular_file(target.enginePath)) {
            lastError_ = "engine does not exist: " + target.enginePath;
            return false;
        }

        std::cout << "[Validate] loading " << target.name
                  << ": " << target.enginePath << std::endl;
        TensorRTInfer infer(target.enginePath);
        if (!infer.load()) {
            lastError_ = "failed to load engine: " + target.enginePath;
            return false;
        }

        PreprocessConfig preprocessConfig;
        preprocessConfig.inputWidth = infer.inputWidth();
        preprocessConfig.inputHeight = infer.inputHeight();
        preprocessConfig.mean = config_.mean;
        preprocessConfig.std = config_.std;
        CudaPreprocessor preprocessor(preprocessConfig);
        SegPostprocessor postprocessor;
        Visualizer visualizer;
        InferenceExecutionResources executionResources;
        if (!executionResources.initialize(
                infer,
                postprocessor,
                maximumImageBytes,
                maximumPixels,
                lastError_)) {
            return false;
        }

        EngineValidationResult result;
        result.name = target.name;
        result.enginePath = target.enginePath;
        if (!validateAccuracy(
                infer,
                executionResources,
                preprocessor,
                postprocessor,
                samples,
                result.accuracy,
                lastError_)) {
            lastError_ = target.name + " accuracy validation failed: " + lastError_;
            return false;
        }

        std::cout << "[Validate] benchmarking " << target.name
                  << ", warmup=" << config_.warmupIterations
                  << ", iterations=" << config_.benchmarkIterations << std::endl;
        TimingSamples timings;
        if (!benchmarkPerformance(
                infer,
                executionResources,
                preprocessor,
                postprocessor,
                visualizer,
                samples,
                config_,
                timings,
                lastError_)) {
            lastError_ = target.name + " benchmark failed: " + lastError_;
            return false;
        }
        result.performance = finishPerformance(std::move(timings));
        results.push_back(std::move(result));
    }

    if (!writeCsv(config_.outputCsv, results, lastError_)) {
        return false;
    }
    std::cout << "[Validate] CSV: " << config_.outputCsv << std::endl;
    return true;
}

const std::string& EngineValidator::lastError() const noexcept {
    return lastError_;
}
