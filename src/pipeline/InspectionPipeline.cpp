#include "pipeline/InspectionPipeline.h"
#include "common/Timer.h"
#include "utils/FileLogger.h"
#include "utils/Int8CompareSaver.h"

#include <cuda_fp16.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

#define DEBUG_PREPROCESS_COMPARE 0

namespace blade::pipeline {

InspectionPipeline::InspectionPipeline(std::unique_ptr<ImageSource> source, const PipelineConfig& config)
    : source_(std::move(source)),
      config_(config),
      rawQueue_(static_cast<size_t>(config.queueSize)),
      preprocessQueue_(static_cast<size_t>(config.queueSize)),
      inferQueue_(static_cast<size_t>(config.queueSize)),
      int8CompareQueue_(static_cast<size_t>(config.queueSize))


InspectionPipeline::~InspectionPipeline() {
    stop();
}

bool InspectionPipeline::start() {
    if (running_) {
        return true;
    }

    // 新增模块: pipeline启动时打开日志文件，后续各线程共享同一个线程安全logger。
    auto& logger = utils::FileLogger::instance();
    if (!logger.isOpen()) {
        logger.open(config_.logDir);
    }
    logger.info("inspection pipeline starting");
    logger.info("engine path: " + config_.enginePath);

    // INT8对比线程是可选旁路；没有INT8 engine时自动关闭，避免影响FP16 baseline主流程。
    if (config_.enableInt8Compare) {
        if (config_.int8EnginePath.empty()) {
            logger.warning("[INT8 Compare] enabled but int8 engine path is empty");
            config_.enableInt8Compare = false;
        } else {
            logger.info("[INT8 Compare] engine path: " + config_.int8EnginePath);
        }
    }

    if (!source_) {
        std::cerr << "Image source is null." << std::endl;
        logger.error("image source is null");
        return false;
    }

    if (!source_->open()) {
        std::cerr << "Failed to open image source." << std::endl;
        logger.error("failed to open image source");
        return false;
    }


    running_ = true;
    lastPongMs_ = blade::net::nowMs();

    // 处理线程们
    captureThread_ = std::thread(&InspectionPipeline::captureLoop, this);
    
    if (config_.useCudaPreprocess) {
        gpuInferThread_ = std::thread(&InspectionPipeline::gpuInferLoop, this);
    } else {
        preprocessThread_ = std::thread(&InspectionPipeline::preprocessLoop, this);
        inferThread_ = std::thread(&InspectionPipeline::inferLoop, this);
    }

    postprocessThread_ = std::thread(&InspectionPipeline::postprocessLoop, this);
    if (config_.enableInt8Compare) {
        int8CompareThread_ = std::thread(&InspectionPipeline::int8CompareLoop, this);
    }

    logger.info("inspection pipeline started");
    return true;
}

void InspectionPipeline::stop() {
    // 即使running_已经被工作线程置为false，也要继续唤醒队列并join线程。
    // 否则离线folder跑完后，析构函数再次调用stop()时可能留下joinable线程。
    running_ = false;

    paused_ = true;
    processingEnabled_ = false;
    
    rawQueue_.stop();
    preprocessQueue_.stop();
    inferQueue_.stop();
    int8CompareQueue_.stop();
 

    if (source_) {
        source_->release();
    }

    // 处理线程们 

    if (captureThread_.joinable()) {
        captureThread_.join();
    }
    if (preprocessThread_.joinable()){
        preprocessThread_.join();
    }
    if (gpuInferThread_.joinable()) {
        gpuInferThread_.join();
    }
    if (postprocessThread_.joinable()) {
        postprocessThread_.join();
    }
    if (int8CompareThread_.joinable()) {
        int8CompareThread_.join();
    }

    utils::FileLogger::instance().info("inspection pipeline stopped");
    utils::FileLogger::instance().close();
}

bool InspectionPipeline::isRunning() const {
    return running_.load();
}



void InspectionPipeline::captureLoop() {
    auto& logger = utils::FileLogger::instance();

    while (running_) {
        
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (!processingEnabled_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        FrameData raw;
        if (!source_->read(raw)) {
            std::cerr << "[CAPTURE] source finished or read failed" << std::endl;
            logger.warning("[CAPTURE] source finished or read failed");

            rawQueue_.stop();
            break;
        }
        if (!rawQueue_.push(std::move(raw))) {
            break;
        }


    }

    std::cout << "[CAPTURE] exit" << std::endl;
    logger.info("[CAPTURE] exit");
}

void InspectionPipeline::preprocessLoop() {
    auto& logger = utils::FileLogger::instance();
    Preprocessor preprocessor(config_.inputWidth, config_.inputHeight);
    CpuTimer timer;

    while (running_) {
        FrameData task;

        if (!rawQueue_.pop(task)) {
            break;
        }     
        // 前处理 
        timer.reset();
        task.prep = preprocessor.process(task.originalImage);
        task.cost.preprocess_ms = timer.elapsedMs();
        task.cost.total_ms += task.cost.preprocess_ms;   

        if (task.prep.blob.empty()) {
            continue;
        }
        if (!preprocessQueue_.push(std::move(task))) {
            break;
        }

    }

    std::cout << "[PREPROCESS] exit" << std::endl;
    logger.info("[PREPROCESS] exit");
    preprocessQueue_.stop();
}

void InspectionPipeline::gpuInferLoop() {
    auto& logger = utils::FileLogger::instance();
    TensorRTInfer infer(config_.enginePath);

    if (!infer.load()) {
        std::cerr << "[GPU] failed to load TensorRT engine" << std::endl;
        logger.error("[GPU] failed to load TensorRT engine");
        running_ = false;

        rawQueue_.stop();
        inferQueue_.stop();
        sendQueue_.stop();

        Tcpserver_.stop();
        return;
    }
    
    logger.info("[GPU] TensorRT engine loaded");
    CudaPreprocessor cudaPreprocessor(infer.inputWidth(), infer.inputHeight());
    CudaEventTimer preprocessTimer(infer.stream());
    CudaEventTimer inferTimer(infer.stream());

    if (!preprocessTimer.isValid() || !inferTimer.isValid()) {
        logger.error("[GPU] failed to create CUDA event timer");
        running_ = false;
        rawQueue_.stop();
        inferQueue_.stop();
        return;
    }

    while (running_) {
        FrameData task;

        if (!rawQueue_.pop(task)) {
            break;
        }
        ///////////////////////// test
        #if DEBUG_PREPROCESS_COMPARE 
            if(task.frameId <= 10){
                Preprocessor cpuPreprocessor(640, 640);
                compareCpuCudaPreprocess(
                task.originalImage,
                cpuPreprocessor,
                cudaPreprocessor,
                infer
            );
            }
        #endif
        ///////////////////////// test
        PreprocessResult prep;
        if (!preprocessTimer.start()) {
            logger.error("[GPU] failed to start CUDA preprocess timer");
            continue;
        }

        const bool preprocessOk = cudaPreprocessor.process(
            task.originalImage,
            infer.inputDeviceBuffer(),
            infer.inputBufferBytes(),
            infer.inputElementSize(),
            prep,
            infer.stream()
        );

        const bool preprocessTimerStopped = preprocessTimer.stop();
        if (!preprocessTimerStopped || !preprocessOk) {
            if (preprocessTimerStopped) {
                preprocessTimer.elapsedMs();
            }
            std::cerr << "[GPU] cuda preprocess failed, frameId="<< task.frameId<< std::endl;
            logger.error("[GPU] cuda preprocess failed, frameId=" + std::to_string(task.frameId));
            continue;
        }

        if (!inferTimer.start()) {
            preprocessTimer.elapsedMs();
            logger.error("[GPU] failed to start CUDA inference timer");
            continue;
        }

        const bool inferOk = infer.inferFromDevice(task.modelMask);
        const bool inferTimerOk = inferTimer.stop();

        task.cost.preprocess_ms = preprocessTimer.elapsedMs();
        task.cost.infer_ms = inferTimerOk ? inferTimer.elapsedMs() : -1.0;
        task.cost.total_ms += task.cost.preprocess_ms;
        task.cost.total_ms += task.cost.infer_ms;

        if (!inferOk || task.modelMask.empty() || task.cost.infer_ms < 0.0) {
            std::cerr << "[GPU] TensorRT output empty, frameId="
                      << task.frameId
                      << std::endl;
            logger.error("[GPU] TensorRT output empty, frameId=" + std::to_string(task.frameId));
            continue;
        }
        
        task.prep = std::move(prep);
        
        if (!inferQueue_.push(std::move(task))) {
            break;
        }

    }

    std::cout << "[GPU] gpuInferLoop exit" << std::endl;
    logger.info("[GPU] gpuInferLoop exit");
    inferQueue_.stop();
}

void InspectionPipeline::postprocessLoop() {
    auto& logger = utils::FileLogger::instance();
    YoloSegPostprocessor postprocessor(config_.confThreshold, config_.nmsThreshold, config_.maskThreshold);
    Visualizer visualizer;
    std::unique_ptr<utils::BaselineSaver> baselineSaver;
    if (config_.saveBaseline) {
        // BaselineSaver只负责把FP16结果落盘留档；INT8对比线程不依赖这些文件。
        baselineSaver = std::make_unique<utils::BaselineSaver>(config_.baselineDir);
    }
    CpuTimer timer;

    while (running_) {
        FrameData processed;

        if (!inferQueue_.pop(processed)) {
            break;
        }        
        //后处理
        timer.reset();
        processed.results = postprocessor.process(processed.outputs, processed.prep, config_.classNames);
        processed.cost.postprocess_ms = timer.elapsedMs();
        processed.cost.total_ms += processed.cost.postprocess_ms;

        timer.reset();
        processed.visualizedImage = visualizer.draw(processed.originalImage, processed.results);
        processed.cost.visualize_ms = timer.elapsedMs();
        processed.cost.total_ms += processed.cost.visualize_ms;

        if (baselineSaver && !baselineSaver->save(processed)) {
            logger.warning("[BASELINE] failed to save frameId=" + std::to_string(processed.frameId));
        }

        // INT8对比线程消费的是FP16后处理后的完整帧副本：
        // 这里保留FP16检测结果、耗时和原图，便于INT8线程用同一张图重新推理并逐帧比较。
        if (config_.enableInt8Compare) {
            if (!int8CompareQueue_.push(processed)) {
                logger.warning("[INT8 Compare] failed to enqueue frameId=" + std::to_string(processed.frameId));
            }
        }

        std::ostringstream costLog;
        costLog << "[TIME] frameId=" << processed.frameId
                << ", preprocess=" << processed.cost.preprocess_ms << " ms"
                << ", infer=" << processed.cost.infer_ms << " ms"
                << ", postprocess=" << processed.cost.postprocess_ms << " ms"
                << ", visualize=" << processed.cost.visualize_ms << " ms"
                << ", total=" << processed.cost.total_ms << " ms"
                << ", detections=" << processed.results.size();
        logger.info(costLog.str());

        // std::cout << "[TIME] " << "ID" << ": " << processed.frameId << std::endl
        //           << "[TIME] "<< "preprocess" << ": " << processed.cost.preprocess_ms << " ms" << std::endl
        //           << "[TIME] "<< "infer" << ": " << processed.cost.infer_ms << " ms" << std::endl
        //           << "[TIME] "<< "postprocess" << ": " << processed.cost.postprocess_ms<< " ms" << std::endl
        //           << "[TIME] "<< "visualize" << ": " << processed.cost.visualize_ms << " ms" << std::endl
        //           << "[TIME] "<< "total" << ": " << processed.cost.total_ms << " ms" << std::endl;
        // if(processed.frameId == 10) running_ = false;

        if (!sendQueue_.push(std::move(processed))) {
            break;
        }

    }

    std::cout << "[POSTPROCESS] exit" << std::endl;
    logger.info("[POSTPROCESS] exit");
    int8CompareQueue_.stop();
    sendQueue_.stop();
    running_ = false;
    Tcpserver_.stop();
}

void InspectionPipeline::int8CompareLoop() {
    auto& logger = utils::FileLogger::instance();

    TensorRTInfer int8Infer(config_.int8EnginePath);
    if (!int8Infer.load()) {
        logger.error("[INT8 Compare] failed to load TensorRT INT8 engine");
        int8CompareQueue_.stop();
        return;
    }

    logger.info("[INT8 Compare] TensorRT INT8 engine loaded");

    CudaPreprocessor cudaPreprocessor(int8Infer.inputWidth(), int8Infer.inputHeight());
    Preprocessor cpuPreprocessor(int8Infer.inputWidth(), int8Infer.inputHeight());
    CudaEventTimer preprocessTimer(int8Infer.stream());
    CudaEventTimer inferTimer(int8Infer.stream());
    CpuTimer cpuTimer;

    if (!preprocessTimer.isValid() || !inferTimer.isValid()) {
        logger.error("[INT8 Compare] failed to create CUDA event timer");
        return;
    }

    SegPostprocessor postprocessor(config_.confThreshold, config_.nmsThreshold, config_.maskThreshold);

    utils::Int8CompareConfig compareConfig;
    compareConfig.saveDir = config_.compareDir;
    compareConfig.iouThreshold = config_.compareIouThreshold;
    compareConfig.minMatchRate = config_.compareMinMatchRate;
    compareConfig.maxMeanScoreDiff = config_.compareMaxMeanScoreDiff;
    utils::Int8CompareSaver compareSaver(compareConfig);

    while (true) {
        FrameData fp16Frame;
        if (!int8CompareQueue_.pop(fp16Frame)) {
            break;
        }

        FrameData int8Frame;
        int8Frame.frameId = fp16Frame.frameId;
        int8Frame.source_path = fp16Frame.source_path;
        int8Frame.originalImage = fp16Frame.originalImage;

        // 第一步：预处理。主流程使用CUDA预处理时，INT8对比线程也走同一条CUDA路径。
        if (config_.useCudaPreprocess) {
            if (!preprocessTimer.start()) {
                logger.error("[INT8 Compare] failed to start CUDA preprocess timer");
                continue;
            }

            bool ok = cudaPreprocessor.process(
                int8Frame.originalImage,
                int8Infer.inputDeviceBuffer(),
                int8Infer.inputBufferBytes(),
                int8Infer.inputElementSize(),
                int8Frame.prep,
                int8Infer.stream()
            );

            const bool preprocessTimerStopped = preprocessTimer.stop();
            if (!preprocessTimerStopped || !ok) {
                if (preprocessTimerStopped) {
                    preprocessTimer.elapsedMs();
                }
                logger.error("[INT8 Compare] cuda preprocess failed, frameId=" + std::to_string(fp16Frame.frameId));
                continue;
            }
        } else {
            cpuTimer.reset();
            int8Frame.prep = cpuPreprocessor.process(int8Frame.originalImage);
            int8Frame.cost.preprocess_ms = cpuTimer.elapsedMs();
            int8Frame.cost.total_ms += int8Frame.cost.preprocess_ms;

            if (int8Frame.prep.blob.empty()) {
                logger.error("[INT8 Compare] cpu preprocess failed, frameId=" + std::to_string(fp16Frame.frameId));
                continue;
            }

        }

        // 第二步：无论输入来自 CPU 还是 CUDA，GPU 推理都使用 CUDA Event 计时。
        if (!inferTimer.start()) {
            if (config_.useCudaPreprocess) {
                preprocessTimer.elapsedMs();
            }
            logger.error("[INT8 Compare] failed to start CUDA inference timer");
            continue;
        }

        const bool inferOk = config_.useCudaPreprocess
            ? int8Infer.inferFromDevice(int8Frame.modelMask)
            : int8Infer.infer(int8Frame.prep.blob, int8Frame.modelMask);

        const bool inferTimerOk = inferTimer.stop();
        if (config_.useCudaPreprocess) {
            int8Frame.cost.preprocess_ms = preprocessTimer.elapsedMs();
            int8Frame.cost.total_ms += int8Frame.cost.preprocess_ms;
        }

        int8Frame.cost.infer_ms = inferTimerOk ? inferTimer.elapsedMs() : -1.0;
        int8Frame.cost.total_ms += int8Frame.cost.infer_ms;

        if (!inferOk || int8Frame.modelMask.empty() || int8Frame.cost.infer_ms < 0.0) {
            logger.error("[INT8 Compare] TensorRT output empty, frameId=" + std::to_string(fp16Frame.frameId));
            continue;
        }

        // 第三步：后处理。阈值和类别名与FP16 baseline完全一致，只比较engine差异。
        cpuTimer.reset();
        int8Frame.results = postprocessor.process(int8Frame.outputs, int8Frame.prep, config_.classNames);
        int8Frame.cost.postprocess_ms = cpuTimer.elapsedMs();
        int8Frame.cost.total_ms += int8Frame.cost.postprocess_ms;

        // 第四步：保存逐帧对比指标和最终summary。
        if (!compareSaver.save(fp16Frame, int8Frame)) {
            logger.warning("[INT8 Compare] failed to save frameId=" + std::to_string(fp16Frame.frameId));
        }
    }

    std::cout << "[INT8 Compare] exit" << std::endl;
    logger.info("[INT8 Compare] exit");
}


std::string InspectionPipeline::escapeJson(const std::string& text) const {
    std::ostringstream oss;

    for (char ch : text) {
        switch (ch) {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }

    return oss.str();
}

bool InspectionPipeline::compareCpuCudaPreprocess(
    const cv::Mat& image,
    Preprocessor& cpuPreprocessor,
    CudaPreprocessor& cudaPreprocessor,
    TensorRTInfer& infer
) {
    auto saveBlobAsImage = [](
        const std::string& savePath,
        const float* blob,
        int width,
        int height
    ) -> bool {
        if (!blob) {
            std::cerr << "[COMPARE] blob is null" << std::endl;
            return false;
        }

        cv::Mat out(height, width, CV_8UC3);
        int area = width * height;

        for (int y = 0; y < height; ++y) {
            cv::Vec3b* rowPtr = out.ptr<cv::Vec3b>(y);

            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;

                float r = blob[0 * area + idx];
                float g = blob[1 * area + idx];
                float b = blob[2 * area + idx];

                r = std::max(0.0f, std::min(1.0f, r));
                g = std::max(0.0f, std::min(1.0f, g));
                b = std::max(0.0f, std::min(1.0f, b));

                rowPtr[x] = cv::Vec3b(
                    static_cast<uchar>(std::round(b * 255.0f)),
                    static_cast<uchar>(std::round(g * 255.0f)),
                    static_cast<uchar>(std::round(r * 255.0f))
                );
            }
        }

        if (!cv::imwrite(savePath, out)) {
            std::cerr << "[COMPARE] failed to save image: "
                      << savePath
                      << std::endl;
            return false;
        }

        std::cout << "[COMPARE] saved image: "
                  << savePath
                  << std::endl;

        return true;
    };

    // 1. CPU前处理
    PreprocessResult cpuPrep = cpuPreprocessor.process(image);

    if (cpuPrep.blob.empty()) {
        std::cerr << "[COMPARE] cpu preprocess failed" << std::endl;
        return false;
    }

    // 2. CUDA前处理
    PreprocessResult cudaPrep;

    bool ok = cudaPreprocessor.process(
        image,
        infer.inputDeviceBuffer(),
        infer.inputBufferBytes(),
        infer.inputElementSize(),
        cudaPrep,
        infer.stream()
    );

    if (!ok) {
        std::cerr << "[COMPARE] cuda preprocess failed" << std::endl;
        return false;
    }

    // 3. CUDA结果拷回CPU
    const int inputC = 3;
    const int inputH = 640;
    const int inputW = 640;
    const int count = inputC * inputH * inputW;

    std::vector<float> cudaBlob(count);
    cudaError_t err = cudaSuccess;

    if (infer.inputElementSize() == sizeof(float)) {
        err = cudaMemcpyAsync(
            cudaBlob.data(),
            infer.inputDeviceBuffer(),
            count * sizeof(float),
            cudaMemcpyDeviceToHost,
            infer.stream()
        );
    } else if (infer.inputElementSize() == sizeof(__half)) {
        // Modified: FP16 input buffer must be copied as half and converted back to float before diff.
        std::vector<__half> halfBlob(count);
        err = cudaMemcpyAsync(
            halfBlob.data(),
            infer.inputDeviceBuffer(),
            count * sizeof(__half),
            cudaMemcpyDeviceToHost,
            infer.stream()
        );

        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(infer.stream());
            if (err != cudaSuccess) {
                std::cerr << "[COMPARE] cudaStreamSynchronize failed: "
                          << cudaGetErrorString(err)
                          << std::endl;
                return false;
            }

            for (int i = 0; i < count; ++i) {
                cudaBlob[i] = __half2float(halfBlob[i]);
            }
        }
    } else {
        std::cerr << "[COMPARE] unsupported input element size: "
                  << infer.inputElementSize()
                  << std::endl;
        return false;
    }

    if (err != cudaSuccess) {
        std::cerr << "[COMPARE] cudaMemcpyAsync D2H failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }

    err = cudaStreamSynchronize(infer.stream());

    if (err != cudaSuccess) {
        std::cerr << "[COMPARE] cudaStreamSynchronize failed: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return false;
    }

    // 4. 比较scale/pad
    std::cout << "[COMPARE] cpu scale=" << cpuPrep.scale
              << ", pad=(" << cpuPrep.pad_x << "," << cpuPrep.pad_y << ")"
              << std::endl;

    std::cout << "[COMPARE] cuda scale=" << cudaPrep.scale
              << ", pad=(" << cudaPrep.pad_x << "," << cudaPrep.pad_y << ")"
              << std::endl;

    // 5. 比较blob数据
    const float* cpuData = cpuPrep.blob.ptr<float>();

    double maxDiff = 0.0;
    double meanDiff = 0.0;

    for (int i = 0; i < count; ++i) {
        double diff = std::abs(
            static_cast<double>(cpuData[i]) -
            static_cast<double>(cudaBlob[i])
        );

        maxDiff = std::max(maxDiff, diff);
        meanDiff += diff;
    }

    meanDiff /= static_cast<double>(count);

    std::cout << "[COMPARE] preprocess diff: "
              << "maxDiff=" << maxDiff
              << ", meanDiff=" << meanDiff
              << std::endl;

    // 6. 保存原图和CUDA前处理结果图像
    cv::imwrite("test/original_debug.jpg", image);

    saveBlobAsImage(
        "test/cuda_preprocess_debug.jpg",
        cudaBlob.data(),
        inputW,
        inputH
    );

    std::cout << "[COMPARE] saved image: original_debug.jpg" << std::endl;
    std::cout << "[COMPARE] saved image: cuda_preprocess_debug.jpg" << std::endl;

    return true;
}
} // namespace blade::pipeline
