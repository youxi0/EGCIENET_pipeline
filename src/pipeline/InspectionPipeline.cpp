#include "pipeline/InspectionPipeline.h"

#include "acquisition/ImageSource.h"
#include "common/BlockingQueue.h"
#include "common/PreprocessData.h"
#include "common/Timer.h"
#include "infer/TensorRTInfer.h"
#include "postprocess/SegPostprocessor.h"
#include "preprocess/CudaPreprocessor.h"
#include "utils/FileLogger.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace egcinet::pipeline {

namespace {

bool checkCuda(cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) {
        return true;
    }
    error = std::string(operation) + " failed: " + cudaGetErrorString(status);
    return false;
}

// 计算紧凑 BGR 图像的行跨度和总字节数，并检查尺寸乘法是否溢出。
bool bgrBufferBytes(
    int width,
    int height,
    size_t& packedRowBytes,
    size_t& imageBytes,
    std::string& error
) {
    packedRowBytes = 0;
    imageBytes = 0;
    error.clear();
    if (width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() / 3) {
        error = "BGR image dimensions must be positive";
        return false;
    }

    packedRowBytes = static_cast<size_t>(width) * 3;
    if (static_cast<size_t>(height) >
            std::numeric_limits<size_t>::max() / packedRowBytes) {
        error = "BGR image size overflows size_t";
        return false;
    }

    imageBytes = packedRowBytes * static_cast<size_t>(height);
    return true;
}

// 检查 OpenCV 图像格式，并返回紧凑布局所需字节数。
bool getBgrImageLayout(
    const cv::Mat& image,
    size_t& packedRowBytes,
    size_t& imageBytes,
    std::string& error
) {
    if (image.empty() || image.type() != CV_8UC3 ||
        !bgrBufferBytes(image.cols, image.rows, packedRowBytes, imageBytes, error)) {
        if (error.empty()) {
            error = "image must be a non-empty CV_8UC3 BGR image";
        }
        return false;
    }
    if (image.step < packedRowBytes) {
        error = "BGR image step is smaller than packed row bytes";
        return false;
    }
    return true;
}

// 在获取 stream 上排队 BGR 原图 H2D；本函数不申请显存也不主动同步。
bool enqueueBgrImageUpload(
    const cv::Mat& image,
    size_t packedRowBytes,
    size_t imageBytes,
    unsigned char* destinationDevice,
    size_t destinationBytes,
    cudaStream_t stream,
    std::string& error
) {
    if (image.data == nullptr || packedRowBytes == 0 || imageBytes == 0) {
        error = "BGR image upload received an invalid prepared image layout";
        return false;
    }
    if (destinationDevice == nullptr || stream == nullptr) {
        error = "BGR image upload destination or stream is null";
        return false;
    }
    if (imageBytes > destinationBytes) {
        error = "BGR image exceeds slot device buffer capacity";
        return false;
    }

    const cudaError_t status = cudaMemcpy2DAsync(
        destinationDevice,
        packedRowBytes,
        image.data,
        image.step,
        packedRowBytes,
        static_cast<size_t>(image.rows),
        cudaMemcpyHostToDevice,
        stream
    );
    if (status != cudaSuccess) {
        error = std::string("cudaMemcpy2DAsync H2D failed: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

// 概率图和二值图连续存放在槽位独立的后处理显存中，统一检查大小计算是否溢出。
bool maskBufferBytes(
    int width,
    int height,
    size_t& probabilityBytes,
    size_t& binaryBytes,
    size_t& totalBytes
) noexcept {
    probabilityBytes = 0;
    binaryBytes = 0;
    totalBytes = 0;
    if (width <= 0 || height <= 0) {
        return false;
    }

    const size_t widthValue = static_cast<size_t>(width);
    const size_t heightValue = static_cast<size_t>(height);
    if (heightValue > std::numeric_limits<size_t>::max() / widthValue) {
        return false;
    }

    const size_t pixels = widthValue * heightValue;
    if (pixels > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    probabilityBytes = pixels * sizeof(float);
    binaryBytes = pixels * sizeof(std::uint8_t);
    if (probabilityBytes > std::numeric_limits<size_t>::max() - binaryBytes) {
        return false;
    }
    totalBytes = probabilityBytes + binaryBytes;
    return true;
}

} // 匿名命名空间

class InspectionPipeline::Impl {
public:
    Impl(std::unique_ptr<ImageSource> source, PipelineConfig config)
        : source_(std::move(source)),
          config_(std::move(config)),
          freeSlots_(queueStorageSize(config_.queueSize)),
          acquiredQueue_(queueStorageSize(config_.queueSize)),
          completedQueue_(queueStorageSize(config_.queueSize)) {
        if (config_.queueSize == 0) {
            config_.queueSize = 1;
        }
    }

    ~Impl() {
        stop();
    }

    bool start() {
        std::lock_guard<std::mutex> lifecycleLock(lifecycleMutex_);
        if (running_.load(std::memory_order_acquire)) {
            setError("pipeline is already running");
            return false;
        }

        joinWorkers();
        releaseResources();
        clearError();
        stopRequested_.store(false, std::memory_order_release);
        processedFrames_.store(0, std::memory_order_release);
        resetQueues();
        resetAcquisitionStartup();

        bool initialized = false;
        try {
            initialized = validateConfig() && initializeResources();
        } catch (const std::exception& exception) {
            setError(std::string("pipeline initialization exception: ") + exception.what());
        } catch (...) {
            setError("pipeline initialization caught an unknown exception");
        }
        if (!initialized) {
            releaseResources();
            return false;
        }

        running_.store(true, std::memory_order_release);
        try {
            completionThread_ = std::thread(&Impl::completionLoop, this);
            gpuSchedulerThread_ = std::thread(&Impl::gpuSchedulerLoop, this);
            acquireThread_ = std::thread(&Impl::acquireLoop, this);
        } catch (const std::exception& exception) {
            fail(std::string("failed to create pipeline worker: ") + exception.what());
            joinWorkers();
            running_.store(false, std::memory_order_release);
            releaseResources();
            return false;
        }

        {
            std::unique_lock<std::mutex> startupLock(acquisitionStartupMutex_);
            acquisitionStartupCv_.wait(startupLock, [this]() {
                return acquisitionStartupReady_;
            });
            if (!acquisitionStartupSucceeded_) {
                startupLock.unlock();
                requestStop();
                joinWorkers();
                running_.store(false, std::memory_order_release);
                releaseResources();
                return false;
            }
        }
        return true;
    }

    void stop() noexcept {
        std::lock_guard<std::mutex> lifecycleLock(lifecycleMutex_);
        requestStop();
        joinWorkers();
        running_.store(false, std::memory_order_release);
        releaseResources();
    }

    bool isRunning() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    std::uint64_t processedFrames() const noexcept {
        return processedFrames_.load(std::memory_order_acquire);
    }

    std::string lastError() const {
        std::lock_guard<std::mutex> errorLock(errorMutex_);
        return lastError_;
    }

private:
    struct FrameSlot {
        FrameSlot(
            cudaStream_t acquisitionStream,
            cudaStream_t gpuStream,
            cudaStream_t completionStream
        )
            : h2dTimer(acquisitionStream),
              preprocessTimer(gpuStream),
              inferTimer(gpuStream),
              postprocessTimer(gpuStream),
              d2hTimer(completionStream) {
        }

        ~FrameSlot() {
            if (h2dDone != nullptr) {
                cudaEventDestroy(h2dDone);
            }
            if (gpuDone != nullptr) {
                cudaEventDestroy(gpuDone);
            }
            if (postprocessDevice != nullptr) {
                cudaFree(postprocessDevice);
            }
            if (preprocessDevice != nullptr) {
                cudaFree(preprocessDevice);
            }
        }

        bool initialize(
            size_t preprocessCapacity,
            size_t postprocessCapacity,
            std::string& error
        ) {
            if (!h2dTimer.isValid() || !preprocessTimer.isValid() || !inferTimer.isValid() ||
                !postprocessTimer.isValid() || !d2hTimer.isValid()) {
                error = "failed to create frame CUDA timers";
                return false;
            }
            if (!checkCuda(
                    cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming),
                    "cudaEventCreate h2dDone",
                    error) ||
                !checkCuda(
                    cudaEventCreateWithFlags(&gpuDone, cudaEventDisableTiming),
                    "cudaEventCreate gpuDone",
                    error)) {
                return false;
            }

            preprocessBytes = preprocessCapacity;
            postprocessBytes = postprocessCapacity;
            if (!checkCuda(
                    cudaMalloc(&preprocessDevice, preprocessBytes),
                    "cudaMalloc slot preprocess",
                    error) ||
                !checkCuda(
                    cudaMalloc(&postprocessDevice, postprocessBytes),
                    "cudaMalloc slot postprocess",
                    error)) {
                return false;
            }
            return true;
        }

        float* probabilityDevice() noexcept {
            return static_cast<float*>(postprocessDevice);
        }

        std::uint8_t* binaryDevice() noexcept {
            return static_cast<std::uint8_t*>(postprocessDevice) + probabilityBytes;
        }

        FrameData frame;
        std::chrono::steady_clock::time_point totalStart;
        void* preprocessDevice = nullptr;
        void* postprocessDevice = nullptr;
        size_t preprocessBytes = 0;
        size_t postprocessBytes = 0;
        size_t probabilityBytes = 0;
        size_t binaryBytes = 0;
        size_t imageRowBytes = 0;
        size_t imageBytes = 0;
        cudaEvent_t h2dDone = nullptr;
        cudaEvent_t gpuDone = nullptr;
        CudaEventTimer h2dTimer;
        CudaEventTimer preprocessTimer;
        CudaEventTimer inferTimer;
        CudaEventTimer postprocessTimer;
        CudaEventTimer d2hTimer;
    };

    // 队列先于配置校验构造，因此限制实际分配量，避免异常参数造成巨额分配。
    static std::size_t queueStorageSize(std::size_t size) noexcept {
        if (size == 0) {
            return 1;
        }
        return size > 3 ? 3 : size;
    }

    bool validateConfig() {
        if (source_ == nullptr) {
            setError("image source is null");
            return false;
        }
        if (config_.enginePath.empty()) {
            setError("engine path is empty");
            return false;
        }
        if (config_.queueSize > 3) {
            setError("queue size must not exceed 16");
            return false;
        }
        if (config_.maxSourceWidth <= 0 || config_.maxSourceHeight <= 0) {
            setError("maximum source width and height must be positive");
            return false;
        }
        if (config_.maskThreshold < 0.0f || config_.maskThreshold > 1.0f) {
            setError("mask threshold must be in [0, 1]");
            return false;
        }
        for (float value : config_.std) {
            if (value <= std::numeric_limits<float>::epsilon()) {
                setError("all std values must be greater than zero");
                return false;
            }
        }
        return true;
    }

    // 启动时按最大源尺寸一次性申请所有显存，运行阶段不执行 cudaMalloc/cudaFree。
    bool initializeResources() {
        infer_ = std::make_unique<TensorRTInfer>(config_.enginePath);
        if (!infer_->load()) {
            setError("failed to load TensorRT engine: " + config_.enginePath);
            return false;
        }

        std::string cudaError;
        if (!checkCuda(
                cudaStreamCreateWithFlags(&acquisitionStream_, cudaStreamNonBlocking),
                "cudaStreamCreate acquisition",
                cudaError) ||
            !checkCuda(
                cudaStreamCreateWithFlags(&gpuStream_, cudaStreamNonBlocking),
                "cudaStreamCreate GPU scheduler",
                cudaError) ||
            !checkCuda(
                cudaStreamCreateWithFlags(&completionStream_, cudaStreamNonBlocking),
                "cudaStreamCreate completion",
                cudaError)) {
            setError(cudaError);
            return false;
        }

        PreprocessConfig preprocessConfig;
        preprocessConfig.inputWidth = infer_->inputWidth();
        preprocessConfig.inputHeight = infer_->inputHeight();
        preprocessConfig.mean = config_.mean;
        preprocessConfig.std = config_.std;
        preprocessor_ = std::make_unique<CudaPreprocessor>(preprocessConfig);
        postprocessor_ = std::make_unique<SegPostprocessor>(
            SegPostprocessConfig{config_.maskThreshold});

        size_t maxImageRowBytes = 0;
        size_t maxPreprocessBytes = 0;
        size_t maxProbabilityBytes = 0;
        size_t maxBinaryBytes = 0;
        size_t maxPostprocessBytes = 0;
        if (!bgrBufferBytes(
                config_.maxSourceWidth,
                config_.maxSourceHeight,
                maxImageRowBytes,
                maxPreprocessBytes,
                cudaError) ||
            !maskBufferBytes(
                config_.maxSourceWidth,
                config_.maxSourceHeight,
                maxProbabilityBytes,
                maxBinaryBytes,
                maxPostprocessBytes)) {
            setError(cudaError.empty()
                ? "maximum source dimensions overflow buffer size"
                : cudaError);
            return false;
        }

        inputBytes_ = infer_->inputBufferBytes();
        outputBytes_ = infer_->outputBufferBytes();
        if (!checkCuda(
                cudaMalloc(&inputDevice_, inputBytes_),
                "cudaMalloc shared TensorRT input",
                cudaError) ||
            !checkCuda(
                cudaMalloc(&outputDevice_, outputBytes_),
                "cudaMalloc shared TensorRT output",
                cudaError)) {
            setError(cudaError);
            return false;
        }

        slots_.reserve(config_.queueSize);
        for (std::size_t index = 0; index < config_.queueSize; ++index) {
            auto slot = std::make_unique<FrameSlot>(
                acquisitionStream_, gpuStream_, completionStream_);
            if (!slot->initialize(
                    maxPreprocessBytes,
                    maxPostprocessBytes,
                    cudaError)) {
                setError(cudaError);
                return false;
            }
            slots_.push_back(std::move(slot));
        }

        const size_t slotBytes =
            maxPreprocessBytes + maxPostprocessBytes;
        const size_t explicitCudaBytes =
            slotBytes * slots_.size() + inputBytes_ + outputBytes_;
        std::ostringstream resourceLog;
        resourceLog << "[Pipeline] GPU resources initialized: slots=" << slots_.size()
                    << ", max_source=" << config_.maxSourceWidth << "x"
                    << config_.maxSourceHeight
                    << ", slot_preprocess_bytes=" << maxPreprocessBytes
                    << ", slot_postprocess_bytes=" << maxPostprocessBytes
                    << ", shared_input_bytes=" << inputBytes_
                    << ", shared_output_bytes=" << outputBytes_
                    << ", explicit_cuda_bytes=" << explicitCudaBytes;
        utils::FileLogger::instance().info(resourceLog.str());

        // 初始化阶段不再读取首帧，所有槽位统一交给获取线程。
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            if (!freeSlots_.tryPush(index)) {
                setError("failed to initialize free slot queue");
                return false;
            }
        }
        return true;
    }

    void resetQueues() {
        freeSlots_.reset();
        acquiredQueue_.reset();
        completedQueue_.reset();
    }

    void requestStop() noexcept {
        stopRequested_.store(true, std::memory_order_release);
        freeSlots_.stop();
        acquiredQueue_.stop();
        completedQueue_.stop();
    }

    void fail(const std::string& error) noexcept {
        setError(error);
        requestStop();
    }

    void setError(const std::string& error) const noexcept {
        try {
            bool stored = false;
            {
                std::lock_guard<std::mutex> errorLock(errorMutex_);
                if (lastError_.empty()) {
                    lastError_ = error;
                    stored = true;
                }
            }
            if (stored) {
                utils::FileLogger::instance().error("[Pipeline] " + error);
            }
        } catch (...) {
        }
    }

    void clearError() {
        std::lock_guard<std::mutex> errorLock(errorMutex_);
        lastError_.clear();
    }

    void resetAcquisitionStartup() noexcept {
        std::lock_guard<std::mutex> startupLock(acquisitionStartupMutex_);
        acquisitionStartupReady_ = false;
        acquisitionStartupSucceeded_ = false;
    }

    // 只通知 source 打开结果，不在该锁内执行任何 source 操作。
    void publishAcquisitionStartup(bool succeeded) noexcept {
        {
            std::lock_guard<std::mutex> startupLock(acquisitionStartupMutex_);
            if (acquisitionStartupReady_) {
                return;
            }
            acquisitionStartupSucceeded_ = succeeded;
            acquisitionStartupReady_ = true;
        }
        acquisitionStartupCv_.notify_all();
    }

    void joinWorkers() noexcept {
        if (acquireThread_.joinable()) {
            acquireThread_.join();
        }
        if (gpuSchedulerThread_.joinable()) {
            gpuSchedulerThread_.join();
        }
        if (completionThread_.joinable()) {
            completionThread_.join();
        }
    }

    void releaseResources() noexcept {
        if (acquisitionStream_ != nullptr) {
            cudaStreamSynchronize(acquisitionStream_);
        }
        if (gpuStream_ != nullptr) {
            cudaStreamSynchronize(gpuStream_);
        }
        if (completionStream_ != nullptr) {
            cudaStreamSynchronize(completionStream_);
        }

        slots_.clear();
        postprocessor_.reset();
        preprocessor_.reset();

        if (outputDevice_ != nullptr) {
            cudaFree(outputDevice_);
            outputDevice_ = nullptr;
            outputBytes_ = 0;
        }
        if (inputDevice_ != nullptr) {
            cudaFree(inputDevice_);
            inputDevice_ = nullptr;
            inputBytes_ = 0;
        }

        if (completionStream_ != nullptr) {
            cudaStreamDestroy(completionStream_);
            completionStream_ = nullptr;
        }
        if (gpuStream_ != nullptr) {
            cudaStreamDestroy(gpuStream_);
            gpuStream_ = nullptr;
        }
        if (acquisitionStream_ != nullptr) {
            cudaStreamDestroy(acquisitionStream_);
            acquisitionStream_ = nullptr;
        }
        infer_.reset();
    }

    // 检查动态源尺寸是否处于启动时预分配的上限内，并设置当前帧的 mask 布局。
    bool prepareFrameBuffers(FrameSlot& slot, std::string& error) {
        const int frameWidth = slot.frame.originalImage.cols;
        const int frameHeight = slot.frame.originalImage.rows;
        if (frameWidth > config_.maxSourceWidth ||
            frameHeight > config_.maxSourceHeight) {
            std::ostringstream message;
            message << "frame size " << frameWidth << "x" << frameHeight
                    << " exceeds configured maximum "
                    << config_.maxSourceWidth << "x" << config_.maxSourceHeight;
            error = message.str();
            return false;
        }

        size_t imageRowBytes = 0;
        size_t imageBytes = 0;
        size_t probabilityBytes = 0;
        size_t binaryBytes = 0;
        size_t postprocessBytes = 0;
        if (!getBgrImageLayout(
                slot.frame.originalImage, imageRowBytes, imageBytes, error) ||
            imageRowBytes == 0 ||
            !maskBufferBytes(
                slot.frame.originalImage.cols,
                slot.frame.originalImage.rows,
                probabilityBytes,
                binaryBytes,
                postprocessBytes)) {
            if (error.empty()) {
                error = "invalid frame type or dimensions";
            }
            return false;
        }

        if (imageBytes > slot.preprocessBytes ||
            postprocessBytes > slot.postprocessBytes) {
            error = "frame buffer requirement exceeds preallocated slot capacity";
            return false;
        }

        slot.probabilityBytes = probabilityBytes;
        slot.binaryBytes = binaryBytes;
        slot.imageRowBytes = imageRowBytes;
        slot.imageBytes = imageBytes;
        return true;
    }

    // 获取线程把 CPU BGR 图像直接上传到当前槽位，并用 event 交给 GPU Scheduler。
    bool enqueueFrameUpload(FrameSlot& slot, std::string& error) {
        if (!prepareFrameBuffers(slot, error)) {
            return false;
        }
        if (!slot.h2dTimer.start()) {
            error = "failed to start H2D CUDA timer";
            return false;
        }
        if (!enqueueBgrImageUpload(
                slot.frame.originalImage,
                slot.imageRowBytes,
                slot.imageBytes,
                static_cast<unsigned char*>(slot.preprocessDevice),
                slot.preprocessBytes,
                acquisitionStream_,
                error)) {
            return false;
        }
        if (!slot.h2dTimer.stop()) {
            error = "failed to stop H2D CUDA timer";
            return false;
        }
        return checkCuda(
            cudaEventRecord(slot.h2dDone, acquisitionStream_),
            "cudaEventRecord h2dDone",
            error
        );
    }

    // 获取线程负责读取 CPU 图像、排队 H2D，并把槽位编号交给 GPU Scheduler。
    void acquireLoop() noexcept {
        bool sourceOpened = false;
        try {
            if (!source_->open()) {
                publishAcquisitionStartup(false);
                fail("failed to open image source");
            } else {
                sourceOpened = true;
                publishAcquisitionStartup(true);
                utils::FileLogger::instance().info(
                    "[Pipeline] image source opened by acquisition worker");
            }

            while (!stopRequested_.load(std::memory_order_acquire)) {
                std::size_t slotIndex = 0;
                if (!freeSlots_.pop(slotIndex)) {
                    break;
                }

                FrameSlot& slot = *slots_[slotIndex];
                slot.frame = FrameData{};
                slot.totalStart = std::chrono::steady_clock::now();
                CpuTimer acquireTimer;
                const bool readOk =
                    !stopRequested_.load(std::memory_order_acquire) &&
                    source_->read(slot.frame);
                slot.frame.cost.acquire_ms = acquireTimer.elapsedMs();

                if (!readOk) {
                    break;
                }
                slot.frame.timestamp_ms = getCurrentTimestampMs();
                std::string cudaError;
                if (!enqueueFrameUpload(slot, cudaError)) {
                    fail(cudaError + " for frame " + std::to_string(slot.frame.frameId));
                    break;
                }
                if (!acquiredQueue_.push(slotIndex)) {
                    break;
                }
            }
        } catch (const std::exception& exception) {
            publishAcquisitionStartup(false);
            fail(std::string("acquisition worker exception: ") + exception.what());
        } catch (...) {
            publishAcquisitionStartup(false);
            fail("acquisition worker caught an unknown exception");
        }

        if (sourceOpened) {
            source_->release();
        }
        acquiredQueue_.stop();
    }

    // 单个 Scheduler 线程在同一 stream 上依次提交预处理、推理和后处理。
    void gpuSchedulerLoop() noexcept {
        try {
            std::size_t slotIndex = 0;
            while (acquiredQueue_.pop(slotIndex)) {
                if (stopRequested_.load(std::memory_order_acquire)) {
                    break;
                }

                FrameSlot& slot = *slots_[slotIndex];
                std::string cudaError;

                if (!checkCuda(
                        cudaStreamWaitEvent(gpuStream_, slot.h2dDone, 0),
                        "cudaStreamWaitEvent h2dDone",
                        cudaError)) {
                    fail(cudaError);
                    break;
                }

                if (!slot.preprocessTimer.start()) {
                    fail("failed to start preprocess CUDA timer");
                    break;
                }
                const bool preprocessOk = preprocessor_->process(
                    static_cast<const unsigned char*>(slot.preprocessDevice),
                    slot.preprocessBytes,
                    slot.frame.originalImage.cols,
                    slot.frame.originalImage.rows,
                    slot.imageRowBytes,
                    inputDevice_,
                    inputBytes_,
                    infer_->inputElementSize(),
                    slot.frame.prep,
                    gpuStream_);
                if (!slot.preprocessTimer.stop() || !preprocessOk) {
                    fail("CUDA preprocessing failed for frame " +
                        std::to_string(slot.frame.frameId));
                    break;
                }

                if (!slot.inferTimer.start()) {
                    fail("failed to start inference CUDA timer");
                    break;
                }
                const bool inferOk = infer_->inferFromDeviceBuffers(
                    inputDevice_,
                    inputBytes_,
                    outputDevice_,
                    outputBytes_,
                    gpuStream_);
                if (!slot.inferTimer.stop() || !inferOk) {
                    fail("TensorRT inference failed for frame " +
                        std::to_string(slot.frame.frameId));
                    break;
                }

                if (!slot.postprocessTimer.start()) {
                    fail("failed to start postprocess CUDA timer");
                    break;
                }
                const bool postprocessOk = postprocessor_->processToBuffers(
                    outputDevice_,
                    outputBytes_,
                    infer_->outputElementSize(),
                    infer_->outputWidth(),
                    infer_->outputHeight(),
                    slot.probabilityDevice(),
                    slot.probabilityBytes,
                    slot.binaryDevice(),
                    slot.binaryBytes,
                    slot.frame.originalImage.cols,
                    slot.frame.originalImage.rows,
                    gpuStream_);
                if (!slot.postprocessTimer.stop() || !postprocessOk) {
                    fail("CUDA postprocessing failed for frame " +
                        std::to_string(slot.frame.frameId));
                    break;
                }

                if (!checkCuda(
                        cudaEventRecord(slot.gpuDone, gpuStream_),
                        "cudaEventRecord gpuDone",
                        cudaError)) {
                    fail(cudaError);
                    break;
                }
                if (!completedQueue_.push(slotIndex)) {
                    break;
                }
            }
        } catch (const std::exception& exception) {
            fail(std::string("GPU scheduler exception: ") + exception.what());
        } catch (...) {
            fail("GPU scheduler caught an unknown exception");
        }
        completedQueue_.stop();
    }

    // 完成线程等待 GPU Scheduler，执行 D2H、计时汇总、回调并回收槽位。
    void completionLoop() noexcept {
        try {
            std::size_t slotIndex = 0;
            while (completedQueue_.pop(slotIndex)) {
                if (stopRequested_.load(std::memory_order_acquire)) {
                    break;
                }

                FrameSlot& slot = *slots_[slotIndex];
                std::string cudaError;
                if (!checkCuda(
                        cudaStreamWaitEvent(completionStream_, slot.gpuDone, 0),
                        "cudaStreamWaitEvent gpuDone",
                        cudaError)) {
                    fail(cudaError);
                    break;
                }
                if (!slot.d2hTimer.start()) {
                    fail("failed to start D2H CUDA timer");
                    break;
                }
                const bool downloadOk = postprocessor_->enqueueDownloadFromBuffers(
                    slot.probabilityDevice(),
                    slot.probabilityBytes,
                    slot.binaryDevice(),
                    slot.binaryBytes,
                    slot.frame.originalImage.cols,
                    slot.frame.originalImage.rows,
                    slot.frame.probabilityMask,
                    slot.frame.binaryMask,
                    completionStream_);
                if (!slot.d2hTimer.stop() || !downloadOk) {
                    fail("mask D2H failed for frame " +
                        std::to_string(slot.frame.frameId));
                    break;
                }
                if (!checkCuda(
                        cudaStreamSynchronize(completionStream_),
                        "cudaStreamSynchronize completion",
                        cudaError)) {
                    fail(cudaError);
                    break;
                }

                // 所有 GPU 工作完成后，在完成线程统一读取各阶段 CUDA Event 耗时。
                slot.frame.cost.h2d_ms = slot.h2dTimer.elapsedMs();
                slot.frame.cost.preprocess_ms = slot.preprocessTimer.elapsedMs();
                slot.frame.cost.infer_ms = slot.inferTimer.elapsedMs();
                slot.frame.cost.postprocess_ms = slot.postprocessTimer.elapsedMs();
                slot.frame.cost.d2h_ms = slot.d2hTimer.elapsedMs();
                if (slot.frame.cost.h2d_ms < 0.0 ||
                    slot.frame.cost.preprocess_ms < 0.0 ||
                    slot.frame.cost.infer_ms < 0.0 ||
                    slot.frame.cost.postprocess_ms < 0.0 ||
                    slot.frame.cost.d2h_ms < 0.0) {
                    fail("failed to read CUDA timing for frame " +
                        std::to_string(slot.frame.frameId));
                    break;
                }
                slot.frame.cost.total_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - slot.totalStart).count();

                if (config_.resultCallback) {
                    try {
                        config_.resultCallback(slot.frame);
                    } catch (const std::exception& exception) {
                        fail(std::string("result callback failed: ") + exception.what());
                        break;
                    } catch (...) {
                        fail("result callback failed with an unknown exception");
                        break;
                    }
                }

                processedFrames_.fetch_add(1, std::memory_order_release);
                slot.frame.releaseTransient();
                slot.frame = FrameData{};
                slot.probabilityBytes = 0;
                slot.binaryBytes = 0;
                slot.imageRowBytes = 0;
                slot.imageBytes = 0;
                if (!freeSlots_.push(slotIndex)) {
                    break;
                }
            }
        } catch (const std::exception& exception) {
            fail(std::string("completion worker exception: ") + exception.what());
        } catch (...) {
            fail("completion worker caught an unknown exception");
        }

        running_.store(false, std::memory_order_release);
        freeSlots_.stop();
    }

private:
    std::unique_ptr<ImageSource> source_;
    PipelineConfig config_;

    BlockingQueue<std::size_t> freeSlots_;
    BlockingQueue<std::size_t> acquiredQueue_;
    BlockingQueue<std::size_t> completedQueue_;

    std::unique_ptr<TensorRTInfer> infer_;
    std::unique_ptr<CudaPreprocessor> preprocessor_;
    std::unique_ptr<SegPostprocessor> postprocessor_;
    std::vector<std::unique_ptr<FrameSlot>> slots_;

    // GPU Scheduler 串行提交任务，因此所有帧安全复用同一组 TensorRT 输入输出显存。
    void* inputDevice_ = nullptr;
    void* outputDevice_ = nullptr;
    size_t inputBytes_ = 0;
    size_t outputBytes_ = 0;

    cudaStream_t acquisitionStream_ = nullptr;
    cudaStream_t gpuStream_ = nullptr;
    cudaStream_t completionStream_ = nullptr;

    std::thread acquireThread_;
    std::thread gpuSchedulerThread_;
    std::thread completionThread_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};
    std::atomic<std::uint64_t> processedFrames_{0};

    mutable std::mutex lifecycleMutex_;
    mutable std::mutex acquisitionStartupMutex_;
    mutable std::mutex errorMutex_;
    std::condition_variable acquisitionStartupCv_;
    mutable std::string lastError_;
    bool acquisitionStartupReady_ = false;
    bool acquisitionStartupSucceeded_ = false;
};

InspectionPipeline::InspectionPipeline(
    std::unique_ptr<ImageSource> source,
    PipelineConfig config
)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(config))) {
}

InspectionPipeline::~InspectionPipeline() = default;

bool InspectionPipeline::start() {
    return impl_->start();
}

void InspectionPipeline::stop() noexcept {
    impl_->stop();
}

bool InspectionPipeline::isRunning() const noexcept {
    return impl_->isRunning();
}

std::uint64_t InspectionPipeline::processedFrames() const noexcept {
    return impl_->processedFrames();
}

std::string InspectionPipeline::lastError() const {
    return impl_->lastError();
}

} // namespace egcinet::pipeline
