#pragma once

#include "acquisition/ImageSource.h"
#include "common/BlockingQueue.h"
#include "common/DetectionResult.h"
#include "common/FrameData.h"
#include "infer/TensorRTInfer.h"
#include "network/TcpServer.h"
#include "postprocess/YoloSegPostprocessor.h"
#include "preprocess/Preprocessor.h"
#include "visualize/Visualizer.h"
#include "common/FrameData.h"
#include "preprocess/CudaPreprocessor.h"
#include "utils/BaselineSaver.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace blade::pipeline {
//模块:流水线配置
struct PipelineConfig {
    std::string enginePath;
    std::vector<std::string> classNames;
    uint16_t listenPort = 9000;
    int inputWidth = 640;
    int inputHeight = 640;

    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    float maskThreshold = 0.5f;

    int jpegQuality = 85;
    int queueSize = 4;

    int heartbeatIntervalMs = 2000;
    int heartbeatTimeoutMs = 8000;

    bool useCudaPreprocess = true;

    // 新增配置: FP16 baseline和运行日志的保存目录，便于后续和INT8结果做对比。
    bool saveBaseline = true;
    std::string baselineDir = "results/baseline_fp16";
    std::string logDir = "results/logs";

    // INT8对比开关：开启后额外启动一个线程，用INT8 engine重跑同一帧并和FP16 baseline做对比。
    bool enableInt8Compare = false;
    std::string int8EnginePath;
    std::string compareDir = "results/compare_fp16_int8";
    float compareIouThreshold = 0.5f;
    float compareMinMatchRate = 0.95f;
    float compareMaxMeanScoreDiff = 0.10f;
};

//模块:端侧检测服务主类

class InspectionPipeline {
public:
    InspectionPipeline(std::unique_ptr<ImageSource> source, const PipelineConfig& config);
    ~InspectionPipeline();

    bool start();
    void stop();
    bool isRunning() const;

private:
    void captureLoop();
    void preprocessLoop();
    void gpuInferLoop();
    void postprocessLoop();

    void int8CompareLoop();
    

    bool sendFrame(const FrameData& frame);
    std::string buildMetaJson(const FrameData& frame) const;
    std::string escapeJson(const std::string& text) const;

    bool compareCpuCudaPreprocess(
    const cv::Mat& image,
    Preprocessor& cpuPreprocessor,
    CudaPreprocessor& cudaPreprocessor,
    TensorRTInfer& infer
    );


private:
    std::unique_ptr<ImageSource> source_;
    PipelineConfig config_;

    BlockingQueue<FrameData> sendQueue_;

    BlockingQueue<FrameData> rawQueue_;
    BlockingQueue<FrameData> preprocessQueue_;
    BlockingQueue<FrameData> inferQueue_;
    BlockingQueue<FrameData> int8CompareQueue_;


    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> processingEnabled_{true};
    

    //线程划分:

    std::thread captureThread_;

    std::thread gpuInferThread_;
    std::thread preprocessThread_;

    std::thread postprocessThread_;
    std::thread int8CompareThread_;

    //5.sendLoop:把可视化图像和检测信息通过TCP发送给Qt
    std::thread sendThread_;
};

} // namespace blade::pipeline
