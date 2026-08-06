#pragma once

#include "common/FrameData.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

class ImageSource;

namespace egcinet::pipeline {

struct PipelineConfig {
    std::string enginePath;
    std::size_t queueSize = 3;

    // 所有 slot 在启动时按该上限预分配原图和后处理显存，运行时不再动态扩容。
    int maxSourceWidth = 1920;
    int maxSourceHeight = 1080;

    // BGR 顺序，数值基于原始 0-255 像素尺度。
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};
    float maskThreshold = 0.6f;

    // 回调在完成线程中同步执行，只能在调用期间读取 frame。
    // 不要在回调内调用 stop；如需异步持有结果，应 clone 对应 cv::Mat。
    std::function<void(const FrameData&)> resultCallback;
};

// 三线程流水线：采集/H2D 线程 -> GPU Scheduler -> D2H/结果完成线程。
// 每个 slot 持有独立的原图和后处理显存，TensorRT 输入/输出由 GPU Scheduler 共享。
class InspectionPipeline {
public:
    InspectionPipeline(std::unique_ptr<ImageSource> source, PipelineConfig config);
    ~InspectionPipeline();

    InspectionPipeline(const InspectionPipeline&) = delete;
    InspectionPipeline& operator=(const InspectionPipeline&) = delete;
    InspectionPipeline(InspectionPipeline&&) = delete;
    InspectionPipeline& operator=(InspectionPipeline&&) = delete;

    bool start();
    void stop() noexcept;

    bool isRunning() const noexcept;
    std::uint64_t processedFrames() const noexcept;
    std::string lastError() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace egcinet::pipeline
