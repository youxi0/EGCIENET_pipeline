#pragma once

#include "common/PreprocessData.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <string>

struct FrameCost {
    double acquire_ms = 0.0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double visualize_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

struct FrameData {
    uint64_t frameId = 0;
    double timestamp_ms = 0.0;
    std::string source_path;

    // 采集到的 BGR 原图；GPU 链路用其尺寸恢复 mask，并生成可视化结果。
    cv::Mat originalImage;

    // CPU/CUDA 预处理和后处理共享的 resize/归一化元信息。
    PreprocessResult prep;

    // GPU 后处理完成后统一下载的原图尺寸结果。
    cv::Mat probabilityMask;
    cv::Mat binaryMask;
    cv::Mat visualizedImage;

    FrameCost cost;

    void releaseTransient() {
        prep.blob.release();
    }
};

inline double getCurrentTimestampMs() {
    using namespace std::chrono;

    const auto now = steady_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();

    return static_cast<double>(ms);
}
