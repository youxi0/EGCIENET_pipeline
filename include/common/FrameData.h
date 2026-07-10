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
    double total_ms = 0.0;
};

struct FrameData {
    uint64_t frameId = 0;
    double timestamp_ms = 0.0;
    std::string source_path;

    // 采集到的 BGR 原图；保留一份用于后处理恢复 mask 到原图尺寸。
    cv::Mat originalImage;

    // CPU/CUDA 预处理和后处理共享的 resize/归一化元信息。
    PreprocessResult prep;

    // TensorRT 输出的模型尺度 FP32 概率图，形状为 352x352。
    cv::Mat modelMask;

    // 后处理结果，用于部署输出或调试保存。
    cv::Mat probabilityMask;
    cv::Mat binaryMask;

    FrameCost cost;

    void releaseTransient() {
        prep.blob.release();
        modelMask.release();
    }
};

inline double getCurrentTimestampMs() {
    using namespace std::chrono;

    const auto now = steady_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();

    return static_cast<double>(ms);
}
