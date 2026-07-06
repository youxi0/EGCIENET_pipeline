#pragma once

#include "common/DetectionResult.h"
#include "preprocess/Preprocessor.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <vector>


struct FrameCost {
    double acquire_ms = 0.0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double visualize_ms = 0.0;
    double send_ms = 0.0;
    double total_ms = 0.0;
};

struct FrameData {
    int frameId = 0;
    std::string source_path;

    cv::Mat originalImage;
    // cv::Mat inputBlob;
    cv::Mat visualizedImage;

    PreprocessResult prep;
    std::vector<cv::Mat> outputs;
    std::vector<DetectionResult> results;

    FrameCost cost;
};


inline double getCurrentTimestampMs() {
    using namespace std::chrono;

    auto now = steady_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();

    return static_cast<double>(ms);
}
