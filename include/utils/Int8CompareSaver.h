#pragma once

#include "common/FrameData.h"

#include <fstream>
#include <string>

namespace utils {

// INT8对比配置：控制输出目录和“精度是否可接受”的判定阈值。
struct Int8CompareConfig {
    std::string saveDir = "results/compare_fp16_int8";
    float iouThreshold = 0.5f;
    float minMatchRate = 0.95f;
    float maxMeanScoreDiff = 0.10f;
};

// 单帧对比指标：同时记录检测差异和速度差异。
struct Int8CompareMetrics {
    int fp16Count = 0;
    int int8Count = 0;
    int matchedCount = 0;
    int missedCount = 0;
    int extraCount = 0;
    double matchRate = 0.0;
    double meanIou = 0.0;
    double meanScoreAbsDiff = 0.0;
    double inferSpeedup = 0.0;
    double totalSpeedup = 0.0;
    bool pass = false;
};

// FP16 baseline vs INT8结果保存器：
// 1. 每帧写一行compare.csv，方便直接用表格查看速度和精度。
// 2. 析构时写summary.txt，给一轮验证提供平均指标。
class Int8CompareSaver {
public:
    explicit Int8CompareSaver(const Int8CompareConfig& config);
    ~Int8CompareSaver();

    bool save(const FrameData& fp16Frame, const FrameData& int8Frame);

private:
    std::string getFileStem(const std::string& path) const;
    double boxIou(const cv::Rect& a, const cv::Rect& b) const;
    Int8CompareMetrics compare(const FrameData& fp16Frame, const FrameData& int8Frame) const;
    bool saveFrameCsv(
        const FrameData& fp16Frame,
        const FrameData& int8Frame,
        const std::string& imageName,
        const Int8CompareMetrics& metrics
    );
    void updateSummary(const Int8CompareMetrics& metrics);
    void writeSummary();

private:
    Int8CompareConfig config_;
    std::string compareCsvPath_;
    std::string summaryPath_;
    std::ofstream compareFile_;

    size_t frameCount_ = 0;
    size_t passCount_ = 0;
    int totalFp16Count_ = 0;
    int totalInt8Count_ = 0;
    int totalMatchedCount_ = 0;
    int totalMissedCount_ = 0;
    int totalExtraCount_ = 0;
    double totalMatchRate_ = 0.0;
    double totalMeanIou_ = 0.0;
    double totalMeanScoreAbsDiff_ = 0.0;
    double totalInferSpeedup_ = 0.0;
    double totalTotalSpeedup_ = 0.0;
};

} // namespace utils
