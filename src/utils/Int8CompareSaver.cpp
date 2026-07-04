#include "utils/Int8CompareSaver.h"
#include "utils/FileLogger.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace utils {

Int8CompareSaver::Int8CompareSaver(const Int8CompareConfig& config)
    : config_(config) {
    std::filesystem::create_directories(config_.saveDir);

    compareCsvPath_ = config_.saveDir + "/compare.csv";
    summaryPath_ = config_.saveDir + "/summary.txt";

    compareFile_.open(compareCsvPath_, std::ios::out);
    if (!compareFile_.is_open()) {
        std::cerr << "[INT8 Compare] failed to open csv: "
                  << compareCsvPath_ << std::endl;
        FileLogger::instance().error("[INT8 Compare] failed to open csv: " + compareCsvPath_);
        return;
    }

    // 第一步：写CSV表头，后续每处理一帧都会追加一行对比结果。
    compareFile_ << "frame_id,image_name,"
                 << "fp16_count,int8_count,matched,missed,extra,"
                 << "match_rate,mean_iou,mean_score_abs_diff,"
                 << "fp16_preprocess_ms,fp16_infer_ms,fp16_postprocess_ms,fp16_compute_ms,"
                 << "int8_preprocess_ms,int8_infer_ms,int8_postprocess_ms,int8_compute_ms,"
                 << "infer_speedup,total_speedup,status\n";

    FileLogger::instance().info("[INT8 Compare] output dir: " + config_.saveDir);
}

Int8CompareSaver::~Int8CompareSaver() {
    if (compareFile_.is_open()) {
        compareFile_.close();
    }

    writeSummary();
}

bool Int8CompareSaver::save(const FrameData& fp16Frame, const FrameData& int8Frame) {
    if (!compareFile_.is_open()) {
        return false;
    }

    const std::string imageName = getFileStem(fp16Frame.source_path);
    const Int8CompareMetrics metrics = compare(fp16Frame, int8Frame);

    if (!saveFrameCsv(fp16Frame, int8Frame, imageName, metrics)) {
        return false;
    }

    updateSummary(metrics);
    return true;
}

std::string Int8CompareSaver::getFileStem(const std::string& path) const {
    std::filesystem::path p(path);
    return p.stem().string();
}

double Int8CompareSaver::boxIou(const cv::Rect& a, const cv::Rect& b) const {
    const cv::Rect interRect = a & b;
    const double interArea = static_cast<double>(interRect.area());
    const double unionArea = static_cast<double>(a.area() + b.area()) - interArea;

    if (unionArea <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    return interArea / unionArea;
}

Int8CompareMetrics Int8CompareSaver::compare(
    const FrameData& fp16Frame,
    const FrameData& int8Frame
) const {
    Int8CompareMetrics metrics;
    metrics.fp16Count = static_cast<int>(fp16Frame.results.size());
    metrics.int8Count = static_cast<int>(int8Frame.results.size());

    std::vector<bool> int8Used(int8Frame.results.size(), false);
    double totalIou = 0.0;
    double totalScoreAbsDiff = 0.0;

    // 第二步：按class_id和box IoU做贪心匹配，统计INT8是否保住FP16 baseline的检测。
    for (const auto& fp16Result : fp16Frame.results) {
        int bestIndex = -1;
        double bestIou = 0.0;

        for (size_t i = 0; i < int8Frame.results.size(); ++i) {
            if (int8Used[i]) {
                continue;
            }

            const auto& int8Result = int8Frame.results[i];
            if (fp16Result.classId != int8Result.classId) {
                continue;
            }

            const double iou = boxIou(fp16Result.box, int8Result.box);
            if (iou > bestIou) {
                bestIou = iou;
                bestIndex = static_cast<int>(i);
            }
        }

        if (bestIndex >= 0 && bestIou >= static_cast<double>(config_.iouThreshold)) {
            int8Used[bestIndex] = true;
            ++metrics.matchedCount;
            totalIou += bestIou;
            totalScoreAbsDiff += std::abs(
                static_cast<double>(fp16Result.confidence) -
                static_cast<double>(int8Frame.results[bestIndex].confidence)
            );
        }
    }

    metrics.missedCount = metrics.fp16Count - metrics.matchedCount;
    metrics.extraCount = metrics.int8Count - metrics.matchedCount;

    if (metrics.fp16Count == 0) {
        metrics.matchRate = metrics.int8Count == 0 ? 1.0 : 0.0;
    } else {
        metrics.matchRate = static_cast<double>(metrics.matchedCount) /
                            static_cast<double>(metrics.fp16Count);
    }

    if (metrics.matchedCount > 0) {
        metrics.meanIou = totalIou / static_cast<double>(metrics.matchedCount);
        metrics.meanScoreAbsDiff = totalScoreAbsDiff / static_cast<double>(metrics.matchedCount);
    }

    // 第三步：计算速度提升倍率，>1表示INT8更快。
    if (int8Frame.cost.infer_ms > std::numeric_limits<double>::epsilon()) {
        metrics.inferSpeedup = fp16Frame.cost.infer_ms / int8Frame.cost.infer_ms;
    }

    const double fp16ComputeMs =
        fp16Frame.cost.preprocess_ms +
        fp16Frame.cost.infer_ms +
        fp16Frame.cost.postprocess_ms;
    const double int8ComputeMs =
        int8Frame.cost.preprocess_ms +
        int8Frame.cost.infer_ms +
        int8Frame.cost.postprocess_ms;

    if (int8ComputeMs > std::numeric_limits<double>::epsilon()) {
        metrics.totalSpeedup = fp16ComputeMs / int8ComputeMs;
    }

    metrics.pass =
        metrics.matchRate >= static_cast<double>(config_.minMatchRate) &&
        metrics.meanScoreAbsDiff <= static_cast<double>(config_.maxMeanScoreDiff);

    return metrics;
}

bool Int8CompareSaver::saveFrameCsv(
    const FrameData& fp16Frame,
    const FrameData& int8Frame,
    const std::string& imageName,
    const Int8CompareMetrics& metrics
) {
    const double fp16ComputeMs =
        fp16Frame.cost.preprocess_ms +
        fp16Frame.cost.infer_ms +
        fp16Frame.cost.postprocess_ms;
    const double int8ComputeMs =
        int8Frame.cost.preprocess_ms +
        int8Frame.cost.infer_ms +
        int8Frame.cost.postprocess_ms;

    compareFile_ << std::fixed << std::setprecision(6)
                 << fp16Frame.frameId << ","
                 << imageName << ","
                 << metrics.fp16Count << ","
                 << metrics.int8Count << ","
                 << metrics.matchedCount << ","
                 << metrics.missedCount << ","
                 << metrics.extraCount << ","
                 << metrics.matchRate << ","
                 << metrics.meanIou << ","
                 << metrics.meanScoreAbsDiff << ","
                 << fp16Frame.cost.preprocess_ms << ","
                 << fp16Frame.cost.infer_ms << ","
                 << fp16Frame.cost.postprocess_ms << ","
                 << fp16ComputeMs << ","
                 << int8Frame.cost.preprocess_ms << ","
                 << int8Frame.cost.infer_ms << ","
                 << int8Frame.cost.postprocess_ms << ","
                 << int8ComputeMs << ","
                 << metrics.inferSpeedup << ","
                 << metrics.totalSpeedup << ","
                 << (metrics.pass ? "PASS" : "REVIEW")
                 << "\n";

    compareFile_.flush();
    return true;
}

void Int8CompareSaver::updateSummary(const Int8CompareMetrics& metrics) {
    ++frameCount_;
    passCount_ += metrics.pass ? 1 : 0;
    totalFp16Count_ += metrics.fp16Count;
    totalInt8Count_ += metrics.int8Count;
    totalMatchedCount_ += metrics.matchedCount;
    totalMissedCount_ += metrics.missedCount;
    totalExtraCount_ += metrics.extraCount;
    totalMatchRate_ += metrics.matchRate;
    totalMeanIou_ += metrics.meanIou;
    totalMeanScoreAbsDiff_ += metrics.meanScoreAbsDiff;
    totalInferSpeedup_ += metrics.inferSpeedup;
    totalTotalSpeedup_ += metrics.totalSpeedup;
}

void Int8CompareSaver::writeSummary() {
    std::ofstream summary(summaryPath_, std::ios::out);
    if (!summary.is_open()) {
        FileLogger::instance().error("[INT8 Compare] failed to open summary: " + summaryPath_);
        return;
    }

    summary << std::fixed << std::setprecision(6);
    summary << "INT8 Compare Summary\n";
    summary << "frames=" << frameCount_ << "\n";
    summary << "iou_threshold=" << config_.iouThreshold << "\n";
    summary << "min_match_rate=" << config_.minMatchRate << "\n";
    summary << "max_mean_score_diff=" << config_.maxMeanScoreDiff << "\n";

    if (frameCount_ == 0) {
        summary << "status=NO_FRAMES\n";
        return;
    }

    const double frameCount = static_cast<double>(frameCount_);
    summary << "pass_frames=" << passCount_ << "\n";
    summary << "pass_rate=" << static_cast<double>(passCount_) / frameCount << "\n";
    summary << "total_fp16_detections=" << totalFp16Count_ << "\n";
    summary << "total_int8_detections=" << totalInt8Count_ << "\n";
    summary << "total_matched=" << totalMatchedCount_ << "\n";
    summary << "total_missed=" << totalMissedCount_ << "\n";
    summary << "total_extra=" << totalExtraCount_ << "\n";
    summary << "avg_match_rate=" << totalMatchRate_ / frameCount << "\n";
    summary << "avg_mean_iou=" << totalMeanIou_ / frameCount << "\n";
    summary << "avg_mean_score_abs_diff=" << totalMeanScoreAbsDiff_ / frameCount << "\n";
    summary << "avg_infer_speedup=" << totalInferSpeedup_ / frameCount << "\n";
    summary << "avg_total_speedup=" << totalTotalSpeedup_ / frameCount << "\n";
}

} // namespace utils
