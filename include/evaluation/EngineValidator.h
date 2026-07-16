#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 描述一个待验证的 TensorRT engine。
struct EngineValidationTarget {
    std::string name;
    std::string enginePath;
};

struct ValidationConfig {
    std::string imageDirectory;
    std::string labelDirectory;
    std::string outputCsv = "results/validation/summary.csv";

    // BGR 顺序，数值基于原始 0-255 像素尺度。
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};
    float maskThreshold = 0.5f;

    // maxImages 为 0 时使用验证集中的全部图像。
    std::size_t maxImages = 0;
    std::size_t warmupIterations = 20;
    std::size_t benchmarkIterations = 200;
    bool includeVisualization = true;
};

struct TimingDistribution {
    double meanMs = 0.0;
    double p50Ms = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
};

struct AccuracySummary {
    std::size_t imageCount = 0;
    std::uint64_t pixelCount = 0;
    double globalDice = 0.0;
    double meanDice = 0.0;
    double globalIou = 0.0;
    double meanIou = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
};

struct PerformanceSummary {
    TimingDistribution preprocess;
    TimingDistribution inference;
    TimingDistribution postprocess;
    TimingDistribution visualization;
    TimingDistribution d2h;
    TimingDistribution endToEnd;
    double fps = 0.0;
};

struct EngineValidationResult {
    std::string name;
    std::string enginePath;
    AccuracySummary accuracy;
    PerformanceSummary performance;
};

// 顺序加载多个 engine，统计分割精度和完整 GPU 推理链路耗时。
// 顺序加载可以避免在 Jetson 上同时驻留多份 engine 权重。
class EngineValidator {
public:
    explicit EngineValidator(ValidationConfig config);

    // 验证成功后把每个 engine 的结果写入 results，并输出 CSV。
    bool run(
        const std::vector<EngineValidationTarget>& targets,
        std::vector<EngineValidationResult>& results
    );

    const std::string& lastError() const noexcept;

private:
    ValidationConfig config_;
    std::string lastError_;
};
