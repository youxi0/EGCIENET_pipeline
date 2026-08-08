#pragma once

#include "common/SegmentationClasses.h"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct BgrColor {
    std::uint8_t b;
    std::uint8_t g;
    std::uint8_t r;
};

struct VisualizerConfig {
    // 类别 0 保留原图；1~4 依次使用红、黄、洋红、青色叠加。
    BgrColor classColors[egcinet::segmentation::kClassCount]{
        {0, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
        {255, 255, 0}
    };
    float alpha = 0.55f;
};

static_assert(std::is_trivially_copyable_v<VisualizerConfig>,
              "VisualizerConfig must be safe to pass to a CUDA kernel by value");

// 在 GPU 上按类别配色叠加到 BGR 原图，处理过程直接复用原图设备缓冲区。
class Visualizer {
public:
    explicit Visualizer(VisualizerConfig config = {});

    // 原地生成可视化图，不执行 D2H；classMask 每个像素必须是 0~4。
    bool process(
        unsigned char* imageDevice,
        size_t imageStep,
        int imageWidth,
        int imageHeight,
        const std::uint8_t* classMaskDevice,
        cudaStream_t stream
    ) const;

    // 在同一 stream 上排队下载可视化结果；该函数不主动同步。
    bool enqueueDownload(
        const unsigned char* imageDevice,
        size_t imageStep,
        int imageWidth,
        int imageHeight,
        cv::Mat& visualizedImage,
        cudaStream_t stream
    ) const;

private:
    VisualizerConfig config_;
};
