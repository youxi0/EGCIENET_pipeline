#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

struct VisualizerConfig {
    // OpenCV 图像采用 BGR 顺序，默认用红色显示分割区域。
    std::array<std::uint8_t, 3> maskColor{0, 0, 255};
    float alpha = 0.5f;
};

// 在 GPU 上把概率图叠加到 BGR 原图，处理过程直接复用原图设备缓冲区。
class Visualizer {
public:
    explicit Visualizer(VisualizerConfig config = {});

    // 原地生成可视化图，不执行 D2H；mask 必须与原图尺寸一致。
    bool process(
        unsigned char* imageDevice,
        size_t imageStep,
        int imageWidth,
        int imageHeight,
        const float* probabilityDevice,
        const std::uint8_t* binaryDevice,
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
