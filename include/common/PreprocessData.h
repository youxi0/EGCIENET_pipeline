#pragma once

#include <opencv2/core.hpp>

#include <array>

struct PreprocessConfig {
    int inputWidth = 352;
    int inputHeight = 352;

    // BGR 顺序，数值基于原始 0-255 像素尺度。
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};
};

struct PreprocessResult {
    int originalWidth = 0;
    int originalHeight = 0;
    int inputWidth = 0;
    int inputHeight = 0;

    float scaleX = 1.0f;
    float scaleY = 1.0f;

    // CPU 预处理路径会在这里填入 1x3xHxW 的 FP32 NCHW 张量。
    // CUDA 预处理路径直接写 GPU 输入 buffer，因此这里只填元信息。
    cv::Mat blob;
};
