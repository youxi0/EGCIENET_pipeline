#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "common/DetectionResult.h"

//模块:Visualizer负责把检测框、类别文字和实例mask画回原图
class Visualizer{
public:
    cv::Mat draw(const cv::Mat& image, const std::vector<DetectionResult>& results);
};