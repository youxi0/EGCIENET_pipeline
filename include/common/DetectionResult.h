#pragma once

#include <opencv2/opencv.hpp>
#include <string>

//模块:DetectionResult用于保存一个目标的检测框、类别、置信度和实例分割mask
struct DetectionResult
{
    int classId = -1;
    float confidence = 0.0f;
    cv::Rect box;
    cv::Mat mask;
    std::string className;
};