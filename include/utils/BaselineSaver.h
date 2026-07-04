#pragma once

#include "common/FrameData.h"

#include <fstream>
#include <string>

namespace utils {

// FP16 baseline结果保存器: 保存可视化图、检测txt和耗时csv。
class BaselineSaver {
public:
    explicit BaselineSaver(const std::string& saveDir);
    ~BaselineSaver();

    bool save(const FrameData& frame);

private:
    std::string getFileStem(const std::string& path) const;

    bool saveVisualizedImage(const FrameData& frame, const std::string& imageName);
    bool saveDetectionTxt(const FrameData& frame, const std::string& imageName);
    bool saveTimeCsv(const FrameData& frame, const std::string& imageName);

private:
    std::string saveDir_;
    std::string visDir_;
    std::string txtDir_;
    std::string timeDir_;
    std::string timeCsvPath_;
    std::ofstream timeFile_;
};

} // namespace utils
