#include "utils/BaselineSaver.h"
#include "utils/FileLogger.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>

namespace utils {

BaselineSaver::BaselineSaver(const std::string& saveDir)
    : saveDir_(saveDir) {
    visDir_ = saveDir_ + "/vis";
    txtDir_ = saveDir_ + "/txt";
    timeDir_ = saveDir_ + "/time";
    timeCsvPath_ = timeDir_ + "/fp16_time.csv";

    std::filesystem::create_directories(visDir_);
    std::filesystem::create_directories(txtDir_);
    std::filesystem::create_directories(timeDir_);

    timeFile_.open(timeCsvPath_, std::ios::out);
    if (!timeFile_.is_open()) {
        std::cerr << "[BaselineSaver] failed to open time csv: "
                  << timeCsvPath_ << std::endl;
        FileLogger::instance().error("[BaselineSaver] failed to open time csv: " + timeCsvPath_);
        return;
    }

    // 新增日志: 记录baseline输出目录，避免多次量化实验时结果目录混淆。
    FileLogger::instance().info("[BaselineSaver] output dir: " + saveDir_);
    timeFile_ << "frame_id,image_name,acquire_ms,preprocess_ms,infer_ms,postprocess_ms,visualize_ms,total_ms\n";
}

BaselineSaver::~BaselineSaver() {
    if (timeFile_.is_open()) {
        timeFile_.close();
    }
}

bool BaselineSaver::save(const FrameData& frame) {
    const std::string imageName = getFileStem(frame.source_path);

    const bool okVis = saveVisualizedImage(frame, imageName);
    const bool okTxt = saveDetectionTxt(frame, imageName);
    const bool okTime = saveTimeCsv(frame, imageName);

    return okVis && okTxt && okTime;
}

std::string BaselineSaver::getFileStem(const std::string& path) const {
    std::filesystem::path p(path);
    return p.stem().string();
}

bool BaselineSaver::saveVisualizedImage(const FrameData& frame, const std::string& imageName) {
    if (frame.visualizedImage.empty()) {
        std::cerr << "[BaselineSaver] visualized image empty, frame id: "
                  << frame.frameId << std::endl;
        return false;
    }

    const std::string savePath = visDir_ + "/" + imageName + ".jpg";
    const bool ok = cv::imwrite(savePath, frame.visualizedImage);

    if (!ok) {
        std::cerr << "[BaselineSaver] failed to save image: "
                  << savePath << std::endl;
        FileLogger::instance().error("[BaselineSaver] failed to save image: " + savePath);
    }

    return ok;
}

bool BaselineSaver::saveDetectionTxt(const FrameData& frame, const std::string& imageName) {
    const std::string savePath = txtDir_ + "/" + imageName + ".txt";

    std::ofstream ofs(savePath, std::ios::out);
    if (!ofs.is_open()) {
        std::cerr << "[BaselineSaver] failed to open txt: "
                  << savePath << std::endl;
        FileLogger::instance().error("[BaselineSaver] failed to open txt: " + savePath);
        return false;
    }

    // 保存格式: class_id class_name score x y w h mask_area
    for (const auto& result : frame.results) {
        ofs << result.classId << " "
            << result.className << " "
            << result.confidence << " "
            << result.box.x << " "
            << result.box.y << " "
            << result.box.width << " "
            << result.box.height << " "
            << cv::countNonZero(result.mask)
            << "\n";
    }

    return true;
}

bool BaselineSaver::saveTimeCsv(const FrameData& frame, const std::string& imageName) {
    if (!timeFile_.is_open()) {
        return false;
    }

    timeFile_ << frame.frameId << ","
              << imageName << ","
              << frame.cost.acquire_ms << ","
              << frame.cost.preprocess_ms << ","
              << frame.cost.infer_ms << ","
              << frame.cost.postprocess_ms << ","
              << frame.cost.visualize_ms << ","
              << frame.cost.total_ms
              << "\n";

    timeFile_.flush();
    return true;
}

} // namespace utils
