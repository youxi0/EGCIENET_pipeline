#include "acquisition/ImageFolderSource.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

ImageFolderSource::ImageFolderSource(const std::string& folder_path)
    : folder_path_(folder_path) {}

bool ImageFolderSource::open() {
    image_paths_.clear();
    current_index_ = 0;


    if (!fs::exists(folder_path_)) {
        std::cerr << "Folder does not exist: " << folder_path_ << std::endl;
        return false;
    }

    for (const auto& entry : fs::directory_iterator(folder_path_)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string path = entry.path().string();

        if (isImageFile(path)) {
            image_paths_.push_back(path);
        }
    }

    std::sort(image_paths_.begin(), image_paths_.end());

    if (image_paths_.empty()) {
        std::cerr << "No image files found in folder: " << folder_path_ << std::endl;
        return false;
    }

    std::cout << "Loaded " << image_paths_.size()
              << " images from folder: " << folder_path_ << std::endl;

    return true;
}

bool ImageFolderSource::read(FrameData& frame) {
    if (current_index_ >= image_paths_.size()) {

        return false;
    }
    

    std::string image_path = image_paths_[current_index_];

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        current_index_++;
        return read(frame);
    }

    frame.frameId = current_index_++;
    // frame.timestamp_ms = getCurrentTimestampMs();
    frame.source_path = image_path;
    frame.originalImage = image;


    return true;
}

void ImageFolderSource::reset() {
    current_index_ = 0;
}

void ImageFolderSource::release() {
    image_paths_.clear();
    current_index_ = 0;
}

bool ImageFolderSource::isImageFile(const std::string& path) const {
    std::string ext = fs::path(path).extension().string();
    // 用于统一转换为小写
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png" ||
           ext == ".bmp" ||
           ext == ".tif" ||
           ext == ".tiff";
}