#include "acquisition/ImageFolderSource.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <utility>

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
    // 跳过损坏图像时使用循环，避免连续坏图导致递归栈增长。
    while (current_index_ < image_paths_.size()) {
        const std::size_t frameId = current_index_;
        const std::string& imagePath = image_paths_[current_index_++];
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << imagePath << std::endl;
            continue;
        }

        frame.frameId = frameId;
        frame.source_path = imagePath;
        frame.originalImage = std::move(image);
        return true;
    }
    return false;
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
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });

    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png" ||
           ext == ".bmp" ||
           ext == ".tif" ||
           ext == ".tiff";
}
