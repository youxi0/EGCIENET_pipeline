#include "acquisition/VideoSource.h"

#include <iostream>

VideoSource::VideoSource(const std::string& video_path)
    : use_camera_(false),
      video_path_(video_path) {}

VideoSource::VideoSource(int camera_id)
    : use_camera_(true),
      camera_id_(camera_id) {}

bool VideoSource::open() {
    /*
        每次打开输入源时，把帧编号重新置 0。

        这样做的好处是：
        每次重新打开一个视频，第一帧编号都从 0 开始。
    */
    current_index_ = 0;

    if (use_camera_) {
        cap_.open(camera_id_);
    } else {
        cap_.open(video_path_);
    }

    if (!cap_.isOpened()) {
        if (use_camera_) {
            std::cerr << "Failed to open camera: " << camera_id_ << std::endl;
        } else {
            std::cerr << "Failed to open video: " << video_path_ << std::endl;
        }
        return false;
    }

    double width = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap_.get(cv::CAP_PROP_FPS);

    std::cout << "Video source opened. "
              << "width=" << width
              << ", height=" << height
              << ", fps=" << fps
              << std::endl;

    return true;
}

bool VideoSource::read(FrameData& frame) {
    // cv::Mat 是 OpenCV 中最常用的图像容器。图像矩阵
    cv::Mat image;

    if (!cap_.read(image)) {
        return false;
    }

    if (image.empty()) {
        return false;
    }

    // frame.frame_id = frame_id_++;
    // frame.timestamp_ms = getCurrentTimestampMs();
    frame.frameId = current_index_++;
    frame.source_path = use_camera_ ? "camera" : video_path_;
    frame.originalImage = image;

    return true;
}

void VideoSource::reset(){
    current_index_ = 0;
}

void VideoSource::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }

    current_index_ = 0;
}