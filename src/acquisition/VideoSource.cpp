#include "acquisition/VideoSource.h"

#include <iostream>
#include <sstream>

namespace {

// Jetson CSI 摄像头输出 RAW Bayer，必须经过 Argus/ISP 去马赛克后再交给 OpenCV。
std::string buildJetsonCameraPipeline(
    int cameraId,
    int width,
    int height,
    int fps
) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << cameraId
             << " ! video/x-raw(memory:NVMM),width=" << width
             << ",height=" << height
             << ",framerate=" << fps << "/1,format=NV12"
             << " ! nvvidconv"
             << " ! video/x-raw,format=BGRx"
             << " ! videoconvert"
             << " ! video/x-raw,format=BGR"
             << " ! appsink drop=true max-buffers=1 sync=false";
    return pipeline.str();
}

} // namespace

VideoSource::VideoSource(const std::string& video_path)
    : use_camera_(false),
      video_path_(video_path) {}

VideoSource::VideoSource(
    int camera_id,
    int camera_width,
    int camera_height,
    int camera_fps
)
    : use_camera_(true),
      camera_id_(camera_id),
      camera_width_(camera_width),
      camera_height_(camera_height),
      camera_fps_(camera_fps) {}

bool VideoSource::open() {
    /*
        每次打开输入源时，把帧编号重新置 0。

        这样做的好处是：
        每次重新打开一个视频，第一帧编号都从 0 开始。
    */
    current_index_ = 0;

    if (use_camera_) {
        const std::string cameraPipeline = buildJetsonCameraPipeline(
            camera_id_, camera_width_, camera_height_, camera_fps_);
        cap_.open(cameraPipeline, cv::CAP_GSTREAMER);
        if (!cap_.isOpened()) {
            std::cerr << "Failed to open Jetson Argus camera pipeline: "
                      << cameraPipeline << std::endl;
            return false;
        }
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
