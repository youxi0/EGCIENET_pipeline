#pragma once

#include "acquisition/ImageSource.h"
#include <opencv2/opencv.hpp>
#include <string>

class VideoSource : public ImageSource {
public:
    explicit VideoSource(const std::string& video_path);
    VideoSource(int camera_id, int camera_width, int camera_height, int camera_fps);

    bool open() override;
/*
    read() 函数作用：
    从视频文件或摄像头中读取一帧图像。

    参数：
    FrameData& frame 是引用传参。
    也就是说，这个函数会把读取到的数据写入 frame 里面。
*/
    bool read(FrameData& frame) override;

    void reset() override;
/*
    release() 函数作用：
    释放视频文件或摄像头资源。
    程序结束前应该调用它。
*/
    void release() override;

private:
    bool use_camera_ = false;
    int camera_id_ = 0;
    int camera_width_ = 1920;
    int camera_height_ = 1080;
    int camera_fps_ = 30;
    std::string video_path_;

    cv::VideoCapture cap_;
    size_t current_index_ = 0;
};
