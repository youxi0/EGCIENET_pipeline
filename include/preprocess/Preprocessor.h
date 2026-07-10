#pragma once

#include "common/PreprocessData.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

class Preprocessor {
public:
    explicit Preprocessor(int inputWidth = 352, int inputHeight = 352)
        : config_{inputWidth, inputHeight, {140.505f, 157.845f, 135.66f}, {61.455f, 60.18f, 62.22f}} {}

    explicit Preprocessor(PreprocessConfig config)
        : config_(config) {}

    PreprocessResult process(const cv::Mat& image) const {
        PreprocessResult result;
        if (image.empty() || config_.inputWidth <= 0 || config_.inputHeight <= 0) {
            return result;
        }

        cv::Mat bgr;
        if (!toBgr(image, bgr)) {
            return result;
        }

        cv::Mat resized;
        cv::resize(
            bgr,
            resized,
            cv::Size(config_.inputWidth, config_.inputHeight),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );

        const int shape[4] = {1, 3, config_.inputHeight, config_.inputWidth};
        result.blob.create(4, shape, CV_32F);

        const int area = config_.inputWidth * config_.inputHeight;
        float* dst = result.blob.ptr<float>();

        for (int y = 0; y < config_.inputHeight; ++y) {
            const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < config_.inputWidth; ++x) {
                const int index = y * config_.inputWidth + x;
                const cv::Vec3b& pixel = row[x];

                dst[0 * area + index] =
                    (static_cast<float>(pixel[0]) - config_.mean[0]) / config_.std[0];
                dst[1 * area + index] =
                    (static_cast<float>(pixel[1]) - config_.mean[1]) / config_.std[1];
                dst[2 * area + index] =
                    (static_cast<float>(pixel[2]) - config_.mean[2]) / config_.std[2];
            }
        }

        fillMeta(image, result);
        return result;
    }

    const PreprocessConfig& config() const noexcept {
        return config_;
    }

private:
    bool toBgr(const cv::Mat& image, cv::Mat& bgr) const {
        if (image.channels() == 3 && image.depth() == CV_8U) {
            bgr = image;
            return true;
        }

        if (image.channels() == 1 && image.depth() == CV_8U) {
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
            return true;
        }

        if (image.channels() == 4 && image.depth() == CV_8U) {
            cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
            return true;
        }

        return false;
    }

    void fillMeta(const cv::Mat& image, PreprocessResult& result) const {
        result.originalWidth = image.cols;
        result.originalHeight = image.rows;
        result.inputWidth = config_.inputWidth;
        result.inputHeight = config_.inputHeight;
        result.scaleX = static_cast<float>(config_.inputWidth) / static_cast<float>(image.cols);
        result.scaleY = static_cast<float>(config_.inputHeight) / static_cast<float>(image.rows);
    }

private:
    PreprocessConfig config_;
};
