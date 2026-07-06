#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <algorithm>

struct PreprocessResult {
    cv::Mat blob;
    int original_width = 0;
    int original_height = 0;
    float scale = 1.0f;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct PreprocessConfig {
    int inputWidth = 352;
    int inputHeight = 352;

    // Values are in OpenCV BGR channel order and raw 0-255 pixel scale.
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};
};

class Preprocessor {
public:
    // Creates a default BGR resize-normalize preprocessor for the requested model size.
    explicit Preprocessor(int inputWidth = 352, int inputHeight = 352)
        : config_{
              inputWidth,
              inputHeight,
              {140.505f, 157.845f, 135.66f},
              {61.455f, 60.18f, 62.22f}
          } {}

    // Creates a preprocessor with explicit size and BGR mean/std normalization.
    explicit Preprocessor(PreprocessConfig config)
        : config_(config) {}

    // Converts an OpenCV image to a contiguous FP32 blob with shape 1x3xHxW.
    PreprocessResult process(const cv::Mat& image) const {
        PreprocessResult result;
        if (image.empty() || config_.inputWidth <= 0 || config_.inputHeight <= 0) {
            return result;
        }

        cv::Mat bgr;
        if (image.channels() == 3) {
            bgr = image;
        } else if (image.channels() == 1) {
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        } else {
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

                for (int c = 0; c < 3; ++c) {
                    const float denom = std::max(config_.std[c], 1.0e-12f);
                    dst[c * area + index] =
                        (static_cast<float>(pixel[c]) - config_.mean[c]) / denom;
                }
            }
        }

        result.original_width = image.cols;
        result.original_height = image.rows;
        result.scale_x = static_cast<float>(config_.inputWidth) / static_cast<float>(image.cols);
        result.scale_y = static_cast<float>(config_.inputHeight) / static_cast<float>(image.rows);
        result.scale = std::min(result.scale_x, result.scale_y);
        return result;
    }

    // Exposes immutable preprocessing parameters for diagnostics and tests.
    const PreprocessConfig& config() const noexcept {
        return config_;
    }

private:
    PreprocessConfig config_;
};
