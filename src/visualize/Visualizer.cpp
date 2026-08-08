#include "visualize/Visualizer.h"

#include <iostream>
#include <limits>

void launchVisualizeKernel(
    unsigned char* image,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    const std::uint8_t* classMask,
    VisualizerConfig config,
    cudaStream_t stream
);

namespace {

// 统一检查可视化阶段的 CUDA 调用。
bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "[Visualizer] " << operation << " failed: "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

} // namespace

Visualizer::Visualizer(VisualizerConfig config)
    : config_(config) {}

bool Visualizer::process(
    unsigned char* imageDevice,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    const std::uint8_t* classMaskDevice,
    cudaStream_t stream
) const {
    if (imageDevice == nullptr || classMaskDevice == nullptr) {
        std::cerr << "[Visualizer] input device buffer is null" << std::endl;
        return false;
    }

    if (imageWidth <= 0 || imageHeight <= 0 ||
        imageWidth > std::numeric_limits<int>::max() / 3 ||
        imageStep < static_cast<size_t>(imageWidth) * 3) {
        std::cerr << "[Visualizer] invalid image shape or step" << std::endl;
        return false;
    }

    if (config_.alpha < 0.0f || config_.alpha > 1.0f) {
        std::cerr << "[Visualizer] alpha must be in [0, 1]" << std::endl;
        return false;
    }

    launchVisualizeKernel(
        imageDevice,
        imageStep,
        imageWidth,
        imageHeight,
        classMaskDevice,
        config_,
        stream
    );

    return checkCuda(cudaGetLastError(), "kernel launch");
}

bool Visualizer::enqueueDownload(
    const unsigned char* imageDevice,
    size_t imageStep,
    int imageWidth,
    int imageHeight,
    cv::Mat& visualizedImage,
    cudaStream_t stream
) const {
    if (imageDevice == nullptr || imageWidth <= 0 || imageHeight <= 0 ||
        imageWidth > std::numeric_limits<int>::max() / 3 ||
        imageStep < static_cast<size_t>(imageWidth) * 3) {
        std::cerr << "[Visualizer] invalid GPU image for download" << std::endl;
        return false;
    }

    visualizedImage.create(imageHeight, imageWidth, CV_8UC3);
    const size_t rowBytes = static_cast<size_t>(imageWidth) * 3;

    if (!checkCuda(
            cudaMemcpy2DAsync(
                visualizedImage.data,
                visualizedImage.step,
                imageDevice,
                imageStep,
                rowBytes,
                static_cast<size_t>(imageHeight),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "visualized image D2H")) {
        visualizedImage.release();
        return false;
    }

    return true;
}
