#pragma once

#include "preprocess/Preprocessor.h"

#include <NvInfer.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct Int8CalibratorConfig {
    std::string imageDirectory;
    std::string cacheFile = "models/egcinet_352_int8.cache";
    std::string inputTensorName = "image";

    int batchSize = 1;
    int inputWidth = 352;
    int inputHeight = 352;

    // BGR order, matching cv::imread and the exported model contract.
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};

    // 0 uses all supported images after deterministic path sorting.
    size_t maxImages = 0;
    bool readCache = true;
};

class Int8Calibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit Int8Calibrator(Int8CalibratorConfig config);
    ~Int8Calibrator() override;

    Int8Calibrator(const Int8Calibrator&) = delete;
    Int8Calibrator& operator=(const Int8Calibrator&) = delete;

    // Returns the fixed calibration batch size requested by TensorRT.
    int getBatchSize() const noexcept override;

    // Provides one GPU input batch to TensorRT during entropy calibration.
    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    // Lets TensorRT reuse a previous calibration cache when available.
    const void* readCalibrationCache(size_t& length) noexcept override;

    // Persists the calibration table produced by TensorRT after a fresh run.
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    // Reports whether construction collected data or found a reusable cache.
    bool isValid() const noexcept;

    // Returns the latest error captured during construction or calibration callbacks.
    const std::string& lastError() const noexcept;

    // Returns the number of calibration images selected from imageDirectory.
    size_t imageCount() const noexcept;

private:
    // Recursively collects supported images and sorts paths for reproducible calibration.
    bool collectImages();

    // Allocates the host staging batch and TensorRT device input buffer.
    bool allocateDeviceBuffer();

    // Loads and preprocesses images until one fixed-size calibration batch is ready.
    bool prepareBatch(size_t& validImageCount);

    // Reads one image, preprocesses it to BGR NCHW FP32, and copies it into hostBatch_.
    bool copyImageToBatch(const std::string& imagePath, size_t batchIndex);

    // Checks image extensions supported by OpenCV image loading.
    bool isSupportedImage(const std::string& path) const;

    // Finds the TensorRT binding that corresponds to inputTensorName.
    int findInputBinding(const char* names[], int nbBindings) const;

    // Checks whether a non-empty calibration cache can be read from disk.
    bool hasReadableCache() const;

    // Stores and logs the latest calibration error.
    void setError(const std::string& message);

private:
    Int8CalibratorConfig config_;
    Preprocessor preprocessor_;

    std::vector<std::string> imagePaths_;
    size_t nextImageIndex_ = 0;
    size_t processedImageCount_ = 0;

    size_t imageElementCount_ = 0;
    size_t batchElementCount_ = 0;
    size_t deviceInputBytes_ = 0;

    std::vector<float> hostBatch_;
    void* deviceInput_ = nullptr;

    std::vector<char> calibrationCache_;
    bool valid_ = false;
    std::string lastError_;
};
