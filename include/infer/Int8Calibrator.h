#pragma once

#include "preprocess/Preprocessor.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

// INT8校准器配置。校准预处理尺寸必须与构建TensorRT engine时的输入尺寸一致。
struct Int8CalibratorConfig {
    std::string imageDirectory;
    std::string cacheFile = "models/best_int8.cache";
    std::string inputTensorName;

    int batchSize = 1;
    int inputWidth = 640;
    int inputHeight = 640;

    // 0表示使用目录中的全部图片；非0时仅使用排序后的前maxImages张。
    size_t maxImages = 0;

    // cache有效时TensorRT可以跳过重新校准，显著缩短后续构建时间。
    bool readCache = true;
};

// TensorRT EntropyCalibrator2实现，适合CNN检测/分割网络的PTQ校准。
// 该类必须在builder->buildSerializedNetwork()执行期间保持存活。
class Int8Calibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit Int8Calibrator(Int8CalibratorConfig config);
    ~Int8Calibrator() override;

    Int8Calibrator(const Int8Calibrator&) = delete;
    Int8Calibrator& operator=(const Int8Calibrator&) = delete;

    int getBatchSize() const noexcept override;

    // TensorRT反复调用getBatch获取校准输入。bindings中必须放GPU显存地址。
    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    // 如果cache存在则返回其二进制内容，TensorRT可直接复用量化尺度。
    const void* readCalibrationCache(size_t& length) noexcept override;

    // 首次校准完成后，TensorRT通过该回调把cache交给应用保存。
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    bool isValid() const noexcept;
    const std::string& lastError() const noexcept;
    size_t imageCount() const noexcept;

private:
    bool collectImages();
    bool allocateDeviceBuffer();
    bool prepareBatch(size_t& validImageCount);
    bool copyImageToBatch(const std::string& imagePath, size_t batchIndex);
    bool isSupportedImage(const std::string& path) const;
    int findInputBinding(const char* names[], int nbBindings) const;
    bool hasReadableCache() const;
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
