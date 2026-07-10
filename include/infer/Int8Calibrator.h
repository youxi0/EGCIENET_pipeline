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

    // BGR 顺序，与 cv::imread 和导出模型约定保持一致。
    std::array<float, 3> mean{140.505f, 157.845f, 135.66f};
    std::array<float, 3> std{61.455f, 60.18f, 62.22f};

    // 0 表示使用排序后的全部可支持图片。
    size_t maxImages = 0;
    bool readCache = true;
};

class Int8Calibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit Int8Calibrator(Int8CalibratorConfig config);
    ~Int8Calibrator() override;

    Int8Calibrator(const Int8Calibrator&) = delete;
    Int8Calibrator& operator=(const Int8Calibrator&) = delete;

    // 返回 TensorRT 校准阶段使用的固定 batch size。
    int getBatchSize() const noexcept override;

    // 熵校准期间向 TensorRT 提供一个 GPU 输入 batch。
    bool getBatch(
        void* bindings[],
        const char* names[],
        int nbBindings
    ) noexcept override;

    // 如果已有校准 cache，允许 TensorRT 复用。
    const void* readCalibrationCache(size_t& length) noexcept override;

    // 首次校准后保存 TensorRT 生成的校准表。
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    // 表示构造阶段是否成功收集图片或找到可复用 cache。
    bool isValid() const noexcept;

    // 返回构造或校准回调期间捕获的最近错误。
    const std::string& lastError() const noexcept;

    // 返回从 imageDirectory 中选中的校准图片数量。
    size_t imageCount() const noexcept;

private:
    // 递归收集可支持图片并排序，保证校准可复现。
    bool collectImages();

    // 分配 host staging batch 和 TensorRT device 输入 buffer。
    bool allocateDeviceBuffer();

    // 加载并预处理图片，直到组成一个固定大小的校准 batch。
    bool prepareBatch(size_t& validImageCount);

    // 读取单张图片，预处理成 BGR NCHW FP32，并拷贝到 hostBatch_。
    bool copyImageToBatch(const std::string& imagePath, size_t batchIndex);

    // 检查图片扩展名是否为 OpenCV 支持的格式。
    bool isSupportedImage(const std::string& path) const;

    // 查找 inputTensorName 对应的 TensorRT binding。
    int findInputBinding(const char* names[], int nbBindings) const;

    // 检查磁盘上是否存在非空且可读的校准 cache。
    bool hasReadableCache() const;

    // 保存并记录最近一次校准错误。
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
