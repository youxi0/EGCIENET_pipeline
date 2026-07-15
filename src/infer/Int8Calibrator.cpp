#include "infer/Int8Calibrator.h"

#include "utils/FileLogger.h"

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

Int8Calibrator::Int8Calibrator(Int8CalibratorConfig config)
    : config_(std::move(config)),
      preprocessor_(PreprocessConfig{
          config_.inputWidth,
          config_.inputHeight,
          config_.mean,
          config_.std
      }) {
    if (config_.batchSize <= 0 || config_.inputWidth <= 0 || config_.inputHeight <= 0) {
        setError("batch size and input dimensions must be positive");
        return;
    }

    if (std::any_of(config_.std.begin(), config_.std.end(), [](float value) {
            return value == 0.0f;
        })) {
        setError("preprocess std values must be non-zero");
        return;
    }

    imageElementCount_ = static_cast<size_t>(3) *
                         static_cast<size_t>(config_.inputWidth) *
                         static_cast<size_t>(config_.inputHeight);
    batchElementCount_ = static_cast<size_t>(config_.batchSize) * imageElementCount_;
    deviceInputBytes_ = batchElementCount_ * sizeof(float);

    if (!collectImages()) {
        if (hasReadableCache()) {
            valid_ = true;
            utils::FileLogger::instance().warning(
                "[INT8 Calibrator] calibration images unavailable; existing cache will be used"
            );
        }
        return;
    }

    if (!allocateDeviceBuffer()) {
        return;
    }

    valid_ = true;

    std::ostringstream oss;
    oss << "[INT8 Calibrator] ready, images=" << imagePaths_.size()
        << ", batch=" << config_.batchSize
        << ", input=1x3x" << config_.inputHeight << "x" << config_.inputWidth
        << ", input_name=" << config_.inputTensorName;
    utils::FileLogger::instance().info(oss.str());
}

Int8Calibrator::~Int8Calibrator() {
    if (deviceInput_ != nullptr) {
        cudaFree(deviceInput_);
        deviceInput_ = nullptr;
    }
}

int Int8Calibrator::getBatchSize() const noexcept {
    return config_.batchSize;
}

bool Int8Calibrator::getBatch(
    void* bindings[],
    const char* names[],
    int nbBindings
) noexcept {
    try {
        if (!valid_ || deviceInput_ == nullptr || bindings == nullptr) {
            return false;
        }

        if (nextImageIndex_ >= imagePaths_.size()) {
            utils::FileLogger::instance().info(
                "[INT8 Calibrator] all calibration images have been consumed"
            );
            return false;
        }

        size_t validImageCount = 0;
        if (!prepareBatch(validImageCount)) {
            if (processedImageCount_ == 0) {
                setError("no readable calibration images were provided");
            }
            return false;
        }

        const cudaError_t status = cudaMemcpy(
            deviceInput_,
            hostBatch_.data(),
            deviceInputBytes_,
            cudaMemcpyHostToDevice
        );

        if (status != cudaSuccess) {
            setError(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(status));
            return false;
        }

        const int inputIndex = findInputBinding(names, nbBindings);
        if (inputIndex < 0) {
            setError("cannot find TensorRT calibration input binding");
            return false;
        }

        bindings[inputIndex] = deviceInput_;

        std::ostringstream oss;
        oss << "[INT8 Calibrator] provide batch, valid_images=" << validImageCount
            << ", processed=" << processedImageCount_
            << "/" << imagePaths_.size();
        utils::FileLogger::instance().info(oss.str());
        return true;
    } catch (const std::exception& e) {
        setError(std::string("getBatch exception: ") + e.what());
        return false;
    } catch (...) {
        setError("getBatch unknown exception");
        return false;
    }
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept {
    length = 0;
    calibrationCache_.clear();

    if (!config_.readCache || config_.cacheFile.empty()) {
        return nullptr;
    }

    try {
        std::ifstream file(config_.cacheFile, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            utils::FileLogger::instance().info(
                "[INT8 Calibrator] cache not found, calibration will run"
            );
            return nullptr;
        }

        const std::streamsize fileSize = file.tellg();
        if (fileSize <= 0) {
            utils::FileLogger::instance().warning(
                "[INT8 Calibrator] cache file is empty: " + config_.cacheFile
            );
            return nullptr;
        }

        file.seekg(0, std::ios::beg);
        calibrationCache_.resize(static_cast<size_t>(fileSize));

        if (!file.read(calibrationCache_.data(), fileSize)) {
            calibrationCache_.clear();
            setError("failed to read calibration cache: " + config_.cacheFile);
            return nullptr;
        }

        length = calibrationCache_.size();
        utils::FileLogger::instance().info(
            "[INT8 Calibrator] loaded cache: " + config_.cacheFile
        );
        return calibrationCache_.data();
    } catch (const std::exception& e) {
        setError(std::string("read cache exception: ") + e.what());
        return nullptr;
    } catch (...) {
        setError("read cache unknown exception");
        return nullptr;
    }
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    if (cache == nullptr || length == 0 || config_.cacheFile.empty()) {
        return;
    }

    try {
        const fs::path cachePath(config_.cacheFile);
        const fs::path parent = cachePath.parent_path();

        if (!parent.empty()) {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec) {
                setError("failed to create cache directory: " + ec.message());
                return;
            }
        }

        std::ofstream file(config_.cacheFile, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            setError("failed to open calibration cache for writing: " + config_.cacheFile);
            return;
        }

        file.write(static_cast<const char*>(cache), static_cast<std::streamsize>(length));
        if (!file.good()) {
            setError("failed to write calibration cache: " + config_.cacheFile);
            return;
        }

        utils::FileLogger::instance().info(
            "[INT8 Calibrator] wrote cache: " + config_.cacheFile
        );
    } catch (const std::exception& e) {
        setError(std::string("write cache exception: ") + e.what());
    } catch (...) {
        setError("write cache unknown exception");
    }
}

bool Int8Calibrator::isValid() const noexcept {
    return valid_;
}

const std::string& Int8Calibrator::lastError() const noexcept {
    return lastError_;
}

size_t Int8Calibrator::imageCount() const noexcept {
    return imagePaths_.size();
}

bool Int8Calibrator::collectImages() {
    if (config_.imageDirectory.empty()) {
        setError("calibration image directory is empty");
        return false;
    }

    const fs::path imageDirectory(config_.imageDirectory);
    std::error_code ec;

    if (!fs::exists(imageDirectory, ec) || !fs::is_directory(imageDirectory, ec)) {
        setError("calibration image directory does not exist: " + config_.imageDirectory);
        return false;
    }

    fs::recursive_directory_iterator iterator(
        imageDirectory,
        fs::directory_options::skip_permission_denied,
        ec
    );
    const fs::recursive_directory_iterator end;

    while (iterator != end) {
        if (ec) {
            ec.clear();
            iterator.increment(ec);
            continue;
        }

        if (iterator->is_regular_file(ec)) {
            const std::string path = iterator->path().string();
            if (isSupportedImage(path)) {
                imagePaths_.push_back(path);
            }
        }

        iterator.increment(ec);
    }

    std::sort(imagePaths_.begin(), imagePaths_.end());

    if (config_.maxImages > 0 && imagePaths_.size() > config_.maxImages) {
        imagePaths_.resize(config_.maxImages);
    }

    if (imagePaths_.empty()) {
        setError("no supported calibration images found in: " + config_.imageDirectory);
        return false;
    }

    return true;
}

bool Int8Calibrator::allocateDeviceBuffer() {
    hostBatch_.resize(batchElementCount_);

    const cudaError_t status = cudaMalloc(&deviceInput_, deviceInputBytes_);
    if (status != cudaSuccess) {
        deviceInput_ = nullptr;
        setError(
            std::string("cudaMalloc calibration input failed: ") +
            cudaGetErrorString(status)
        );
        return false;
    }

    return true;
}

bool Int8Calibrator::prepareBatch(size_t& validImageCount) {
    validImageCount = 0;

    while (
        validImageCount < static_cast<size_t>(config_.batchSize) &&
        nextImageIndex_ < imagePaths_.size()
    ) {
        const std::string& imagePath = imagePaths_[nextImageIndex_++];

        if (!copyImageToBatch(imagePath, validImageCount)) {
            utils::FileLogger::instance().warning(
                "[INT8 Calibrator] skip invalid image: " + imagePath
            );
            continue;
        }

        ++validImageCount;
        ++processedImageCount_;
    }

    if (validImageCount == 0) {
        return false;
    }

    for (
        size_t batchIndex = validImageCount;
        batchIndex < static_cast<size_t>(config_.batchSize);
        ++batchIndex
    ) {
        const float* source = hostBatch_.data() +
                              (validImageCount - 1) * imageElementCount_;
        float* destination = hostBatch_.data() + batchIndex * imageElementCount_;
        std::memcpy(destination, source, imageElementCount_ * sizeof(float));
    }

    return true;
}

bool Int8Calibrator::copyImageToBatch(
    const std::string& imagePath,
    size_t batchIndex
) {
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        return false;
    }

    PreprocessResult prep = preprocessor_.process(image);
    if (prep.blob.empty() || prep.blob.type() != CV_32F) {
        return false;
    }

    if (prep.blob.total() != imageElementCount_) {
        utils::FileLogger::instance().error(
            "[INT8 Calibrator] unexpected blob shape for image: " + imagePath
        );
        return false;
    }

    const cv::Mat contiguousBlob = prep.blob.isContinuous()
        ? prep.blob
        : prep.blob.clone();

    float* destination = hostBatch_.data() + batchIndex * imageElementCount_;
    std::memcpy(
        destination,
        contiguousBlob.ptr<float>(),
        imageElementCount_ * sizeof(float)
    );
    return true;
}

bool Int8Calibrator::isSupportedImage(const std::string& path) const {
    std::string extension = fs::path(path).extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );

    return extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".png" ||
           extension == ".bmp" ||
           extension == ".tif" ||
           extension == ".tiff";
}

int Int8Calibrator::findInputBinding(
    const char* names[],
    int nbBindings
) const {
    if (nbBindings <= 0) {
        return -1;
    }

    if (config_.inputTensorName.empty()) {
        return 0;
    }

    if (names == nullptr) {
        return -1;
    }

    for (int index = 0; index < nbBindings; ++index) {
        if (names[index] != nullptr && config_.inputTensorName == names[index]) {
            return index;
        }
    }

    return -1;
}

bool Int8Calibrator::hasReadableCache() const {
    if (!config_.readCache || config_.cacheFile.empty()) {
        return false;
    }

    std::error_code ec;
    if (!fs::is_regular_file(config_.cacheFile, ec) || ec) {
        return false;
    }

    const auto cacheSize = fs::file_size(config_.cacheFile, ec);
    return !ec && cacheSize > 0;
}

void Int8Calibrator::setError(const std::string& message) {
    lastError_ = message;
    utils::FileLogger::instance().error("[INT8 Calibrator] " + message);
}
