#pragma once

#include "acquisition/ImageSource.h"
#include <string>
#include <vector>

class ImageFolderSource : public ImageSource {
public:
    explicit ImageFolderSource(const std::string& folder_path);

    bool open() override;

    bool read(FrameData& frame) override;

    void reset() override;

    void release() override;

private:
    bool isImageFile(const std::string& path) const;

private:
    std::string folder_path_;
    std::vector<std::string> image_paths_;
    size_t current_index_ = 0;

};