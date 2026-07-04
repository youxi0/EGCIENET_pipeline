#pragma once

#include "common/FrameData.h"

class ImageSource {
public:
    virtual ~ImageSource() = default;

    virtual bool open() = 0;

    virtual bool read(FrameData& frame) = 0;

    virtual void reset() = 0;

    virtual void release() = 0;
};