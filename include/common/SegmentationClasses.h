#pragma once

#include <array>
#include <cstdint>

namespace egcinet::segmentation {

// 模型输出固定为 [1, 5, H, W]，类别 ID 同训练标签的灰度值 0~4。
inline constexpr int kClassCount = 5;

enum class ClassId : std::uint8_t {
    Background = 0,
    Burn = 1,
    CrackTear = 2,
    MaterialLoss = 3,
    Deformation = 4
};

inline constexpr std::array<const char*, kClassCount> kClassNames{
    "background",
    "burn",
    "crack_tear",
    "material_loss",
    "deformation"
};

} // namespace egcinet::segmentation
