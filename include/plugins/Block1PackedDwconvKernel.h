#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace egcinet::plugins {

// 输入和输出均为 token-major FP16：[batch, height * width, channels]。
// packedWeight 的物理布局为 [kernel=9][channelPair=C/2]，每个元素是 half2；
// packedBias 的物理布局为 [channelPair=C/2]，每个元素同样是 half2；
// fuseGelu 为 true 时在 FP32 累加结果上融合 FastGELU，再舍入为 FP16。
int32_t launchBlock1PackedDwconv(
    const void* input,
    const void* packedWeight,
    const void* packedBias,
    void* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels,
    bool fuseGelu,
    cudaStream_t stream
) noexcept;

} // 命名空间 egcinet::plugins
