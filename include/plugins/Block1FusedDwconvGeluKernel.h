#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace egcinet::plugins {

// 输入和输出均采用 token-major 布局：[batch, height * width, channels]。
// DWConv 权重布局为 [channels, 1, 3, 3]，偏置布局为 [channels]。
// 启动函数只负责参数校验和 CUDA kernel 调度，不申请额外 workspace。
int32_t launchBlock1FusedDwconvGelu(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t channels,
    cudaStream_t stream
) noexcept;

} // 命名空间 egcinet::plugins
