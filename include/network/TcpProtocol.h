#pragma once

#include <cstddef>
#include <cstdint>

namespace egcinet::network::protocol {

// TCP 是字节流协议。每帧线格式为 [84 字节头部][imageBytes 字节 JPEG]，
// 不依赖 C++ 结构体内存布局；所有整数均使用大端序，便于不同平台稳定互通。
inline constexpr char kMagic[4] = {'E', 'G', 'C', 'I'};
inline constexpr std::uint16_t kVersion = 1;
inline constexpr std::uint16_t kHeaderBytes = 84;
inline constexpr std::uint32_t kJpegEncoding = 1;
inline constexpr std::uint32_t kMaximumImageBytes = 16U * 1024U * 1024U;

// 固定头部字段偏移，供 Jetson 发送端和 Qt 接收端共享。
inline constexpr std::size_t kVersionOffset = 4;
inline constexpr std::size_t kHeaderBytesOffset = 6;
inline constexpr std::size_t kImageBytesOffset = 8;
inline constexpr std::size_t kFrameIdOffset = 12;
inline constexpr std::size_t kTimestampUsOffset = 20;
inline constexpr std::size_t kImageWidthOffset = 28;
inline constexpr std::size_t kImageHeightOffset = 32;
inline constexpr std::size_t kImageEncodingOffset = 36;
inline constexpr std::size_t kAcquireUsOffset = 40;
inline constexpr std::size_t kH2dUsOffset = 44;
inline constexpr std::size_t kPreprocessUsOffset = 48;
inline constexpr std::size_t kInferUsOffset = 52;
inline constexpr std::size_t kPostprocessUsOffset = 56;
inline constexpr std::size_t kVisualizeUsOffset = 60;
inline constexpr std::size_t kD2hUsOffset = 64;
inline constexpr std::size_t kTotalUsOffset = 68;
inline constexpr std::size_t kDroppedFramesOffset = 72;
inline constexpr std::size_t kSentFramesOffset = 76;
inline constexpr std::size_t kReservedOffset = 80;

// 修改字段时必须同步更新头部长度和 Qt 接收端，避免双方协议悄然错位。
static_assert(kReservedOffset + sizeof(std::uint32_t) == kHeaderBytes,
              "TCP protocol header layout does not match kHeaderBytes");

} // namespace egcinet::network::protocol
