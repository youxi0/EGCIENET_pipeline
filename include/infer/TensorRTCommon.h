#pragma once

#include <NvInfer.h>

#include <type_traits>
#include <utility>

// 构建工具和运行时推理共用的 TensorRT 日志器。
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

namespace trt_detail {

template <typename T, typename = void>
struct HasTensorRTDestroy : std::false_type {};

template <typename T>
struct HasTensorRTDestroy<T, std::void_t<decltype(std::declval<T*>()->destroy())>> : std::true_type {};

} // 命名空间 trt_detail

// 兼容不同 TensorRT 版本对象释放方式的 RAII 删除器。
// TensorRT 8.x 的某些对象，比如 IOptimizationProfile，既没有 destroy()
// 也没有公开析构函数；这类对象创建后由 TensorRT 内部管理生命周期。
template <typename T>
class TrtDestroy {
public:
    void operator()(T* obj) const {
        if (obj == nullptr) {
            return;
        }

        if constexpr (trt_detail::HasTensorRTDestroy<T>::value) {
            obj->destroy();
        } else if constexpr (std::is_destructible<T>::value) {
            delete obj;
        } else {
            (void)obj;
        }
    }
};

template <>
class TrtDestroy<nvinfer1::IOptimizationProfile> {
public:
    void operator()(nvinfer1::IOptimizationProfile* obj) const noexcept {
        // TensorRT 8.6 中 IOptimizationProfile 析构函数是 protected，且没有 destroy()。
        // engine 构建期间由 builder/config 管理生命周期，这里只保持指针存活，
        // 不通过 C++ delete 释放。
        (void)obj;
    }
};
