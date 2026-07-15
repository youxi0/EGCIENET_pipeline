#pragma once

#include <NvInfer.h>

#include <type_traits>
#include <utility>

// INT8 校准工具和运行时推理共用的 TensorRT 日志器。
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
