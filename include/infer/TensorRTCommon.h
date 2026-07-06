#pragma once

#include <NvInfer.h>

#include <type_traits>
#include <utility>

// TensorRT logger shared by build-time tools and runtime inference.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

namespace trt_detail {

template <typename T, typename = void>
struct HasTensorRTDestroy : std::false_type {};

template <typename T>
struct HasTensorRTDestroy<T, std::void_t<decltype(std::declval<T*>()->destroy())>> : std::true_type {};

} // namespace trt_detail

// RAII deleter for TensorRT objects across API versions.
// Some TensorRT 8.x objects, such as IOptimizationProfile, expose neither destroy()
// nor a public destructor. Those objects are owned by TensorRT internals after creation.
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
