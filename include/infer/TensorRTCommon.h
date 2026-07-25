#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>

#if NV_TENSORRT_MAJOR < 10
#error "EGCINET runtime requires TensorRT 10 or newer."
#endif

// INT8 校准工具和运行时推理共用的 TensorRT 日志器。
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};
