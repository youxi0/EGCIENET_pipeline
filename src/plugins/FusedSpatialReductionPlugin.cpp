#include "plugins/FusedSpatialReductionPlugin.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace egcinet::plugins {

int32_t launchPackSpatialReductionWindows(
    int32_t stage,
    int32_t int8Mode,
    bool floatInput,
    float activationScale,
    const void* input,
    void* packedInput,
    cudaStream_t stream
) noexcept;

int32_t launchDequantizeSpatialReductionOutput(
    const void* accumulator,
    const void* bias,
    const void* dequantScale,
    void* output,
    int32_t outputTokens,
    int32_t outputChannels,
    int32_t accumulatorTokens,
    cudaStream_t stream
) noexcept;

namespace {

constexpr int32_t kSuccess = 0;
constexpr int32_t kFailure = -1;
constexpr char kFp16InputLayout[] = "BNC_FP16";
constexpr char kInt8InputLayout[] = "BNC_INT8";
constexpr char kFusedQuantizeInputLayout[] = "BNC_FP_PACK_INT8";
constexpr char kOutputLayout[] = "BNC_FP16";
constexpr char kFp16PackedWeightLayout[] =
    "KHKW_CI_CO_ROW_MAJOR_FP16";
constexpr char kInt8PackedWeightLayout[] =
    "CO_KHKW_CI_ROW_MAJOR_INT8";
constexpr int32_t kInt8GemmTokens = 128;

bool isSupportedParameters(
    const FusedSpatialReductionParameters& parameters
) noexcept {
    if (parameters.groups != 1 || parameters.fuseLayerNorm != 0 ||
        parameters.outputHeight != 11 || parameters.outputWidth != 11 ||
        parameters.inputChannels != parameters.outputChannels ||
        parameters.packedN != parameters.outputChannels ||
        parameters.packedK !=
            parameters.kernelHeight * parameters.kernelWidth *
            parameters.inputChannels ||
        parameters.layerIndex < 0 || parameters.int8Mode < 0 ||
        parameters.int8Mode > 2 ||
        (parameters.int8Mode == 2 &&
         (!std::isfinite(parameters.activationScale) ||
          parameters.activationScale <= 0.0F))) {
        return false;
    }

    switch (parameters.stage) {
    case 1:
        return parameters.inputHeight == 88 &&
               parameters.inputWidth == 88 &&
               parameters.inputChannels == 64 &&
               parameters.kernelHeight == 8 &&
               parameters.kernelWidth == 8 &&
               parameters.strideHeight == 8 &&
               parameters.strideWidth == 8 &&
               parameters.packedK == 4096;
    case 2:
        return parameters.inputHeight == 44 &&
               parameters.inputWidth == 44 &&
               parameters.inputChannels == 128 &&
               parameters.kernelHeight == 4 &&
               parameters.kernelWidth == 4 &&
               parameters.strideHeight == 4 &&
               parameters.strideWidth == 4 &&
               parameters.packedK == 2048;
    case 3:
        return parameters.inputHeight == 22 &&
               parameters.inputWidth == 22 &&
               parameters.inputChannels == 320 &&
               parameters.kernelHeight == 2 &&
               parameters.kernelWidth == 2 &&
               parameters.strideHeight == 2 &&
               parameters.strideWidth == 2 &&
               parameters.packedK == 1280;
    default:
        return false;
    }
}

bool isHalfLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kHALF &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isInt8Linear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kINT8 &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isFloatLinear(const nvinfer1::PluginTensorDesc& descriptor) noexcept {
    return descriptor.type == nvinfer1::DataType::kFLOAT &&
           descriptor.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isHalfOrFloatLinear(
    const nvinfer1::PluginTensorDesc& descriptor
) noexcept {
    return descriptor.format == nvinfer1::PluginFormat::kLINEAR &&
           (descriptor.type == nvinfer1::DataType::kHALF ||
            descriptor.type == nvinfer1::DataType::kFLOAT);
}

bool matchesDimension(
    int32_t actual,
    int32_t expected,
    bool allowDynamic
) noexcept {
    return actual == expected || (allowDynamic && actual == -1);
}

bool readScalarInt32(
    const nvinfer1::PluginField& field,
    int32_t& value
) noexcept {
    if (field.data == nullptr || field.length != 1) {
        return false;
    }
    if (field.type == nvinfer1::PluginFieldType::kINT32) {
        value = *static_cast<const int32_t*>(field.data);
        return true;
    }
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        const int64_t source = *static_cast<const int64_t*>(field.data);
        if (source < std::numeric_limits<int32_t>::min() ||
            source > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        value = static_cast<int32_t>(source);
        return true;
    }
    return false;
}

bool readScalarFloat(
    const nvinfer1::PluginField& field,
    float& value
) noexcept {
    if (field.data == nullptr || field.length != 1 ||
        field.type != nvinfer1::PluginFieldType::kFLOAT32) {
        return false;
    }
    value = *static_cast<const float*>(field.data);
    return true;
}

bool readString(
    const nvinfer1::PluginField& field,
    std::string& value
) {
    if (field.data == nullptr || field.length < 0 ||
        field.type != nvinfer1::PluginFieldType::kCHAR) {
        return false;
    }
    const auto* source = static_cast<const char*>(field.data);
    value.assign(source, source + field.length);
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return true;
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " +
            std::to_string(static_cast<int32_t>(status))
        );
    }
}

} // namespace

struct FusedSpatialReductionRuntime {
    explicit FusedSpatialReductionRuntime(
        const FusedSpatialReductionParameters& parameters
    )
        : int8Mode(parameters.int8Mode != 0) {
        // FP16 继续用等价的 column-major NN 解释。INT8 IMMA 的普通布局
        // 只支持 TN，因此把 M 补齐到 128：packed input 解释为 [K,M]，
        // 离线转置的 weight 解释为 [K,N]，INT32 输出为 [M,N] column-major。
        const uint64_t m = static_cast<uint64_t>(
            int8Mode
                ? kInt8GemmTokens
                : parameters.outputHeight * parameters.outputWidth
        );
        const uint64_t k = static_cast<uint64_t>(parameters.packedK);
        const uint64_t n = static_cast<uint64_t>(parameters.packedN);

        checkCublas(cublasLtCreate(&handle), "cublasLtCreate");
        try {
            checkCublas(
                cublasLtMatmulDescCreate(
                    &operation,
                    int8Mode ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F,
                    int8Mode ? CUDA_R_32I : CUDA_R_32F
                ),
                "cublasLtMatmulDescCreate"
            );
            if (!int8Mode) {
                const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
                checkCublas(
                    cublasLtMatmulDescSetAttribute(
                        operation,
                        CUBLASLT_MATMUL_DESC_EPILOGUE,
                        &epilogue,
                        sizeof(epilogue)
                    ),
                    "set cuBLASLt bias epilogue"
                );
            } else {
                const cublasOperation_t transposeInput = CUBLAS_OP_T;
                checkCublas(
                    cublasLtMatmulDescSetAttribute(
                        operation,
                        CUBLASLT_MATMUL_DESC_TRANSA,
                        &transposeInput,
                        sizeof(transposeInput)
                    ),
                    "set cuBLASLt INT8 input transpose"
                );
            }

            checkCublas(
                cublasLtMatrixLayoutCreate(
                    &weightLayout,
                    int8Mode ? CUDA_R_8I : CUDA_R_16F,
                    int8Mode ? k : n,
                    int8Mode ? n : k,
                    static_cast<int64_t>(int8Mode ? k : n)
                ),
                "create cuBLASLt weight layout"
            );
            checkCublas(
                cublasLtMatrixLayoutCreate(
                    &inputLayout,
                    int8Mode ? CUDA_R_8I : CUDA_R_16F,
                    k,
                    m,
                    static_cast<int64_t>(k)
                ),
                "create cuBLASLt input layout"
            );
            checkCublas(
                cublasLtMatrixLayoutCreate(
                    &outputLayout,
                    int8Mode ? CUDA_R_32I : CUDA_R_16F,
                    int8Mode ? m : n,
                    int8Mode ? n : m,
                    static_cast<int64_t>(int8Mode ? m : n)
                ),
                "create cuBLASLt output layout"
            );

            cublasLtMatmulPreference_t preference = nullptr;
            checkCublas(
                cublasLtMatmulPreferenceCreate(&preference),
                "cublasLtMatmulPreferenceCreate"
            );
            try {
                const size_t maximumWorkspaceBytes = 0;
                checkCublas(
                    cublasLtMatmulPreferenceSetAttribute(
                        preference,
                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &maximumWorkspaceBytes,
                        sizeof(maximumWorkspaceBytes)
                    ),
                    "set cuBLASLt workspace preference"
                );

                if (!int8Mode) {
                    // Heuristic 只读取指针对齐，不会解引用 bias。实际
                    // enqueue 前会用本次网络输入的设备地址覆盖它。
                    const void* alignedBiasPlaceholder =
                        reinterpret_cast<const void*>(256U);
                    checkCublas(
                        cublasLtMatmulDescSetAttribute(
                            operation,
                            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                            &alignedBiasPlaceholder,
                            sizeof(alignedBiasPlaceholder)
                        ),
                        "set cuBLASLt heuristic bias pointer"
                    );
                }

                cublasLtMatmulHeuristicResult_t heuristic{};
                int32_t returnedResults = 0;
                checkCublas(
                    cublasLtMatmulAlgoGetHeuristic(
                        handle,
                        operation,
                        int8Mode ? inputLayout : weightLayout,
                        int8Mode ? weightLayout : inputLayout,
                        outputLayout,
                        outputLayout,
                        preference,
                        1,
                        &heuristic,
                        &returnedResults
                    ),
                    "cublasLtMatmulAlgoGetHeuristic"
                );
                if (returnedResults != 1 ||
                    heuristic.state != CUBLAS_STATUS_SUCCESS ||
                    heuristic.workspaceSize != 0) {
                    throw std::runtime_error(
                        int8Mode
                            ? "no zero-workspace cuBLASLt INT8 SR algorithm"
                            : "no zero-workspace cuBLASLt FP16 SR algorithm"
                    );
                }
                algorithm = heuristic.algo;
                algorithmReady = true;
            } catch (...) {
                cublasLtMatmulPreferenceDestroy(preference);
                throw;
            }
            checkCublas(
                cublasLtMatmulPreferenceDestroy(preference),
                "cublasLtMatmulPreferenceDestroy"
            );
        } catch (...) {
            reset();
            throw;
        }
    }

    ~FusedSpatialReductionRuntime() noexcept {
        reset();
    }

    FusedSpatialReductionRuntime(
        const FusedSpatialReductionRuntime&
    ) = delete;
    FusedSpatialReductionRuntime& operator=(
        const FusedSpatialReductionRuntime&
    ) = delete;

    int32_t matmul(
        const void* packedInput,
        const void* weight,
        const void* bias,
        void* output,
        cudaStream_t stream
    ) noexcept {
        if (!algorithmReady || packedInput == nullptr || weight == nullptr ||
            bias == nullptr || output == nullptr) {
            return kFailure;
        }

        if (int8Mode) {
            constexpr int32_t alpha = 1;
            constexpr int32_t beta = 0;
            return cublasLtMatmul(
                       handle,
                       operation,
                       &alpha,
                       packedInput,
                       inputLayout,
                       weight,
                       weightLayout,
                       &beta,
                       output,
                       outputLayout,
                       output,
                       outputLayout,
                       &algorithm,
                       nullptr,
                       0,
                       stream
                   ) == CUBLAS_STATUS_SUCCESS
                ? kSuccess
                : kFailure;
        }

        const void* biasPointer = bias;
        if (cublasLtMatmulDescSetAttribute(
                operation,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &biasPointer,
                sizeof(biasPointer)
            ) != CUBLAS_STATUS_SUCCESS) {
            return kFailure;
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        return cublasLtMatmul(
                   handle,
                   operation,
                   &alpha,
                   weight,
                   weightLayout,
                   packedInput,
                   inputLayout,
                   &beta,
                   output,
                   outputLayout,
                   output,
                   outputLayout,
                   &algorithm,
                   nullptr,
                   0,
                   stream
               ) == CUBLAS_STATUS_SUCCESS
            ? kSuccess
            : kFailure;
    }

private:
    void reset() noexcept {
        if (outputLayout != nullptr) {
            cublasLtMatrixLayoutDestroy(outputLayout);
            outputLayout = nullptr;
        }
        if (inputLayout != nullptr) {
            cublasLtMatrixLayoutDestroy(inputLayout);
            inputLayout = nullptr;
        }
        if (weightLayout != nullptr) {
            cublasLtMatrixLayoutDestroy(weightLayout);
            weightLayout = nullptr;
        }
        if (operation != nullptr) {
            cublasLtMatmulDescDestroy(operation);
            operation = nullptr;
        }
        if (handle != nullptr) {
            cublasLtDestroy(handle);
            handle = nullptr;
        }
        algorithmReady = false;
    }

    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weightLayout = nullptr;
    cublasLtMatrixLayout_t inputLayout = nullptr;
    cublasLtMatrixLayout_t outputLayout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    bool algorithmReady = false;
    bool int8Mode = false;
};

FusedSpatialReductionPlugin::FusedSpatialReductionPlugin(
    FusedSpatialReductionParameters parameters
)
    : parameters_(parameters) {
    if (!isSupportedParameters(parameters_)) {
        throw std::invalid_argument(
            "unsupported fused spatial reduction parameters"
        );
    }
    runtime_ = std::make_unique<FusedSpatialReductionRuntime>(parameters_);
}

FusedSpatialReductionPlugin::~FusedSpatialReductionPlugin() noexcept = default;

nvinfer1::IPluginCapability*
FusedSpatialReductionPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type
) noexcept {
    switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    default:
        return nullptr;
    }
}

nvinfer1::IPluginV3* FusedSpatialReductionPlugin::clone() noexcept {
    try {
        auto* plugin = new FusedSpatialReductionPlugin(parameters_);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] clone failed: %s\n",
            error.what()
        );
        return nullptr;
    } catch (...) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] clone failed: "
            "unknown exception\n"
        );
        return nullptr;
    }
}

const char* FusedSpatialReductionPlugin::getPluginName() const noexcept {
    return kFusedSpatialReductionPluginName;
}

const char* FusedSpatialReductionPlugin::getPluginVersion() const noexcept {
    return kFusedSpatialReductionPluginVersion;
}

const char* FusedSpatialReductionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void FusedSpatialReductionPlugin::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    if (pluginNamespace == nullptr) {
        return;
    }
    try {
        namespace_ = pluginNamespace;
    } catch (...) {
    }
}

int32_t FusedSpatialReductionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t FusedSpatialReductionPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 ||
        nbInputs != expectedInputs) {
        return kFailure;
    }
    if (parameters_.int8Mode != 0) {
        const bool inputTypeValid = parameters_.int8Mode == 1
            ? inputTypes[0] == nvinfer1::DataType::kINT8
            : inputTypes[0] == nvinfer1::DataType::kHALF ||
                  inputTypes[0] == nvinfer1::DataType::kFLOAT;
        if (!inputTypeValid ||
            inputTypes[1] != nvinfer1::DataType::kINT8 ||
            inputTypes[2] != nvinfer1::DataType::kHALF ||
            inputTypes[3] != nvinfer1::DataType::kFLOAT) {
            return kFailure;
        }
    } else if (inputTypes[0] != nvinfer1::DataType::kHALF ||
               inputTypes[1] != nvinfer1::DataType::kHALF ||
               inputTypes[2] != nvinfer1::DataType::kHALF) {
        return kFailure;
    }
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return kSuccess;
}

int32_t FusedSpatialReductionPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* /* shapeInputs */,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedInputs ||
        nbOutputs != 1 || nbShapeInputs != 0 || inputs[0].nbDims != 3 ||
        inputs[1].nbDims != 2 || inputs[2].nbDims != 1 ||
        (parameters_.int8Mode != 0 && inputs[3].nbDims != 1)) {
        return kFailure;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(
        parameters_.outputHeight * parameters_.outputWidth
    );
    outputs[0].d[2] = exprBuilder.constant(parameters_.outputChannels);
    return kSuccess;
}

bool FusedSpatialReductionPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (inOut == nullptr || nbInputs != expectedInputs || nbOutputs != 1 ||
        pos < 0 ||
        pos >= nbInputs + nbOutputs) {
        return false;
    }
    if (parameters_.int8Mode == 0) {
        return isHalfLinear(inOut[pos].desc);
    }
    switch (pos) {
    case 0:
        return parameters_.int8Mode == 1
            ? isInt8Linear(inOut[pos].desc)
            : isHalfOrFloatLinear(inOut[pos].desc);
    case 1:
        return isInt8Linear(inOut[pos].desc);
    case 2:
    case 4:
        return isHalfLinear(inOut[pos].desc);
    case 3:
        return isFloatLinear(inOut[pos].desc);
    default:
        return false;
    }
}

bool FusedSpatialReductionPlugin::validateDescriptors(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs,
    bool allowDynamic
) const noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (inputs == nullptr || outputs == nullptr ||
        nbInputs != expectedInputs ||
        nbOutputs != 1 || runtime_ == nullptr ||
        inputs[0].dims.nbDims != 3 || inputs[1].dims.nbDims != 2 ||
        inputs[2].dims.nbDims != 1 || outputs[0].dims.nbDims != 3 ||
        (parameters_.int8Mode != 0 && inputs[3].dims.nbDims != 1)) {
        return false;
    }
    const bool int8InputValid = parameters_.int8Mode == 1
        ? isInt8Linear(inputs[0])
        : isHalfOrFloatLinear(inputs[0]);
    const bool formatsValid = parameters_.int8Mode != 0
        ? int8InputValid && isInt8Linear(inputs[1]) &&
              isHalfLinear(inputs[2]) && isFloatLinear(inputs[3])
        : isHalfLinear(inputs[0]) && isHalfLinear(inputs[1]) &&
              isHalfLinear(inputs[2]);
    if (!formatsValid || !isHalfLinear(outputs[0])) {
        return false;
    }

    return
        matchesDimension(inputs[0].dims.d[0], 1, allowDynamic) &&
        matchesDimension(
            inputs[0].dims.d[1],
            parameters_.inputHeight * parameters_.inputWidth,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[0].dims.d[2], parameters_.inputChannels, allowDynamic
        ) &&
        matchesDimension(
            inputs[1].dims.d[0],
            parameters_.int8Mode != 0
                ? parameters_.packedN
                : parameters_.packedK,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[1].dims.d[1],
            parameters_.int8Mode != 0
                ? parameters_.packedK
                : parameters_.packedN,
            allowDynamic
        ) &&
        matchesDimension(
            inputs[2].dims.d[0], parameters_.packedN, allowDynamic
        ) &&
        (parameters_.int8Mode == 0 ||
         matchesDimension(
             inputs[3].dims.d[0], parameters_.packedN, allowDynamic
         )) &&
        matchesDimension(outputs[0].dims.d[0], 1, allowDynamic) &&
        matchesDimension(
            outputs[0].dims.d[1],
            parameters_.outputHeight * parameters_.outputWidth,
            allowDynamic
        ) &&
        matchesDimension(
            outputs[0].dims.d[2], parameters_.outputChannels, allowDynamic
        );
}

int32_t FusedSpatialReductionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    if (inputs == nullptr || outputs == nullptr) {
        return kFailure;
    }
    nvinfer1::PluginTensorDesc inputDescriptors[4]{
        inputs[0].desc,
        inputs[1].desc,
        inputs[2].desc,
        {},
    };
    if (parameters_.int8Mode != 0 && nbInputs == 4) {
        inputDescriptors[3] = inputs[3].desc;
    }
    return validateDescriptors(
               inputDescriptors,
               nbInputs,
               &outputs[0].desc,
               nbOutputs,
               true
           )
        ? kSuccess
        : kFailure;
}

int32_t FusedSpatialReductionPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) noexcept {
    return validateDescriptors(inputs, nbInputs, outputs, nbOutputs, false)
        ? kSuccess
        : kFailure;
}

size_t FusedSpatialReductionPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* /* inputs */,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* /* outputs */,
    int32_t nbOutputs
) const noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (nbInputs != expectedInputs || nbOutputs != 1) {
        return 0;
    }
    const size_t spatialOutputTokens =
        static_cast<size_t>(parameters_.outputHeight) *
        static_cast<size_t>(parameters_.outputWidth);
    const size_t gemmTokens = parameters_.int8Mode != 0
        ? static_cast<size_t>(kInt8GemmTokens)
        : spatialOutputTokens;
    const size_t packedBytes = gemmTokens *
        static_cast<size_t>(parameters_.packedK) *
        (parameters_.int8Mode != 0 ? sizeof(int8_t) : sizeof(half));
    const size_t accumulatorBytes = parameters_.int8Mode != 0
        ? gemmTokens * static_cast<size_t>(parameters_.packedN) *
              sizeof(int32_t)
        : 0;
    return packedBytes + accumulatorBytes;
}

int32_t FusedSpatialReductionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    const int32_t expectedInputs = parameters_.int8Mode != 0 ? 4 : 3;
    if (!validateDescriptors(
            inputDesc, expectedInputs, outputDesc, 1, false
        ) ||
        inputs == nullptr || outputs == nullptr || workspace == nullptr ||
        inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr ||
        (parameters_.int8Mode != 0 && inputs[3] == nullptr) ||
        outputs[0] == nullptr) {
        return kFailure;
    }

    if (launchPackSpatialReductionWindows(
            parameters_.stage,
            parameters_.int8Mode,
            inputDesc[0].type == nvinfer1::DataType::kFLOAT,
            parameters_.activationScale,
            inputs[0],
            workspace,
            stream
        ) != kSuccess) {
        return kFailure;
    }
    if (parameters_.int8Mode == 0) {
        return runtime_->matmul(
            workspace,
            inputs[1],
            inputs[2],
            outputs[0],
            stream
        );
    }

    const size_t outputTokens =
        static_cast<size_t>(parameters_.outputHeight) *
        static_cast<size_t>(parameters_.outputWidth);
    const size_t packedBytes = static_cast<size_t>(kInt8GemmTokens) *
        static_cast<size_t>(parameters_.packedK) * sizeof(int8_t);
    auto* accumulator = static_cast<unsigned char*>(workspace) + packedBytes;
    if (runtime_->matmul(
            workspace,
            inputs[1],
            inputs[2],
            accumulator,
            stream
        ) != kSuccess) {
        return kFailure;
    }
    return launchDequantizeSpatialReductionOutput(
        accumulator,
        inputs[2],
        inputs[3],
        outputs[0],
        static_cast<int32_t>(outputTokens),
        parameters_.outputChannels,
        kInt8GemmTokens,
        stream
    );
}

nvinfer1::IPluginV3* FusedSpatialReductionPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /* context */
) noexcept {
    return clone();
}

const nvinfer1::PluginFieldCollection*
FusedSpatialReductionPlugin::getFieldsToSerialize() noexcept {
    try {
        serializedFields_.clear();
        auto addInt = [this](const char* name, int32_t* value) {
            serializedFields_.emplace_back(
                name, value, nvinfer1::PluginFieldType::kINT32, 1
            );
        };
        addInt("stage", &parameters_.stage);
        addInt("layer_index", &parameters_.layerIndex);
        addInt("input_height", &parameters_.inputHeight);
        addInt("input_width", &parameters_.inputWidth);
        addInt("input_channels", &parameters_.inputChannels);
        addInt("output_height", &parameters_.outputHeight);
        addInt("output_width", &parameters_.outputWidth);
        addInt("output_channels", &parameters_.outputChannels);
        addInt("kernel_height", &parameters_.kernelHeight);
        addInt("kernel_width", &parameters_.kernelWidth);
        addInt("stride_height", &parameters_.strideHeight);
        addInt("stride_width", &parameters_.strideWidth);
        addInt("groups", &parameters_.groups);
        addInt("packed_k", &parameters_.packedK);
        addInt("packed_n", &parameters_.packedN);
        addInt("fuse_layernorm", &parameters_.fuseLayerNorm);
        addInt("int8_mode", &parameters_.int8Mode);
        serializedFields_.emplace_back(
            "activation_scale",
            &parameters_.activationScale,
            nvinfer1::PluginFieldType::kFLOAT32,
            1
        );
        const char* inputLayout = parameters_.int8Mode == 0
            ? kFp16InputLayout
            : parameters_.int8Mode == 1
                ? kInt8InputLayout
                : kFusedQuantizeInputLayout;
        const char* weightLayout = parameters_.int8Mode != 0
            ? kInt8PackedWeightLayout
            : kFp16PackedWeightLayout;
        serializedFields_.emplace_back(
            "input_layout",
            inputLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(std::string_view(inputLayout).size())
        );
        serializedFields_.emplace_back(
            "output_layout",
            kOutputLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(sizeof(kOutputLayout) - 1)
        );
        serializedFields_.emplace_back(
            "packed_weight_layout",
            weightLayout,
            nvinfer1::PluginFieldType::kCHAR,
            static_cast<int32_t>(std::string_view(weightLayout).size())
        );
        serializedFieldCollection_.nbFields =
            static_cast<int32_t>(serializedFields_.size());
        serializedFieldCollection_.fields = serializedFields_.data();
        return &serializedFieldCollection_;
    } catch (...) {
        return nullptr;
    }
}

FusedSpatialReductionPluginCreator::
FusedSpatialReductionPluginCreator() noexcept {
    try {
        constexpr const char* kIntegerFields[] = {
            "stage",
            "layer_index",
            "input_height",
            "input_width",
            "input_channels",
            "output_height",
            "output_width",
            "output_channels",
            "kernel_height",
            "kernel_width",
            "stride_height",
            "stride_width",
            "groups",
            "packed_k",
            "packed_n",
            "fuse_layernorm",
            "int8_mode",
        };
        for (const char* name : kIntegerFields) {
            fields_.emplace_back(
                name, nullptr, nvinfer1::PluginFieldType::kINT32, 1
            );
        }
        fields_.emplace_back(
            "activation_scale",
            nullptr,
            nvinfer1::PluginFieldType::kFLOAT32,
            1
        );
        constexpr const char* kStringFields[] = {
            "input_layout",
            "output_layout",
            "packed_weight_layout",
        };
        for (const char* name : kStringFields) {
            fields_.emplace_back(
                name, nullptr, nvinfer1::PluginFieldType::kCHAR, 0
            );
        }
        fieldCollection_.nbFields = static_cast<int32_t>(fields_.size());
        fieldCollection_.fields = fields_.data();
    } catch (...) {
        fields_.clear();
        fieldCollection_ = {};
    }
}

const char* FusedSpatialReductionPluginCreator::getPluginName() const noexcept {
    return kFusedSpatialReductionPluginName;
}

const char* FusedSpatialReductionPluginCreator::getPluginVersion() const noexcept {
    return kFusedSpatialReductionPluginVersion;
}

const nvinfer1::PluginFieldCollection*
FusedSpatialReductionPluginCreator::getFieldNames() noexcept {
    return &fieldCollection_;
}

const char*
FusedSpatialReductionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

void FusedSpatialReductionPluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    if (pluginNamespace == nullptr) {
        return;
    }
    try {
        namespace_ = pluginNamespace;
    } catch (...) {
    }
}

nvinfer1::IPluginV3* FusedSpatialReductionPluginCreator::createPlugin(
    const char* /* name */,
    const nvinfer1::PluginFieldCollection* fieldCollection,
    nvinfer1::TensorRTPhase /* phase */
) noexcept {
    if (fieldCollection == nullptr || fieldCollection->fields == nullptr) {
        return nullptr;
    }

    try {
        FusedSpatialReductionParameters parameters{};
        std::string inputLayout;
        std::string outputLayout;
        std::string weightLayout;
        uint32_t seen = 0;

        for (int32_t index = 0; index < fieldCollection->nbFields; ++index) {
            const nvinfer1::PluginField& field = fieldCollection->fields[index];
            if (field.name == nullptr) {
                continue;
            }
            const std::string_view name(field.name);
            int32_t* destination = nullptr;
            uint32_t bit = 0;
            if (name == "stage") {
                destination = &parameters.stage; bit = 1U << 0;
            } else if (name == "layer_index") {
                destination = &parameters.layerIndex; bit = 1U << 1;
            } else if (name == "input_height") {
                destination = &parameters.inputHeight; bit = 1U << 2;
            } else if (name == "input_width") {
                destination = &parameters.inputWidth; bit = 1U << 3;
            } else if (name == "input_channels") {
                destination = &parameters.inputChannels; bit = 1U << 4;
            } else if (name == "output_height") {
                destination = &parameters.outputHeight; bit = 1U << 5;
            } else if (name == "output_width") {
                destination = &parameters.outputWidth; bit = 1U << 6;
            } else if (name == "output_channels") {
                destination = &parameters.outputChannels; bit = 1U << 7;
            } else if (name == "kernel_height") {
                destination = &parameters.kernelHeight; bit = 1U << 8;
            } else if (name == "kernel_width") {
                destination = &parameters.kernelWidth; bit = 1U << 9;
            } else if (name == "stride_height") {
                destination = &parameters.strideHeight; bit = 1U << 10;
            } else if (name == "stride_width") {
                destination = &parameters.strideWidth; bit = 1U << 11;
            } else if (name == "groups") {
                destination = &parameters.groups; bit = 1U << 12;
            } else if (name == "packed_k") {
                destination = &parameters.packedK; bit = 1U << 13;
            } else if (name == "packed_n") {
                destination = &parameters.packedN; bit = 1U << 14;
            } else if (name == "fuse_layernorm") {
                destination = &parameters.fuseLayerNorm; bit = 1U << 15;
            } else if (name == "int8_mode") {
                destination = &parameters.int8Mode; bit = 1U << 16;
            }

            if (destination != nullptr) {
                if (!readScalarInt32(field, *destination)) {
                    return nullptr;
                }
                seen |= bit;
            } else if (name == "activation_scale") {
                if (!readScalarFloat(field, parameters.activationScale)) {
                    return nullptr;
                }
            } else if (name == "input_layout") {
                if (!readString(field, inputLayout)) {
                    return nullptr;
                }
            } else if (name == "output_layout") {
                if (!readString(field, outputLayout)) {
                    return nullptr;
                }
            } else if (name == "packed_weight_layout") {
                if (!readString(field, weightLayout)) {
                    return nullptr;
                }
            }
        }

        constexpr uint32_t kRequiredIntegerFields = (1U << 16) - 1U;
        const char* expectedInputLayout = parameters.int8Mode == 0
            ? kFp16InputLayout
            : parameters.int8Mode == 1
                ? kInt8InputLayout
                : kFusedQuantizeInputLayout;
        const char* expectedWeightLayout = parameters.int8Mode != 0
            ? kInt8PackedWeightLayout
            : kFp16PackedWeightLayout;
        if ((seen & kRequiredIntegerFields) != kRequiredIntegerFields ||
            (!inputLayout.empty() && inputLayout != expectedInputLayout) ||
            (!outputLayout.empty() && outputLayout != kOutputLayout) ||
            (!weightLayout.empty() && weightLayout != expectedWeightLayout)) {
            return nullptr;
        }

        auto* plugin = new FusedSpatialReductionPlugin(parameters);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] createPlugin failed: %s\n",
            error.what()
        );
        return nullptr;
    } catch (...) {
        std::fprintf(
            stderr,
            "[EGCINET_FusedSpatialReduction] createPlugin failed: "
            "unknown exception\n"
        );
        return nullptr;
    }
}

REGISTER_TENSORRT_PLUGIN(FusedSpatialReductionPluginCreator);

} // namespace egcinet::plugins
