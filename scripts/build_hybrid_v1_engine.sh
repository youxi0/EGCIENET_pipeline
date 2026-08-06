#!/bin/bash

set -Eeuo pipefail

# Hybrid V1 is a diagnostic mixed-precision preset for the current EGCIENet ONNX:
#   1. keep the three observed encoder block1 DWConv INT8 islands in FP16;
#   2. keep every FEM CBAM subgraph in FP16 as one precision island;
#   3. force the 12 FEM convolutions and 3 GAM conv_pre convolutions to INT8.
#
# The original build_int8_engine.sh remains the single implementation of the
# trtexec invocation. This wrapper only composes layer precision constraints and
# writes a separate engine by default.

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE_BUILDER="${PROJECT_ROOT}/scripts/build_int8_engine.sh"

ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352.onnx}"
ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_hybrid_v1.engine}"
CALIB_CACHE="${CALIB_CACHE:-${PROJECT_ROOT}/models/egcienet_352_int8.cache}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
VALIDATE_LAYER_NAMES="${VALIDATE_LAYER_NAMES:-1}"
PRINT_ONLY="${PRINT_ONLY:-0}"

# User constraints are appended after the preset. TensorRT reads specifications
# from left to right, so a duplicate user entry overrides the preset entry.
USER_LAYER_PRECISIONS="${LAYER_PRECISIONS:-}"
USER_LAYER_OUTPUT_TYPES="${LAYER_OUTPUT_TYPES:-}"
LAYER_PRECISIONS=""
LAYER_OUTPUT_TYPES="${USER_LAYER_OUTPUT_TYPES}"

PRESET_LAYER_NAMES=()
FP16_LAYER_COUNT=0
INT8_LAYER_COUNT=0

append_layer_precision() {
    local layer_name="$1"
    local precision="$2"

    if [ -n "${LAYER_PRECISIONS}" ]; then
        LAYER_PRECISIONS+=","
    fi
    LAYER_PRECISIONS+="${layer_name}:${precision}"
    PRESET_LAYER_NAMES+=("${layer_name}")

    case "${precision}" in
    fp16)
        FP16_LAYER_COUNT=$((FP16_LAYER_COUNT + 1))
        ;;
    int8)
        INT8_LAYER_COUNT=$((INT8_LAYER_COUNT + 1))
        ;;
    esac
}

# These are the only encoder layers that entered INT8 in the current profile.
# Keeping them in FP16 removes three isolated FP16 -> INT8 -> FP16 islands.
for block_index in 0 1 2; do
    append_layer_precision \
        "/model/block1.${block_index}/mlp/dwconv/dwconv/Conv" \
        fp16
done

# Keep each complete CBAM in FP16. Do not constrain the surrounding FEM
# convolutions here; they are explicitly set to INT8 below.
CBAM_LAYER_SUFFIXES=(
    "ca/avg_pool/GlobalAveragePool"
    "ca/max_pool/MaxPool"
    "ca/fc1/Conv"
    "ca/relu1/Relu"
    "ca/fc2/Conv"
    "ca/fc1_1/Conv"
    "ca/relu1_1/Relu"
    "ca/fc2_1/Conv"
    "ca/Add"
    "ca/sigmoid/Sigmoid"
    "Mul"
    "sa/ReduceMean"
    "sa/ReduceMax"
    "sa/Concat"
    "sa/conv1/Conv"
    "sa/sigmoid/Sigmoid"
    "Mul_1"
)

for fem_index in 1 2 3 4; do
    for suffix in "${CBAM_LAYER_SUFFIXES[@]}"; do
        append_layer_precision \
            "/model/decoder/FEM${fem_index}/cbam/${suffix}" \
            fp16
    done
done

# Preserve INT8 compute on the Decoder's large FEM convolutions.
for fem_index in 1 2 3 4; do
    for conv_index in 1 2 3; do
        append_layer_precision \
            "/model/decoder/FEM${fem_index}/conv${conv_index}/Conv" \
            int8
    done
done

# Preserve INT8 compute on the three multi-scale fusion convolutions.
for gam_name in d31 d42 d42_31; do
    append_layer_precision \
        "/model/decoder/${gam_name}/conv_pre/conv_pre.0/Conv" \
        int8
done

if [ -n "${USER_LAYER_PRECISIONS}" ]; then
    LAYER_PRECISIONS+=",${USER_LAYER_PRECISIONS}"
fi

validate_preset_layer_names() {
    local missing_count=0
    local layer_name

    if [ ! -s "${ONNX}" ]; then
        echo "[ERROR] ONNX model not found: ${ONNX}" >&2
        return 1
    fi

    for layer_name in "${PRESET_LAYER_NAMES[@]}"; do
        if ! grep -aFq -- "${layer_name}" "${ONNX}"; then
            echo "[ERROR] hybrid_v1 layer is absent from ONNX: ${layer_name}" >&2
            missing_count=$((missing_count + 1))
        fi
    done

    if [ "${missing_count}" -ne 0 ]; then
        echo "[ERROR] hybrid_v1 validation failed; missing layers=${missing_count}" >&2
        return 1
    fi
}

case "${VALIDATE_LAYER_NAMES}" in
0)
    ;;
1)
    validate_preset_layer_names
    ;;
*)
    echo "[ERROR] VALIDATE_LAYER_NAMES must be 0 or 1" >&2
    exit 1
    ;;
esac

echo "[INFO] precision preset: hybrid_v1"
echo "[INFO] constrained FP16 layers: ${FP16_LAYER_COUNT}"
echo "[INFO] constrained INT8 layers: ${INT8_LAYER_COUNT}"
echo "[INFO] output engine: ${ENGINE}"

case "${PRINT_ONLY}" in
0)
    ;;
1)
    echo "[INFO] layer precisions: ${LAYER_PRECISIONS}"
    exit 0
    ;;
*)
    echo "[ERROR] PRINT_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac

if [ ! -f "${BASE_BUILDER}" ]; then
    echo "[ERROR] base builder not found: ${BASE_BUILDER}" >&2
    exit 1
fi

export ONNX
export ENGINE
export CALIB_CACHE
export WORKSPACE_MIB
export LAYER_PRECISIONS
export LAYER_OUTPUT_TYPES

exec bash "${BASE_BUILDER}"
