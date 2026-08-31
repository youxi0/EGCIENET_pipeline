#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export ONNX="${ONNX:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v28_q_all_kv_all_packed_sr_block13_int8_block2_native_conv.onnx}"
export ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_qdq_v28_q_all_kv_all_packed_sr_block13_int8_block2_native_conv.engine}"
export LAYER_INFO="${LAYER_INFO:-${ENGINE%.engine}_layers.json}"
export BUILD_LABEL="${BUILD_LABEL:-V28 Block2 native INT8 spatial-reduction Conv}"

exec bash "${PROJECT_ROOT}/scripts/engine/build_v26_all_int8_sr_engine.sh"
