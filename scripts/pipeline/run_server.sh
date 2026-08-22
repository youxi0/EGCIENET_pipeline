#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${PROJECT_ROOT}/scripts/common/tensorrt_env.sh"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
EXECUTABLE="${BUILD_DIR}/bin/egcinet_pipeline"

ENGINE="${ENGINE:-${PROJECT_ROOT}/models/egcienet_352_multiclass_fp16.engine}"
TYPE="${TYPE:-folder}"
if [ "${TYPE}" = "camera" ]; then
    SOURCE="${SOURCE:-0}"
else
    SOURCE="${SOURCE:-${PROJECT_ROOT}/datasets/images/val}"
fi
QUEUE_SIZE="${QUEUE_SIZE:-3}"
MAX_WIDTH="${MAX_WIDTH:-1920}"
MAX_HEIGHT="${MAX_HEIGHT:-1080}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1920}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1080}"
CAMERA_FPS="${CAMERA_FPS:-30}"
SAVE_DIR="${SAVE_DIR:-}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/logs}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"
TCP_HOST="${TCP_HOST:-}"
TCP_PORT="${TCP_PORT:-9000}"
TCP_QUEUE="${TCP_QUEUE:-2}"
JPEG_QUALITY="${JPEG_QUALITY:-85}"

if [ ! -f "${EXECUTABLE}" ]; then
    echo "[ERROR] executable not found: ${EXECUTABLE}"
    echo "[HINT] run ./scripts/pipeline/build.sh first"
    exit 1
fi

if [ ! -f "${ENGINE}" ]; then
    echo "[ERROR] engine not found: ${ENGINE}"
    exit 1
fi

case "${TYPE}" in
    camera)
        if [[ ! "${SOURCE}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] camera source must be a non-negative camera ID, got: ${SOURCE}" >&2
            echo "[HINT] use TYPE=camera SOURCE=0 for /dev/video0" >&2
            exit 2
        fi
        ;;
    folder|video)
        if [ ! -e "${SOURCE}" ]; then
            echo "[ERROR] source not found: ${SOURCE}" >&2
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] TYPE must be folder, video or camera, got: ${TYPE}" >&2
        exit 2
        ;;
esac

configure_tensorrt_library_path

if [ -z "${EGCINET_TRT_PLUGIN_LIBS:-}" ]; then
    AUTO_PLUGIN_LIBS=()
    for plugin_library in \
        "${BUILD_DIR}/lib/libegcinet_block1_fused_plugin.so" \
        "${BUILD_DIR}/lib/libegcinet_packed_dwconv_plugin.so"
    do
        if [ -f "${plugin_library}" ]; then
            AUTO_PLUGIN_LIBS+=("${plugin_library}")
        fi
    done

    if [ "${#AUTO_PLUGIN_LIBS[@]}" -gt 0 ]; then
        printf -v EGCINET_TRT_PLUGIN_LIBS '%s:' "${AUTO_PLUGIN_LIBS[@]}"
        EGCINET_TRT_PLUGIN_LIBS="${EGCINET_TRT_PLUGIN_LIBS%:}"
        export EGCINET_TRT_PLUGIN_LIBS
    fi
fi

echo "[INFO] start EGCINET pipeline"
echo "[INFO] engine    : ${ENGINE}"
echo "[INFO] source    : ${SOURCE}"
echo "[INFO] type      : ${TYPE}"
echo "[INFO] queue size: ${QUEUE_SIZE}"
echo "[INFO] max source: ${MAX_WIDTH}x${MAX_HEIGHT}"
if [ "${TYPE}" = "camera" ]; then
    echo "[INFO] camera    : ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${CAMERA_FPS} FPS"
fi
echo "[INFO] log dir   : ${LOG_DIR}"
echo "[INFO] log every : ${LOG_INTERVAL} frame(s); 0 disables frame timing logs"
if [ -n "${EGCINET_TRT_PLUGIN_LIBS:-}" ]; then
    echo "[INFO] plugins   : ${EGCINET_TRT_PLUGIN_LIBS}"
else
    echo "[INFO] plugins   : none"
fi
if [ -n "${TCP_HOST}" ]; then
    echo "[INFO] tcp target: ${TCP_HOST}:${TCP_PORT}"
    echo "[INFO] tcp queue : ${TCP_QUEUE}"
    echo "[INFO] jpeg      : quality ${JPEG_QUALITY}"
else
    echo "[INFO] tcp       : disabled"
fi

ARGS=(
    --engine "${ENGINE}"
    --source "${SOURCE}"
    --type "${TYPE}"
    --queue_size "${QUEUE_SIZE}"
    --max_width "${MAX_WIDTH}"
    --max_height "${MAX_HEIGHT}"
    --camera_width "${CAMERA_WIDTH}"
    --camera_height "${CAMERA_HEIGHT}"
    --camera_fps "${CAMERA_FPS}"
    --log_dir "${LOG_DIR}"
    --log_interval "${LOG_INTERVAL}"
)

if [ -n "${TCP_HOST}" ]; then
    ARGS+=(
        --tcp_host "${TCP_HOST}"
        --tcp_port "${TCP_PORT}"
        --tcp_queue "${TCP_QUEUE}"
        --jpeg_quality "${JPEG_QUALITY}"
    )
fi

if [ -n "${SAVE_DIR}" ]; then
    ARGS+=(--save_dir "${SAVE_DIR}")
fi

"${EXECUTABLE}" "${ARGS[@]}"
