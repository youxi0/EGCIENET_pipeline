#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"

echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] build dir: ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ..
cmake --build . -j10

echo "[INFO] build finished"
echo "[INFO] executable: ${BUILD_DIR}/bin/blade_demo"
echo "[INFO] executable: ${BUILD_DIR}/bin/blade_build_engine"
