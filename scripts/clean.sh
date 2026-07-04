#!/bin/bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"

echo "[INFO] remove build dir: ${BUILD_DIR}"
rm -r "${BUILD_DIR}"

echo "[INFO] clean finished"