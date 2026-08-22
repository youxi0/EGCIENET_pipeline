#!/bin/bash

# 返回当前 Linux 平台的 multiarch 名称。
tensorrt_multiarch() {
    if command -v dpkg-architecture >/dev/null 2>&1; then
        dpkg-architecture -qDEB_HOST_MULTIARCH
        return
    fi
    if command -v gcc >/dev/null 2>&1; then
        gcc -dumpmachine
        return
    fi

    case "$(uname -m)" in
        aarch64)
            echo "aarch64-linux-gnu"
            ;;
        x86_64)
            echo "x86_64-linux-gnu"
            ;;
        *)
            echo ""
            ;;
    esac
}

# 将自定义 TensorRT 或 Jetson 系统包的库目录加入运行时搜索路径。
configure_tensorrt_library_path() {
    local multiarch
    multiarch=$(tensorrt_multiarch)
    local candidates=()

    if [ -n "${TENSORRT_ROOT:-}" ]; then
        candidates+=(
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib64"
        )
        if [ -n "${multiarch}" ]; then
            candidates+=("${TENSORRT_ROOT}/lib/${multiarch}")
        fi
    fi

    if [ -n "${multiarch}" ]; then
        candidates+=("/usr/lib/${multiarch}")
    fi
    candidates+=(
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib/x86_64-linux-gnu"
    )

    local library_path="${LD_LIBRARY_PATH:-}"
    local directory
    for directory in "${candidates[@]}"; do
        if [ ! -d "${directory}" ]; then
            continue
        fi
        case ":${library_path}:" in
            *":${directory}:"*)
                ;;
            *)
                library_path="${directory}${library_path:+:${library_path}}"
                ;;
        esac
    done
    export LD_LIBRARY_PATH="${library_path}"
}

# 查找用户指定、自定义安装或 Jetson 系统安装中的 trtexec。
resolve_trtexec() {
    if [ -n "${TRTEXEC:-}" ]; then
        if [ -x "${TRTEXEC}" ]; then
            echo "${TRTEXEC}"
            return
        fi
        echo "[ERROR] configured trtexec is not executable: ${TRTEXEC}" >&2
        return 1
    fi

    if [ -n "${TENSORRT_ROOT:-}" ] &&
       [ -x "${TENSORRT_ROOT}/bin/trtexec" ]; then
        echo "${TENSORRT_ROOT}/bin/trtexec"
        return
    fi
    if command -v trtexec >/dev/null 2>&1; then
        command -v trtexec
        return
    fi
    if [ -x /usr/src/tensorrt/bin/trtexec ]; then
        echo "/usr/src/tensorrt/bin/trtexec"
        return
    fi

    echo "[ERROR] trtexec was not found; set TRTEXEC or TENSORRT_ROOT" >&2
    return 1
}
