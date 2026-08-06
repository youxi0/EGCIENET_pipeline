cmake_minimum_required(VERSION 3.16)

project(egcinet_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EGCINET_BUILD_PIPELINE "Build the runtime inspection pipeline targets" OFF)
option(EGCINET_BUILD_INT8_CALIBRATOR "Build the INT8 calibration cache tool" ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
set(_TENSORRT_ROOT_HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    $ENV{TENSORRT_DIR}
    /usr
    /usr/local
)

set(_TENSORRT_INCLUDE_SUFFIXES
    include
    include/x86_64-linux-gnu
    include/aarch64-linux-gnu
)
set(_TENSORRT_LIBRARY_SUFFIXES
    lib
    lib64
    lib/x86_64-linux-gnu
    lib/aarch64-linux-gnu
)
if(CMAKE_LIBRARY_ARCHITECTURE)
    list(APPEND _TENSORRT_INCLUDE_SUFFIXES
        include/${CMAKE_LIBRARY_ARCHITECTURE}
    )
    list(APPEND _TENSORRT_LIBRARY_SUFFIXES
        lib/${CMAKE_LIBRARY_ARCHITECTURE}
    )
endif()
list(REMOVE_DUPLICATES _TENSORRT_INCLUDE_SUFFIXES)
list(REMOVE_DUPLICATES _TENSORRT_LIBRARY_SUFFIXES)

# Jetson 的 Debian 包把头文件和动态库安装在 multiarch 系统目录中。
set(_TENSORRT_SYSTEM_INCLUDE_PATHS
    /usr/include
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
)
set(_TENSORRT_SYSTEM_LIBRARY_PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TENSORRT_ROOT_HINTS}
    PATHS ${_TENSORRT_SYSTEM_INCLUDE_PATHS}
    PATH_SUFFIXES ${_TENSORRT_INCLUDE_SUFFIXES}
)

find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ${_TENSORRT_ROOT_HINTS}
    PATHS ${_TENSORRT_SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES ${_TENSORRT_LIBRARY_SUFFIXES}
)

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS ${_TENSORRT_ROOT_HINTS}
    PATHS ${_TENSORRT_SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES ${_TENSORRT_LIBRARY_SUFFIXES}
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
    message(FATAL_ERROR
        "TensorRT was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT "
        "or install the Jetson libnvinfer-dev package."
    )
endif()

set(_TENSORRT_VERSION_HEADER "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${_TENSORRT_VERSION_HEADER}")
    message(FATAL_ERROR
        "NvInferVersion.h was not found under ${TENSORRT_INCLUDE_DIR}."
    )
endif()

file(READ "${_TENSORRT_VERSION_HEADER}" _TENSORRT_VERSION_CONTENT)
foreach(_TENSORRT_VERSION_PART MAJOR MINOR PATCH BUILD)
    string(
        REGEX MATCH
        "#define[ \t]+NV_TENSORRT_${_TENSORRT_VERSION_PART}[ \t]+([0-9]+)"
        _TENSORRT_VERSION_MATCH
        "${_TENSORRT_VERSION_CONTENT}"
    )
    if(NOT _TENSORRT_VERSION_MATCH)
        string(
            REGEX MATCH
            "#define[ \t]+TRT_${_TENSORRT_VERSION_PART}_ENTERPRISE[ \t]+([0-9]+)"
            _TENSORRT_VERSION_MATCH
            "${_TENSORRT_VERSION_CONTENT}"
        )
    endif()
    if(NOT _TENSORRT_VERSION_MATCH)
        message(FATAL_ERROR
            "Failed to parse NV_TENSORRT_${_TENSORRT_VERSION_PART} "
            "from ${_TENSORRT_VERSION_HEADER}."
        )
    endif()
    set(
        _TENSORRT_VERSION_${_TENSORRT_VERSION_PART}
        "${CMAKE_MATCH_1}"
    )
endforeach()

if(_TENSORRT_VERSION_MAJOR LESS 10)
    message(FATAL_ERROR
        "TensorRT 10 or newer is required; found "
        "${_TENSORRT_VERSION_MAJOR}.${_TENSORRT_VERSION_MINOR}."
    )
endif()

set(TENSORRT_VERSION
    "${_TENSORRT_VERSION_MAJOR}.${_TENSORRT_VERSION_MINOR}.${_TENSORRT_VERSION_PATCH}.${_TENSORRT_VERSION_BUILD}"
)

message(STATUS "TensorRT version: ${TENSORRT_VERSION}")
message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer library: ${TENSORRT_NVINFER_LIB}")
if(TENSORRT_NVONNXPARSER_LIB)
    message(STATUS "TensorRT ONNX parser library: ${TENSORRT_NVONNXPARSER_LIB}")
endif()

if(EGCINET_BUILD_INT8_CALIBRATOR)
    if(NOT TENSORRT_NVONNXPARSER_LIB)
        message(FATAL_ERROR
            "TensorRT ONNX parser was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT "
            "or disable EGCINET_BUILD_INT8_CALIBRATOR."
        )
    endif()

    add_executable(egcinet_calibrate_int8
        src/infer/Int8Calibrator.cpp
        src/infer/TensorRTCommon.cpp
        src/tools/calibrate_main.cpp
        src/utils/FileLogger.cpp
    )

    target_include_directories(egcinet_calibrate_int8 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(egcinet_calibrate_int8 PRIVATE
        ${OpenCV_LIBS}
        ${TENSORRT_NVINFER_LIB}
        ${TENSORRT_NVONNXPARSER_LIB}
        CUDA::cudart
        Threads::Threads
    )

    set_target_properties(egcinet_calibrate_int8 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

if(EGCINET_BUILD_PIPELINE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(egcinet_runtime STATIC
        src/acquisition/ImageFolderSource.cpp
        src/acquisition/VideoSource.cpp
        src/common/Timer.cpp
        src/infer/TensorRTCommon.cpp
        src/infer/TensorRTInfer.cpp
        src/network/TcpFrameSender.cpp
        src/pipeline/InspectionPipeline.cpp
        src/postprocess/SegPostprocessor.cpp
        src/postprocess/SegPostprocessKernel.cu
        src/preprocess/CudaPreprocessor.cpp
        src/preprocess/CudaPreprocessKernel.cu
        src/utils/FileLogger.cpp
        src/visualize/Visualizer.cpp
        src/visualize/VisualizeKernel.cu
    )

    target_include_directories(egcinet_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(egcinet_runtime PUBLIC
        ${OpenCV_LIBS}
        ${TENSORRT_NVINFER_LIB}
        CUDA::cudart
        Threads::Threads
    )

    set_target_properties(egcinet_runtime PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    add_library(egcinet_validation STATIC
        src/evaluation/EngineValidator.cpp
    )

    target_include_directories(egcinet_validation PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(egcinet_validation PUBLIC
        egcinet_runtime
    )

    set_target_properties(egcinet_validation PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    add_executable(egcinet_infer_image
        src/tools/infer_image.cpp
    )

    target_link_libraries(egcinet_infer_image PRIVATE
        egcinet_runtime
    )

    set_target_properties(egcinet_infer_image PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(egcinet_pipeline
        src/main.cpp
    )

    target_link_libraries(egcinet_pipeline PRIVATE
        egcinet_runtime
    )

    set_target_properties(egcinet_pipeline PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(egcinet_validate
        src/tools/validate_engines.cpp
    )

    target_link_libraries(egcinet_validate PRIVATE
        egcinet_validation
    )

    set_target_properties(egcinet_validate PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
