cmake_minimum_required(VERSION 3.16)

project(egcinet_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EGCINET_BUILD_PIPELINE "Build the runtime inspection pipeline targets" OFF)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT install root, for example /usr/src/tensorrt")
set(_TENSORRT_HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    $ENV{TENSORRT_DIR}
    /usr
    /usr/local
)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TENSORRT_HINTS}
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu
)

find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ${_TENSORRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS ${_TENSORRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
    message(FATAL_ERROR
        "TensorRT was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT "
        "or export TENSORRT_ROOT/TENSORRT_DIR."
    )
endif()

add_library(egcinet_engine_builder STATIC
    src/infer/EngineBuilder.cpp
    src/infer/Int8Calibrator.cpp
    src/infer/TensorRTCommon.cpp
    src/utils/FileLogger.cpp
)

target_include_directories(egcinet_engine_builder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(egcinet_engine_builder PUBLIC
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    CUDA::cudart
    Threads::Threads
)

add_executable(egcinet_build_engine
    src/tools/build_engine.cpp
)

target_link_libraries(egcinet_build_engine PRIVATE
    egcinet_engine_builder
)

set_target_properties(egcinet_build_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(egcinet_engine_builder PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

if(EGCINET_BUILD_PIPELINE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(egcinet_runtime STATIC
        src/acquisition/ImageFolderSource.cpp
        src/acquisition/VideoSource.cpp
        src/infer/TensorRTCommon.cpp
        src/infer/TensorRTInfer.cpp
        src/preprocess/CudaPreprocessor.cpp
        src/preprocess/CudaPreprocessKernel.cu
        src/utils/FileLogger.cpp
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
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
