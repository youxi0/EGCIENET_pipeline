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

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
    message(FATAL_ERROR
        "TensorRT was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT "
        "or export TENSORRT_ROOT/TENSORRT_DIR."
    )
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
        src/tools/calibrate_int8.cpp
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
endif()
