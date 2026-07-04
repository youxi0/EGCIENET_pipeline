cmake_minimum_required(VERSION 3.16)

project(blade_inspection_system LANGUAGES CXX CUDA)

#模块:设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#模块:控制blade_core编译成静态库还是动态库
#OFF表示静态库,ON表示动态库
option(BUILD_BLADE_SHARED "Build blade core as shared library" OFF)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)


#模块:查找TensorRT推理库nvinfer
#find_library会在指定目录下寻找libnvinfer.so或libnvinfer.a
set(TENSORRT_DIR "$ENV{HOME}/haiyanghuang/TensorRT-8.6.1.6")
find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    PATHS ${TENSORRT_DIR}/lib
    REQUIRED
)

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    PATHS ${TENSORRT_DIR}/lib
    REQUIRED
)

#将找到文件以“列表（List）”的形式存入BLADE_CORE_SOURCES中
#CONFIGURE_DEPENDS表示当新增或删除cpp文件时,CMake会自动重新配置
file(GLOB_RECURSE BLADE_CORE_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)


list(REMOVE_ITEM BLADE_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/build_engine.cpp
)

#模块:根据BUILD_BLADE_SHARED选择生成动态库或静态库
if(BUILD_BLADE_SHARED)
    add_library(blade_core SHARED ${BLADE_CORE_SOURCES})
else()
    add_library(blade_core STATIC ${BLADE_CORE_SOURCES})
endif()

#模块:给blade_core设置头文件搜索路径
#PUBLIC表示blade_core自己需要这些头文件,链接blade_core的目标也能继承这些路径
target_include_directories(blade_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
)

#模块:给blade_core链接依赖库
target_link_libraries(blade_core PUBLIC
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    CUDA::cudart
    Threads::Threads
)

set_target_properties(blade_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(blade_demo
    src/main.cpp
)

#主程序链接核心库
target_link_libraries(blade_demo PRIVATE
    blade_core
)

add_executable(blade_build_engine
    src/tools/build_engine.cpp
)

target_link_libraries(blade_build_engine PRIVATE
    blade_core
)

#设置可执行文件输出目录
set_target_properties(blade_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(blade_build_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

#设置库文件输出目录
set_target_properties(blade_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
