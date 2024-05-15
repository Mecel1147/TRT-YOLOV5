cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(yolov5inferm VERSION 0.1.0 LANGUAGES C CXX)

# function(print_all_variables)
#     message(STATUS "Variables currently defined:")
#     get_cmake_property(_variableNames VARIABLES)
#     foreach (_variableName ${_variableNames})
#         message(STATUS "${_variableName}=${${_variableName}}")
#     endforeach()
# endfunction()

# set(CMAKE_BUILD_TYPE "Release")  # 编译类型
set(CMAKE_BUILD_TYPE Debug)

include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/print_all_variables.cmake)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)  # 输出文件夹路径



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# find required opencv
find_package(OpenCV REQUIRED)   # 使用find_package获取opencv的相关信息,如lib、include、version等
# print_all_variables()
message("opencv********"${OpenCV_LIBS})


set(glog_DIR /home/nano/Downloads/glog/build/)
find_package(glog REQUIRED)



# find_package(tensorrt REQUIRED)

# tensorrt
set(TENSORRT_INSIDE_PATH /usr/lib/aarch64-linux-gnu) # contaioner 中lib的目录
set(TENSORRT_ROOT_PATH /workspace/TensorRT/) # 根目录
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT_PATH}/include/)   # 头文件夹

# set(TENSORRT_LIB ${TENSORRT_ROOT_PATH}lib/libnvinfer.so ${TENSORRT_ROOT_PATH}lib/libnvonnxparser.so)  # 依赖动态库
set(TENSORRT_LIB ${TENSORRT_INSIDE_PATH}/libnvinfer.so ${TENSORRT_INSIDE_PATH}/libnvonnxparser.so)  # 依赖动态库


# cuda
set(CUDA_VERSION )
find_package(CUDA REQUIRED)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    /usr/lib/aarch64-linux-gnu
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

message("cuda ************************"${CUDA_LIB})

include_directories(
    /usr/src/tensorrt/samples/common
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)
message(${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARY})

set(CUDA_ROOT_PATH /usr/local/cuda) # 根目录
# 不能使用以下命令，不然找不到cuda_runtime_api.h
# set(CUDA_INCLUDE_DIRS ${CUDA_ROOT_PATH}/include/)   # 头文件夹

set(CUDA_LIB ${CUDA_ROOT_PATH}/lib64/libcudart.so)  # 依赖动态库

add_subdirectory(src)

# # directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ) # 头文件夹
# include_directories(${PROJECT_SOURCE_DIR}/include)
# # directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})    # 指定第三方库所在路径,该路径添加到第三方库搜索路径中
# # opencv libraries




