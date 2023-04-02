project(Ada1DLUTInference)
cmake_minimum_required(VERSION 3.4.1)


# 手动添加 MNN
set(MNN_DIR ${CMAKE_SOURCE_DIR}/mnn/)
include_directories(${MNN_DIR}/include/)
set(MNN_LIBRARY_DIRS ${MNN_DIR}/libs/${ANDROID_ABI}/)
link_directories(${MNN_LIBRARY_DIRS})
set(MNN_LIBs
        ${MNN_LIBRARY_DIRS}/libc++_shared.so
        ${MNN_LIBRARY_DIRS}/libmnncore.so
        ${MNN_LIBRARY_DIRS}/libMNN.so
        )

# 设定用于图像数据交互的库
set(IMG_INTERACT_LIBs "")

# 自定义的选择 (Android Studio 下不生效)
option(Ada1DLUTInference_USING_OPENCV  "using opencv to interact!"  ON)
add_definitions(-DAda1DLUTInference_USING_OPENCV)

# 判断是否通过 ncnn 来实现 bitmap 图像和 C 数组的交互
if (DEFINED Ada1DLUTInference_USING_OPENCV)
    message(STATUS "using OpenCV to interact!")
    # 直接链接官方的会导致错误, 包括 opencv-mobile
    # OpenCV-tiny 动态库, 重新编译的迷你库
    set(OPENCV_TINY_PATH ${CMAKE_SOURCE_DIR}/opencvtiny/${ANDROID_ABI}/sdk/native)
    include_directories(${OPENCV_TINY_PATH}/jni/include)
    link_directories(${OPENCV_TINY_PATH}/libs/${ANDROID_ABI})
    set(IMG_INTERACT_LIBs ${IMG_INTERACT_LIBs} libopencv_core.so libopencv_highgui.so libopencv_imgproc.so)
endif()

# 把源代码编译成共享库
add_library(ada1dlut SHARED ada1dlut_net_jni.cpp)

# 链接
target_link_libraries(ada1dlut
        ${MNN_LIBs}
        jnigraphics
        ${IMG_INTERACT_LIBs}
        log
        )