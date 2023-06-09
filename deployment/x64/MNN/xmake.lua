-- 设置工程名
set_project("adalutnet_inference")
-- 设置工程版本
set_version("0.0.1")
-- 设置 xmake 版本
set_xmakever("2.4.0")
-- 设置支持的平台
set_allowedplats("windows", "mingw", "linux", "other")


-- 添加 opencv 支持
function add_opencv_support()
    -- 添加宏定义
    add_defines("USE_OPENCV")
    -- 找到 opencv 目录和版本
    opencv_root    = ""
    opencv_version = ""
    -- 如果是 windows 平台
    if is_plat("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.7.0-win/build"
        opencv_version = "470"
        add_linkdirs(opencv_root .. "/x64/vc16/lib")
        add_includedirs(opencv_root .. "/include")
        print("windows using MSVC")
    else
        -- 如果是 linux 平台
        if is_plat("linux") then
            opencv_root    = "/home/dx/usrs/liuchang/tools/opencv/build/install"
            opencv_version = "452"
            add_includedirs(opencv_root .. "/include/opencv4")
            add_linkdirs(opencv_root .. "/lib")
            print("linux using GCC")
        end
        -- 如果是 MinGW 平台
        if is_plat("mingw") then
            opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install"
            opencv_version = "455"
            add_linkdirs(opencv_root .. "/x64/mingw/bin")
            add_includedirs(opencv_root .. "/include")
            print("link OpenCV in windows via MinGW")
        end
    end
        
    -- 这里要再链接一次的原因是, cnn_layers 动态库中只有 OpenCV 动态库的入口, 不是完整的拷贝
    if is_plat("windows") then
        add_links("opencv_world" .. opencv_version)
    else
        add_links(
            "libopencv_core" .. opencv_version, 
            "libopencv_highgui" .. opencv_version, 
            "libopencv_imgproc" .. opencv_version,  
            "libopencv_imgcodecs" .. opencv_version
        )
    end
end


-- 添加 stb_image 支持
function add_stb_support()
    -- 添加宏定义
    add_defines("USE_STB_IMAGE")
    add_defines("STB_IMAGE_IMPLEMENTATION")
    add_defines("STB_IMAGE_WRITE_IMPLEMENTATION")
    add_defines("STB_IMAGE_RESIZE_IMPLEMENTATION")
    -- 添加头文件
    add_includedirs("$(projectdir)/include/stb_image")
end


-- 添加 MNN 支持
function add_mnn_support()
    if is_plat("windows") then
        print("empty!")
    else
        -- 如果是 linux 平台
        if is_plat("linux") then
            print("linux using GCC but empty now")
        end
        -- 如果是 MinGW 平台
        if is_plat("mingw") then
            mnn_root = "E:/environments/C++/mnn/MNN/install_gcc"
            add_includedirs(mnn_root .. "/include")
            add_linkdirs(mnn_root .. "/lib")
            add_links("MNN")
            print("link MNN in windows via MinGW")
        end
    end
end



-- 默认设置
function use_default_config()
    -- 设置生成类型, 可执行文件
    set_kind("binary")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 设置目标工作目录
    set_rundir("$(projectdir)")
end



-- 处理动态输入的 demo
target("dynamic_inference")

    -- 执行相同的操作
    use_default_config()

    -- 添加源文件, 含有 main 函数入口
    add_files("$(projectdir)/src/inference_fp32.cpp")

    -- 添加自己的头文件
    add_includedirs("$(projectdir)/include")
    
    -- 添加 OpenCV 支持
    -- add_opencv_support()

    -- 添加 stb_image 支持(要注意这个 RGB 顺序还是 BGR 顺序影响)
    add_stb_support()

    -- 添加 MNN 支持
    add_mnn_support()
    
-- 结束 cnn_train
target_end()



-- 处理动态输入的 demo
target("dynamic_inference_buffer")

    -- 执行相同的操作
    use_default_config()

    -- 添加源文件, 含有 main 函数入口
    add_files("$(projectdir)/src/inference_fp32_buffer.cpp")

    -- 添加自己的头文件
    add_includedirs("$(projectdir)/include")
    
    -- 添加 OpenCV 支持
    add_opencv_support()

    -- 添加 stb_image 支持(要注意这个 RGB 顺序还是 BGR 顺序影响)
    -- add_stb_support()

    -- 添加 MNN 支持
    add_mnn_support()
    
-- 结束 cnn_train
target_end()