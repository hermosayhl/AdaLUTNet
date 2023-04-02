#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#define TAG "Ada1DLUTSDK"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// C++
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <functional>

#ifdef Ada1DLUTInference_USING_OPENCV
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#endif

// MNN
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

// 二进制字符串, 免去 asset 找不到的问题
// 生成字符串: linux 下 xxd -i CurveNet.mnn >curve_net.mem.h
// 不过这种方式的编译速度和调试速度慢的很, 酌情考虑, 或者把模型逻辑简化
#include "ada1dlut_dynamic.mem.h"
#include "ada1dlut_learned_lut.mem.h"




/** 
 * @brief  把数据截断在 0-255 之间, 方便转存为 uint8
 * @param  x        参数 1    通常为 float 数据
 *
 * @return 
 *         返回 0-255 之间的值
 */
inline unsigned char cv_clip(const float x) {
    if (x < 0) return 0;
    else if (x > 255) return 255;
    return x;
}


/** 
 * @brief  测试 CPU 耗时的函数
 * @param  work        参数 1    函数闭包, 嵌入了一段可执行代码
 * @param  message     参数 2    该段执行代码的简要描述
 *
 * @return 
 */
void run_timer(const std::function<void()>& work=[]{}, const std::string message="") {
    auto start = std::chrono::steady_clock::now();
    work();
    auto finish = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << message << " " << duration.count() << " ms" <<  std::endl;
}


/**
 * @brief 
 * 将 MNN 推理的部分写成接口的形式, 处理多张图像更方便
 */
template<const int CHANNEL=3, const int LUT_NODES=256>
class AdaLUTNetMNNInference {
public:
    bool                load_ok = false;   /*!< 标记模型是否已经成功加载 */
private:
    MNN::Interpreter*   network = nullptr; /*!< 模型部分, 跟 ONNX 一致, 推理的主要接口 */
    float*              luts;              /*!< 模型部分, 包含 ONNX 在外, 推理的后处理(自定义算子部分所需变量) */
    int32_t             luts_count = 0;    /*!< 模型部分, lut的个数 */

    MNN::ScheduleConfig config;            /*!< MNN 推理配置 */
    MNN::BackendConfig  backendConfig;	   /*!< MNN 推理精度、内存、耗电量等设置 */
    MNN::Session*       session; 		   /*!< MNN 推理会话 */

public:
    /**
     * @brief  析构函数, 释放资源
     */
    ~AdaLUTNetMNNInference() noexcept {
        this->network->releaseModel();
        this->network->releaseSession(this->session);
        delete this->network;
        this->network = nullptr;
    }

    /**
     * @brief  构造函数, 加载模型、创建会话
     */
    explicit AdaLUTNetMNNInference() {
        this->load_ok = this->load_model() and this->load_luts();
    }

    /**
     * @brief  加载 MNN 模型, 配置推理参数, 创建会话
     *
     * @return 如果加载成功, 返回 true; 否则返回 false
     */
    bool load_model() {
        const size_t mnn_len = sizeof(Ada1DLUT_dynamic_mnn);
        this->network = MNN::Interpreter::createFromBuffer(Ada1DLUT_dynamic_mnn, mnn_len);
        if (this->network == nullptr) {
            printf("Cannot load model\n");
            return false;
        }
        this->config.numThread        = 4;
        const int32_t forward         = MNN_FORWARD_CPU;
        this->config.type             = static_cast<MNNForwardType>(forward);
        this->backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_High;
        this->backendConfig.power     = MNN::BackendConfig::PowerMode::Power_High;
        this->backendConfig.memory    = MNN::BackendConfig::MemoryMode::Memory_High;
        this->config.backendConfig    = &(this->backendConfig);
        this->session = this->network->createSession(config);
        return true;
    }

    /**
     * @brief  加载自定义算子所需的变量
     *
     * @return 如果加载成功, 返回 true; 否则返回 false
     */
    bool load_luts() {
        // 获取 lut 表的大小
        const size_t luts_len = sizeof(learned_lut_bin);
        // 计算 lut 表的元素个数, 默认 float32
        const size_t luts_elements = luts_len / sizeof(float);
        // 将 lut 表数据拷贝
        this->luts = (float*)learned_lut_bin;
        // 计算曲线的个数
        this->luts_count = static_cast<int32_t>(luts_elements / (CHANNEL * LUT_NODES));
        return true;
    }

    /**
     * @brief  推理过程
     * @param  output_raw_ptr        参数 1    用于保存模型增强结果的指针, 类型 uint8, 数据范围 0-255
     * @param  input_raw_ptr         参数 2    输入图像的指针, 类型 uint8, 数据范围 0-255
     * @param  height                参数 3    图像高度
     * @param  width                 参数 4    图像宽度
     * @param  channel               参数 5    图像通道数目
     * @param  channel               参数 6    是否使用 BGR 顺序
     *
     * @return
     */
    void run(
            unsigned char* const output_raw_ptr,
            unsigned char* const input_raw_ptr,
            const int32_t height, 
            const int32_t width, 
            const int32_t channel,
            const bool use_BGR) {
        // 目前只支持 3 通道 RGB 24 图像
        if (channel != CHANNEL) {
            printf("Only images of 'channel = 3' are supported!\n");
            return;
        }
        // 如果模型没有加载成功, 直接返回
        if (not this->load_ok) {
            printf("Please ensure the model is load completely!\n");
            return;
        }
        // 从网络中取出输入
        auto input_tensor = this->network->getSessionInput(this->session, "image");
        this->network->resizeTensor(input_tensor, {1, channel, height, width});
        this->network->resizeSession(session);
        // input_tensor->printShape();

        // 准备一块跟 input_tensor 同样 shape 的张量, 数据从 hwc 转换成 chw
        auto input_buffer = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);
        const int32_t infer_hw = height * width;
        run_timer([&](){
            for (int32_t c = 0; c < channel; ++c) {
                float* write_ptr = input_buffer->host<float>() + (use_BGR? c * infer_hw: (CHANNEL - 1 - c) * infer_hw);
                for (int32_t i  = 0; i < infer_hw; ++i) {
                    write_ptr[i] = input_raw_ptr[i * channel + c] / 255.f;
                }
            }
        }, "data pre process");

        // 把输入拷贝到网络中, 如果需要的话
        input_tensor->copyFromHostTensor(input_buffer);
        delete input_buffer;

        // 推理
        run_timer([&](){
            this->network->runSession(session);
        }, "run session");

        // 根据 weight 字段获取输出
        auto output_tensor = this->network->getSessionOutput(this->session, "weight");

        // host 上准备一块数据
        auto output_buffer = new MNN::Tensor(output_tensor, MNN::Tensor::CAFFE);
        // output_buffer->printShape();

        // 把数据拷贝到 host, 通常是 NC4HW4 转换到 NCHW 排布
        output_tensor->copyToHostTensor(output_buffer);

        float* weight_ptr = output_buffer->host<float>();

        // openmp
        run_timer([&, this](){
            for (int c = 0; c < CHANNEL; ++c) {
                for (int i = 0; i < infer_hw; ++i) {
                    const int pos = i * CHANNEL + c;
                    float* weight_start = weight_ptr + pos * this->luts_count;
                    // 3 * 256 * 9
                    float sum_value{0.f};
                    int pixel_value = input_raw_ptr[pos];
                    float* luts_start   = this->luts + c * LUT_NODES * this->luts_count + pixel_value * this->luts_count;
                    for (int k = 0; k < this->luts_count; ++k) {
                        sum_value += weight_start[k] * luts_start[k];
                    }
                    output_raw_ptr[pos] = cv_clip(sum_value * 255);
                }
            }
        }, "post process");

        // 释放资源
        delete output_buffer;
    }
};

// 声明一个全局变量, 从而获取最高的生命期
AdaLUTNetMNNInference<3, 256> ada1dlut_task;




extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "Ada1DLUT", "JNI_OnLoad");

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "Ada1DLUT", "JNI_OnUnload");
}



/**
* @brief  该函数对应 Java 中 Ada1DLUT 类的 Init 函数 'public native boolean Init(AssetManager mgr);'
* @param  env           参数 1      Jave native interface 接口环境
* @param  thisz         参数 2      对应 Java 中 class Ada1DLUT 对象
* @param  assetManager  参数 3      通常用于获取资源文件, 但本程序中都默认从内存 buffer 中加载, 暂时不需要这一步
* @param  _model_path   参数 4      推理模型的路径, 但本程序中都默认从内存 buffer 中加载, 暂时不需要这一步
*
* @return 是否初始化成功
*/
JNIEXPORT jboolean JNICALL Java_com_fluence2crane_ada1dlut_Ada1DLUT_Init(JNIEnv* env, jobject thiz, jobject assetManager, jstring _model_path)
{
    const char *model_path = env->GetStringUTFChars(_model_path, 0);
    if (model_path == NULL) {
        LOGD("model file is empty");
        return JNI_FALSE;
    }
    // 目前采用的不是从文件中读取, 而是直接从二进制中获取
    return ada1dlut_task.load_ok;
}


/**
* @brief  该函数对应 Java 中 Ada1DLUT 类的 Enhance 函数 'public native Bitmap Enhance(Bitmap bitmap, int style_type, boolean use_gpu);;'
* @param  env           参数 1      Jave native interface 接口环境
* @param  thisz         参数 2      对应 Java 中 class Ada1DLUT 对象
* @param  bitmap        参数 3      图像的位图
* @param  style_type    参数 4      增强类型, 参考自 ncnn_style_transfer 设置, 本程序中后续可能需要
* @param  use_gpu       参数 5      是否使用 GPU, 参考自 ncnn_style_transfer 设置, 本程序中后续可能需要
*
* @return 是否完成了增强过程
*/
JNIEXPORT jboolean JNICALL Java_com_fluence2crane_ada1dlut_Ada1DLUT_Enhance(JNIEnv* env, jobject thiz, jobject bitmap, jint style_type, jboolean use_gpu)
{
    // 选取一个模型进行推理, 待做

    // 如果确认使用 gpu, 但无法获取到 gpu, 返回 false, 待做

    // 获取图像信息
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    // 只接受 8 8 8 8 比特编码的图像数据
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    // 获取原始输入图像和交给网络推理的数据指针
    unsigned char* input_image_ptr = nullptr;
    unsigned char* origin_image_ptr = nullptr;

#ifdef Ada1DLUTInference_USING_OPENCV
    // 首先获取 bitmap 像素数据
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;
    void* bitmap_pixels;
    AndroidBitmap_lockPixels(env, bitmap, &bitmap_pixels);
    if (bitmap_pixels == nullptr)
        return JNI_FALSE;
    // 以 OpenCV 的形式管理数据
    cv::Mat input_image_buffer(info.height, info.width, CV_8UC4);
    cv::Mat temp(info.height, info.width, CV_8UC4, bitmap_pixels);
    // 拷贝到缓冲区
    temp.copyTo(input_image_buffer);

    // 解锁
    AndroidBitmap_unlockPixels(env, bitmap);

    // uint8 → float32
    cv::cvtColor(input_image_buffer, input_image_buffer, cv::COLOR_RGBA2BGR);
    origin_image_ptr = input_image_buffer.ptr<unsigned char>();

    // 因为直接在原图上操作, 因此可以同一个图像指针(下面两行是更浪费的)
    // auto new_image = input_image_buffer.clone();
    // input_image_ptr = new_image.ptr<unsigned char>();
    input_image_ptr = input_image_buffer.ptr<unsigned char>();
#endif

    if (input_image_ptr == nullptr or origin_image_ptr == nullptr)
        return JNI_FALSE;

    ada1dlut_task.run(origin_image_ptr, input_image_ptr, info.height, info.width, 3, true);

#ifdef Ada1DLUTInference_USING_OPENCV
    // 使用 OpenCV, 把数据转移到 bitmap
    AndroidBitmap_lockPixels(env, bitmap, &bitmap_pixels);
    if (bitmap_pixels == nullptr)
        return JNI_FALSE;
    // 还原数据类型
    cv::cvtColor(input_image_buffer, input_image_buffer, cv::COLOR_BGR2RGBA);
    // 找一个中介完成内存数据的转换
    cv::Mat temp2(info.height, info.width, CV_8UC4, bitmap_pixels);
    input_image_buffer.copyTo(temp2);
    // 解锁
    AndroidBitmap_unlockPixels(env, bitmap);
#endif

    return JNI_TRUE;
}

}
