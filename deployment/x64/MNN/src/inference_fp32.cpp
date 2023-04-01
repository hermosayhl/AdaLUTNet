/*****************************************************************************
*                                       *
*  @file     inference_fp32.cpp                                              *
*  @brief    使用MNN推理引擎对含有自定义算子的模型推理                           *
*  Details   MNN 2.4.0; 目前支持 float32 和 动态输入                           *
*  @author   刘畅                                                            *
*  @email    fluenceyhl@163.com                                              *
*  @version  0.0.1                                                           *
*----------------------------------------------------------------------------*
*  Remark         : MNN; 自定义算子                                           *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2023/04/01 | 0.0.1     | liuchang       | 跑通了基本流程                   *
*----------------------------------------------------------------------------*
*****************************************************************************/
// C++
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <functional>
#include <filesystem>
// 读取图像的库
#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#elif defined(USE_STB_IMAGE)
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"
#endif
// MNN
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>


/** 
 * @brief  把数据截断在 0-255 之间, 方便转存为 uint8
 * @param  x        参数 1    通常为 float 数据
 *
 * @return 
 *     	   返回 0-255 之间的值
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
private:
	MNN::Interpreter*   network = nullptr; /*!< 模型部分, 跟 ONNX 一致, 推理的主要接口 */
	std::vector<float>  luts;              /*!< 模型部分, 包含 ONNX 在外, 推理的主要接口 */

	MNN::ScheduleConfig config;            /*!< MNN 推理配置 */
    MNN::BackendConfig  backendConfig;	   /*!< MNN 推理精度、内存、耗电量等设置 */
    MNN::Session*       session; 		   /*!< MNN 推理会话 */
 
	bool                load_ok = false;   /*!< 标记模型是否已经成功加载 */
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
	 * @param  model_path        参数 1    MNN 模型的路径, 与 ONNX 对应
	 * @param  luts_path         参数 2    自定义算子所需变量的路径, 在 ONNX 之外
	 */
	explicit AdaLUTNetMNNInference(const std::string& model_path, const std::string& luts_path) {
		if (not std::filesystem::exists(model_path) or not std::filesystem::exists(luts_path)) {
			printf("Cannot find %s at %d\n", model_path.c_str(), __LINE__);
		} else {
			this->load_ok = this->load_model(model_path) and this->load_luts(luts_path);
		}
	}

	/** 
	 * @brief  加载 MNN 模型, 配置推理参数, 创建会话
	 * @param  model_path        参数 1    MNN 模型的路径, 与 ONNX 对应
	 *
	 * @return 如果加载成功, 返回 true; 否则返回 false
	 */
	bool load_model(const std::string& model_path) {
		this->network = MNN::Interpreter::createFromFile(model_path.c_str());
		if (this->network == nullptr) {
			printf("Cannot read %s at %d\n", model_path.c_str(), __LINE__);
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
	 * @param  luts_path        参数 1    自定义算子所需变量的路径, 在 ONNX 之外
	 *
	 * @return 如果加载成功, 返回 true; 否则返回 false
	 */
	bool load_luts(const std::string& luts_path) {
		// 以二进制读取
		std::ifstream luts_reader(luts_path, std::ios::in | std::ios::binary);
		// 如果文件打开失败
		if (not luts_reader) {
			printf("Cannot open lut file: %s at %d\n", luts_path.c_str(), __LINE__);
			luts_reader.close();
			return false;
		}
		// 获取文件长度
		luts_reader.seekg(0, std::ios::end);
		const int32_t lut_length = static_cast<int32_t>(luts_reader.tellg());
		if (lut_length == 0) {
			printf("Lut file %s is empty!\n", luts_path.c_str());
			luts_reader.close();
			return false;
		}
		// lut 表目前是 float 的, 4 bytes
		const int32_t lut_elements = lut_length / sizeof(float);
		this->luts.resize(lut_elements);
		luts_reader.seekg(0, std::ios::beg);
		luts_reader.read((char*)(this->luts.data()), lut_length);
		luts_reader.close();
		return true;
	}

	/** 
	 * @brief  推理过程
	 * @param  output_raw_ptr        参数 1    用于保存模型增强结果的指针, 类型 uint8, 数据范围 0-255
	 * @param  input_raw_ptr         参数 2    输入图像的指针, 类型 uint8, 数据范围 0-255
	 * @param  height                参数 3    图像高度
	 * @param  width                 参数 3    图像宽度
	 * @param  channel               参数 3    图像通道数目
	 *
	 * @return 
	 */
	void run(
			unsigned char* const output_raw_ptr,
			const unsigned char* input_raw_ptr, 
			const int32_t height, const int32_t width, const int32_t channel) {
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
		run_timer([&](){
			const int32_t infer_hw = height * width;
			for (int32_t c = 0; c < channel; ++c) {
				float* write_ptr = input_buffer->host<float>() + c * infer_hw;
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

		const int area = height * width;
		// openmp
		run_timer([&, this](){
			for (int i = 0; i < area; ++i) {
				unsigned char* write_ptr = output_raw_ptr + i * channel;
				for (int c = 0; c < channel; ++c) {
					float* weight_start = weight_ptr + i * 27 + c * 9;
					float* luts_start   = this->luts.data() + c * 9 * 256;
					float sum_value{0.f};
					int pixel_value = input_raw_ptr[i * 3 + c];
					for (int k = 0; k < 9; ++k) {
						sum_value += weight_start[k] * luts_start[pixel_value];
						luts_start += 256;
					}
					write_ptr[c] = cv_clip(sum_value * 255);
				}
			}
		}, "post process");
		
		// 释放资源
		delete output_buffer;
	}
};



int main() {
	// 查看工作目录, 是否符合预期
	std::cout << std::filesystem::current_path().string() << std::endl;

	// 模型文件
	const std::string model_path("./Ada1DLUT_dynamic.mnn");
	const std::string luts_path("../../IR/learned_lut.bin");
	
	// 创建推理的接口
	AdaLUTNetMNNInference task(model_path, luts_path);

    // 读取图像
	const std::string image_path("./images/IMG_20230315_124509.png");

#if defined(USE_OPENCV)
	printf("USE_OPENCV\n");
	cv::Mat input_image           = cv::imread(image_path);
	if (input_image.empty()) {
		printf("Cannot read image %s\n", image_path.c_str());
		return 0;
	}
	// 获取图像信息
	const int32_t height = input_image.rows;
	const int32_t width  = input_image.cols;
	const int32_t channel = input_image.channels();
	// 准备一个同样大小的图像, 接收增强结果
	cv::Mat enhanced(height, width, input_image.type());
	unsigned char* input_image_ptr = input_image.ptr<unsigned char>();
	unsigned char* output_image_ptr = enhanced.ptr<unsigned char>();

#elif defined(USE_STB_IMAGE)
	printf("USE_STB_IMAGE\n");
	int32_t height, width, channel;
	unsigned char* input_image_ptr = stbi_load(image_path.c_str(), &width, &height, &channel, 0);
	// 分配一个同样大小的图像, 接收增强结果
	unsigned char* output_image_ptr = (unsigned char*)std::malloc(height * width * channel * sizeof(unsigned char));

#else
	printf("No libraries to load images!\n");
	return 0;
#endif
	printf("image: %d X %d\n", height, width);
	
	// 推理
	run_timer([&](){
		task.run(output_image_ptr, input_image_ptr, height, width, channel);
	}, "total");

#if defined(USE_OPENCV)
	// 展示和保存
	cv::imshow("enhanced", enhanced);
	cv::waitKey(0);
	cv::destroyAllWindows();
	cv::imwrite("./images/fp32_enhanced_opencv.png", enhanced, {cv::IMWRITE_PNG_COMPRESSION, 0});

#elif defined(USE_STB_IMAGE)
	// 保存并释放资源
	stbi_write_png("./images/fp32_enhanced_stbimage.png", width, height, channel, output_image_ptr, 0);
	stbi_image_free(input_image_ptr);
	std::free(output_image_ptr);

#endif

	printf("\nEND!\n");
	return 0;
}