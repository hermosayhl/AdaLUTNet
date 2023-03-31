// C & C++
#include <assert.h>
#include <cstdio>
#include <iostream>
// CUDA
#include <cuda_runtime.h>
// Torch
#include <cuda.h>
#include <ATen/ATen.h>
#include <torch/torch.h>


template<const int32_t BLOCK_XY> 
__global__ void adaptive_1dlut_intensity_transform_forward_kernel(
	float* const output,
	const float* const image,
	const float* const weights,
	const float* const luts,
	int32_t* const lut_index,
	const int32_t C,
	const int32_t HW,
	const int32_t luts_count,
	const int32_t luts_nodes) {
	// 计算当前线程在某一通道图像 HW 中的逻辑下标 (16x16 + blockIdx.y * block)
	const int32_t idx = threadIdx.x + blockIdx.y * BLOCK_XY;
	if (idx < HW) {
		// 获取当前线程所在 batch
		const int32_t batch_id = blockIdx.x / C;
		// 获取当前线程所在通道
		const int32_t C_id     = blockIdx.x % C;
		// 1 x 3 x 9 x H x W, 现在开始是 9xHxW
		const float* weights_ptr = weights + (batch_id * C + C_id) * luts_count * HW;
		// 3 x 9 x 256, 现在开始是 9x256
		const float* luts_ptr = luts + C_id * luts_count * luts_nodes;
		// 取出图像的值, 原来范围是 0-1, 不能当作索引, 因此要 *255
		const int32_t image_pos = (batch_id * C + C_id) * HW + idx;
		const int32_t pixel_value = luts_nodes * image[image_pos];
		// 开始算 9 次的加权
		float sum_value = 0.f;
		for (int32_t i = 0; i < luts_count; ++i) {
			sum_value   += luts_ptr[pixel_value] * weights_ptr[idx];
			weights_ptr += HW;
			luts_ptr    += luts_nodes;
		}
		// 写回
		output[image_pos] = sum_value;
		// 为反向传播记录一些信息
		lut_index[image_pos] = pixel_value;
	}
}





namespace {
	inline int32_t CUDA_CEIL(const int32_t x, const int32_t y) {
		return (x + y - 1) / y;
	}
}



void adaptive_1dlut_intensity_transform_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& weights,
		const at::Tensor& luts,
		at::Tensor& lut_index) {
	
	// 获取待处理图像的维度
	const int32_t B = image.size(0);
	const int32_t C = image.size(1);
	const int32_t H = image.size(2);
	const int32_t W = image.size(3);
	// 获取曲线加权的维度
	const int32_t weights_B = weights.size(0);
	const int32_t weights_C = weights.size(1);
	const int32_t weights_H = weights.size(2);
	const int32_t weights_W = weights.size(3);
	// 获取 1DLUT 的维度
	const int32_t luts_C     = luts.size(0);
	const int32_t luts_count = luts.size(1);
	const int32_t luts_nodes = luts.size(2);
	// 打印
	// printf("image  : [%d %d %d %d]\n", B, C, H, W);
	// printf("weights: [%d %d %d %d]\n", weights_B, weights_C, weights_H, weights_W);
	// printf("luts   : [%d %d %d]\n", luts_C, luts_count, luts_nodes);

	// 决定 GPU 逻辑布局
	constexpr int32_t BLOCK_X = 16;
	constexpr int32_t BLOCK_Y = 16;
	dim3 block(BLOCK_X * BLOCK_Y);
	dim3 grid(
		B * C,  /* 1 * 3 */
		CUDA_CEIL(H * W, BLOCK_X * BLOCK_Y) /* 512 * 341 / 256 */
	);

	// 启动 kernel
	adaptive_1dlut_intensity_transform_forward_kernel<BLOCK_X * BLOCK_Y><<<grid, block>>>(
		output.data_ptr<float>(),
		image.data_ptr<float>(),
		weights.data_ptr<float>(),
		luts.data_ptr<float>(),
		lut_index.data_ptr<int32_t>(),
		C,
		H * W,
		luts_count,
		luts_nodes
	);
}

