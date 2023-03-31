// C & C++
#include <assert.h>
#include <cstdio>
// CUDA
#include <cuda_runtime.h>
// Torch
#include <cuda.h>
#include <ATen/ATen.h>
#include <torch/torch.h>




template<const int32_t BLOCK_XY> 
__global__ void adaptive_1dlut_intensity_transform_backward_kernel(
	float* weights_grad,
	float* luts_grad,
	float* grad_from_output,
	float* weights,
	float* luts,
	int32_t* lut_index,
	const int32_t C,
	const int32_t HW,
	const int32_t luts_count,
	const int32_t luts_nodes) {
	// 计算当前线程在某一通道图像 HW 中的逻辑下标 (16x16 + blockIdx.y * block)
	const int32_t idx = threadIdx.x + blockIdx.y * BLOCK_XY;
	if (idx < HW) {
		// 获取当前线程所在 batch
		int32_t batch_id = blockIdx.x / C;
		// 获取当前线程所在通道
		int32_t C_id     = blockIdx.x % C;
		// 1 x 3 x 9 x H x W, 现在开始是 
		int32_t weight_offset = (batch_id * C + C_id) * luts_count * HW;
		float* weights_grad_ptr = weights_grad + weight_offset;
		float* weights_ptr      = weights      + weight_offset;
		// 3 x 9 x 256, 现在开始是 9x256
		int32_t lut_offset = C_id * luts_count * luts_nodes;
		float* luts_grad_ptr = luts_grad + lut_offset;
		float* luts_ptr      = luts      + lut_offset;
		// 得到当前位置的梯度
		int32_t image_pos = (batch_id * C + C_id) * HW + idx;
		float grad_value = grad_from_output[image_pos];
		// 取出图像的值
		int32_t pixel_value = lut_index[image_pos];
		// 开始算 9 次的加权
		for (int32_t i = 0; i < luts_count; ++i) {
			atomicAdd(luts_grad_ptr + pixel_value, 
				                       grad_value * weights_ptr[idx]);
			weights_grad_ptr[idx]   += grad_value * luts_ptr[pixel_value];
			weights_grad            += HW;
			weights_ptr             += HW;
			luts_grad_ptr           += luts_nodes;
			luts_ptr                += luts_nodes;
		}
	}
}




namespace {
	inline int32_t CUDA_CEIL(const int32_t x, const int32_t y) {
		return (x + y - 1) / y;
	}
}




void adaptive_1dlut_intensity_transform_backward_cuda(
		at::Tensor& weights_grad,
		at::Tensor& luts_grad,
		at::Tensor& grad_from_output,
		at::Tensor& weights,
		at::Tensor& luts,
		at::Tensor& lut_index) {
	// 获取返回梯度的信息(图像的信息)
	const int32_t B = grad_from_output.size(0);
	const int32_t C = grad_from_output.size(1);
	const int32_t H = grad_from_output.size(2);
	const int32_t W = grad_from_output.size(3);
	// 获取可学习 lut 的参数
	const int32_t luts_C     = luts_grad.size(0);
	const int32_t luts_count = luts_grad.size(1);
	const int32_t luts_nodes = luts_grad.size(2);
	// 根据前二者可以得到 weights_grad 的形状
	// printf("grad_from_output  : [%d %d %d %d]\n", B, C, H, W);
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
	adaptive_1dlut_intensity_transform_backward_kernel<BLOCK_X * BLOCK_Y><<<grid, block>>>(
		weights_grad.data_ptr<float>(),
		luts_grad.data_ptr<float>(),
		grad_from_output.data_ptr<float>(),
		weights.data_ptr<float>(),
		luts.data_ptr<float>(),
		lut_index.data_ptr<int32_t>(),
		C,
		H * W,
		luts_count,
		luts_nodes
	);
}

