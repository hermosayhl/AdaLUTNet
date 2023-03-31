// C++
#include <cmath>
#include <vector>
#include <iostream>
// torch
#include <torch/extension.h>



#ifdef WITH_CUDA
void adaptive_1dlut_intensity_transform_forward_cuda(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& weights,
		const at::Tensor& luts,
		at::Tensor& lut_index);

void adaptive_1dlut_intensity_transform_backward_cuda(
	at::Tensor& weights_grad,
	at::Tensor& luts_grad,
	at::Tensor& grad_from_output,
	at::Tensor& weights,
	at::Tensor& luts,
	at::Tensor& lut_index);
#endif


void adaptive_1dlut_intensity_transform_forward(
		at::Tensor& output, 
		const at::Tensor& image, 
		const at::Tensor& weights,
		const at::Tensor& luts,
		at::Tensor& lut_index) {
	
// #ifdef WITH_ASSERT
	// &&
	// assert (image.is_contiguous()   and "image is not contiguous!");
	// assert (weights.is_contiguous() and "weights is not contiguous!");
	// assert (luts.is_contiguous()    and "luts is not contiguous!");
	// assert (weights_C == C and weights_H == H and weights_W == W and "image and weights should have same size of image");
	// assert (luts_C == C and luts_count * luts_C == weight_C and "luts have invalid dimensions");
// #endif

	if (image.is_cuda()) {
#ifdef WITH_CUDA
		adaptive_1dlut_intensity_transform_forward_cuda(output, image, weights, luts, lut_index);
#else
		AT_ERROR("Function 'adaptive_1dlut_intensity_transform_forward' is not complied with GPU support!");
#endif
	}
	else {
		AT_ERROR("Function 'adaptive_1dlut_intensity_transform_forward' is not complied with CPU support yet! Coming soon");
	}
}



void adaptive_1dlut_intensity_transform_backward(
	at::Tensor& weights_grad,
	at::Tensor& luts_grad,
	at::Tensor& grad_from_output,
	at::Tensor& weights,
	at::Tensor& luts,
	at::Tensor& lut_index) {
	if (grad_from_output.is_cuda()) {
#ifdef WITH_CUDA
		adaptive_1dlut_intensity_transform_backward_cuda(weights_grad, luts_grad, grad_from_output, weights, luts, lut_index);
#else
		AT_ERROR("Function 'adaptive_intensity_transform_1dlut_backward' is not complied with GPU support!");
#endif
	}
	else {
		AT_ERROR("Function 'adaptive_intensity_transform_1dlut_backward' is not complied with CPU support yet! Coming soon");
	}
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"adaptive_forward", 
		&adaptive_1dlut_intensity_transform_forward, 
		"adaptive forward for 1dlut intensity transform", 
		py::arg("output"), 
		py::arg("weights"), 
		py::arg("image"), 
		py::arg("luts"), 
		py::arg("lut_index")
	);
	m.def(
		"adaptive_backward", 
		&adaptive_1dlut_intensity_transform_backward, 
		"adaptive backward for 1dlut intensity transform", 
		py::arg("weights_grad"), 
		py::arg("luts_grad"), 
		py::arg("grad_from_output"), 
		py::arg("weights"),
		py::arg("luts"),
		py::arg("lut_index")
	);

}