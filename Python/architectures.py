# Python
import math
import numpy
# Torch
import torch


# 特征提取网路
class MultiScaleFeatureExtractor(torch.nn.Module):
	def __init__(self, in_channels=3, begin_channels=16, activate_func=torch.nn.Hardswish):
		super(MultiScaleFeatureExtractor, self).__init__()
		# 四个特征提取层
		FC = [begin_channels, begin_channels * 2, begin_channels * 4, begin_channels * 8]
		self.conv_1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels, FC[0], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
			activate_func())
		self.conv_2 = torch.nn.Sequential(
			torch.nn.Conv2d(FC[0], FC[1], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
			activate_func())
		self.conv_3 = torch.nn.Sequential(
			torch.nn.Conv2d(FC[1], FC[2], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
			activate_func())
		self.conv_4 = torch.nn.Sequential(
			torch.nn.Conv2d(FC[2], FC[3], kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
			activate_func())
		# ONNX 不支持 avgpool2d、padding=1 的情况
		self.pool_1 = torch.nn.AvgPool2d((16, 16))
		self.pool_2 = torch.nn.AvgPool2d((8,  8))
		self.pool_3 = torch.nn.AvgPool2d((4,  4))
		self.pool_4 = torch.nn.AvgPool2d((2,  2))


	def forward(self, x):
		conv_1_out = self.conv_1(x)
		conv_2_out = self.conv_2(conv_1_out)
		conv_3_out = self.conv_3(conv_2_out)
		conv_4_out = self.conv_4(conv_3_out)
		# print(conv_1_out.shape, conv_2_out.shape, conv_3_out.shape, conv_4_out)

		conv_1_out = self.pool_1(conv_1_out)
		conv_2_out = self.pool_2(conv_2_out)
		conv_3_out = self.pool_3(conv_3_out)
		conv_4_out = self.pool_4(conv_4_out)

		return torch.cat([conv_1_out, conv_2_out, conv_3_out, conv_4_out], dim=1)




import intensity_transform_1dlut
''' TODO
		1. CPU 推理没写
		2. 统计梯度和访问次数, 做后续优化
		3. forward 和 backward kernel 的优化
		4. 也可以写成多次 1d_lut 叠加的形式, 目前的写法是直接把 “全分辨率” 的加权值直接嵌入到 cuda/cpu 计算里, 不需要多次叠加
'''

# 先写训练阶段的变换, 推理阶段可以省去一些变量, 如 lut_index
class Adaptive1DLUTTransform(torch.autograd.Function):
	@staticmethod
	def forward(ctx, image, weights, luts):
		# 定义输出
		image_size = image.size()
		output = torch.zeros(image_size).float().to(image.device)

		# 定义辅助变量, 记录哪些地方有梯度
		lut_index = torch.zeros(image_size).type(torch.IntTensor).to(image.device)

		# 强制输入是连续存储的
		if (not image.contiguous()):
			image = image.contiguous()

		# C++/CUDA
		intensity_transform_1dlut.adaptive_forward(
			output,
			image,
			weights,
			luts,
			lut_index)

		# 保存反向梯度计算需要的张量
		ctx.save_for_backward(weights, luts, lut_index)
		ctx.weights_size = weights.size()
		ctx.luts_size = luts.size()

		# 结果截断在 0-1 之间? 返回
		return output.clamp_(0, 1)

	@staticmethod
	def backward(ctx, grad_from_output):
		# 返回的梯度有两部分, 一个是 weights, 一个是 luts
		weights_grad = torch.zeros(ctx.weights_size).float().to(grad_from_output.device)
		luts_grad    = torch.zeros(ctx.luts_size).float().to(grad_from_output.device)
		
		# 取出辅助变量
		weights, luts, lut_index = ctx.saved_tensors

		# C++/CUDA 计算反向梯度
		intensity_transform_1dlut.adaptive_backward(
			weights_grad,
			luts_grad,
			grad_from_output,
			weights,
			luts,
			lut_index)

		# 返回梯度, 与 forward 的输入一一对应
		return (None, weights_grad, luts_grad)

	@staticmethod
	def symbolic(g, image, weights, luts):
		return g.op("custom::adaptive_1dlut_transform", image, weights, luts)





# 还可以添加一个 0.8-1.2 的倍数的
class Adaptive1DLUTNet(torch.nn.Module):
	def __init__(self, in_channels=3, base_lut_count=9, lut_nodes=256, begin_channels=32, end_channel=128, activate_func=torch.nn.SiLU):
		super(Adaptive1DLUTNet, self).__init__()
		# 特征提取层
		self.feature_extractor = MultiScaleFeatureExtractor(
			in_channels=in_channels, begin_channels=begin_channels, activate_func=activate_func)
		# 可学习的若干个 1DLUT
		self.in_channels    = in_channels
		self.base_lut_count = base_lut_count
		self.lut_nodes 	    = lut_nodes
		self.learned_lut    = torch.nn.Parameter(torch.randn(in_channels, base_lut_count, lut_nodes).requires_grad_(True))
		# 求解一个粗糙的曲线加权值
		self.linear = torch.nn.Sequential(
			torch.nn.Conv2d(begin_channels * (1 + 2 + 4 + 8), end_channel,  kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
			activate_func(),
			torch.nn.Conv2d(end_channel, in_channels * base_lut_count, kernel_size=1),
			torch.nn.Sigmoid()
		)


	def forward(self, img, onnx_export_lowres=False):

		# 获取输入的张量大小
		b, c, h, w = img.shape
		# 【1】 下采样到固定分辨率
		x = torch.nn.functional.interpolate(img, size=(256, 256), mode="bilinear", align_corners=True)

		# 【2】 提取特征 1x240x8x8
		feature = self.feature_extractor(x)

		# 【3】 得到 3 通道, 8x8 特征图的 9 个 lut 的加权权重 1x3x9x8x8
		weights_lowres = self.linear(feature)

		# 【4】 对加权值上采样
		weights = torch.nn.functional.interpolate(weights_lowres, size=(h, w), mode="bilinear", align_corners=True)
		
		# 如果是导出 onnx, 由于最后一个算子需要自定义, 直接拿到模型最外面
		if (torch.onnx.is_in_onnx_export() and not self.training):
			# 必要的话, 可以返回低分辨率的
			determined_weights = weights_lowres if (onnx_export_lowres) else weights
			# 返回 1xHxWx(3x9)
			return determined_weights.permute(0, 2, 3, 1)

		# 【5】 对原始分辨率的图像做亮度变换
		enhanced = Adaptive1DLUTTransform.apply(img, weights, self.learned_lut)
		return enhanced














if __name__ == '__main__':

	network = Adaptive1DLUTNet(
		base_lut_count=9,
		lut_nodes=256,
		activate_func=torch.nn.ReLU).cuda()

	input_tensor = torch.randn(1, 3, 512, 341).cuda()

	output_tensor = network(input_tensor)
	print("output_tensor  ", output_tensor.shape)
	print(output_tensor.min(), output_tensor.max(), output_tensor.mean())

	trace_model = torch.jit.trace(network, input_tensor) 
	torch.jit.trace.save(trace_model, "Adaptive1DLUTNet.pt")

	
	torch.onnx.export(
		network,
		input_tensor,
		"Adaptive1DLUTNet.onnx",
		input_names=["image"],
		output_names=["enhanced"],
		opset_version=11,
		verbose=False,
		do_constant_folding=True)
	import os
	os.system("python -m onnxsim Adaptive1DLUTNet.onnx Adaptive1DLUTNet-sim.onnx")
	
	# network = MultiScaleFeatureExtractor(activate_func=torch.nn.ReLU6)

	# input_tensor = torch.randn(1, 3, 256, 256)

	# output_tensor = network(input_tensor)
	# print("output_tensor  ", output_tensor.shape)

	# script_module = torch.jit.trace(network, input_tensor, strict=False)
	# torch.jit.save(script_module, "MultiScaleFeatureExtractor.pt")

	# torch.onnx.export(
	# 	network,
	# 	input_tensor,
	# 	"MultiScaleFeatureExtractor.onnx",
	# 	input_names=["image"],
	# 	output_names=["feature"],
	# 	opset_version=11,
	# 	verbose=False,
	# 	do_constant_folding=True)
	# import os
	# os.system("python -m onnxsim MultiScaleFeatureExtractor.onnx MultiScaleFeatureExtractor-sim.onnx")

	

	
	