# Python
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')
import traceback
# 3rd party
import cv2
import numpy
import dill as pickle
# torch
import torch
torch.set_num_threads(2)
# self
import utils
import evaluate
import pipeline
import loss_layer
import architectures


# 定义网络结构
device = torch.device("cpu")
network = architectures.Adaptive1DLUTNet(
		base_lut_count=9,
		lut_nodes=256,
		activate_func=torch.nn.SiLU).to(device)

# 加载训练好的权重
pretrained_weights = "./checkpoints/Ada1DLUT/epoch_100_train_21.672_0.0000_valid_22.214_0.8544.pth"
pretrained_weights = torch.load(pretrained_weights, map_location=device)
network.load_state_dict(pretrained_weights)


# 保存路径
save_dir = "../deployment/IR"
os.makedirs(save_dir, exist_ok=True)

# 参数
export_dynamic = True

# 导出动态形状
if (export_dynamic):
	# 准备一个输入
	input_tensor = torch.randn(1, 3, 512, 341).to(device)
	export_onnx_path = os.path.join(save_dir, "Ada1DLUT_dynamic.onnx")
	torch.onnx.export(
		network,
		input_tensor,
		export_onnx_path,
		input_names=["image"],
		output_names=["weight"],
		dynamic_axes={
			"image": {0: "image_batch", 2: "image_height", 3: "image_width"},
			"weight": {0: "weight_batch", 2: "weight_height", 3: "weight_width"}
		},
		opset_version=12,
		verbose=False
	)

elif (not export_dynamic):
	# 如果导出静态输入的模型, 可以把下采样和上采样拿到模型推理之外, 估计权重这一步完全交给静态输入模型
	input_tensor = torch.randn(1, 3, 768, 1024).to(device)
	export_onnx_path = os.path.join(save_dir, "Ada1DLUT_static.onnx")
	torch.onnx.export(
		network,
		input_tensor,
		export_onnx_path,
		input_names=["image"],
		output_names=["weight"],
		opset_version=12,
		verbose=False
	)

# 简化模型, onnxsim 0.4.0 之后就不用指定动态维度了
os.system("python -m onnxsim {} {}".format(export_onnx_path, export_onnx_path.replace(".onnx", "_sim.onnx")))


# 由于目前导出的模型不包含自定义算子, 因此可学习的几个 lut 不在 onnx 中, 需要单独导出
luts = network.learned_lut.cpu().detach().numpy().flatten()
luts.astype("float32").tofile(os.path.join(save_dir, "learned_lut.bin"))
print(luts.shape)