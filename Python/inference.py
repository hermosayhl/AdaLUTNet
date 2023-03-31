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

# 读取图像
image_path = "./images/test/IMG_20230315_124509.png"
image      = cv2.imread(image_path)

# numpy → torch
image_tensor = torch.as_tensor(image).unsqueeze(0)
image_tensor = image_tensor.permute(0, 3, 1, 2).type(torch.FloatTensor).div(255)
print(image_tensor.shape, image_tensor.dtype)

network.eval()
with torch.no_grad():
	# 推理
	output_tensor = network(image_tensor)
	print("output_tensor  ", output_tensor.shape)

	print(output_tensor[0, 0, 0, :100])
	print(output_tensor.min(), output_tensor.max(), "{:.7f}".format(output_tensor.mean()))


