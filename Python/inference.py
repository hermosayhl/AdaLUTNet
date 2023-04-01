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
device = torch.device("cuda")
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
image      = cv2.imread(image_path).astype("float32")

# numpy → torch, 最好加 contiguous(), 因为后面的自定义算子只能连续存储的 Tensor
image_tensor = torch.as_tensor(image).unsqueeze(0)
image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous().div(255).to(device)
print(image_tensor.shape, image_tensor.dtype)

network.eval()
with torch.no_grad():
	# 推理
	output_tensor = network(image_tensor)
	# 展示
	enhanced = output_tensor.squeeze(0).clamp(0, 1).mul(255).permute(1, 2, 0).detach().cpu().numpy().astype('uint8')
	pipeline.show(enhanced)
	# 保存
	pipeline.write(enhanced, image_path.replace(".png", "_enhanced.png"))
