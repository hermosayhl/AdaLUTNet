import os
import cv2
import random
import numpy
import torch
import dill as pickle
from torch.utils.data import Dataset


# 展示
def show(image, name='crane'):
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 写入
def write(image, save_path):
	cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# 旋转
def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation


# 数据扩充
def make_augment(low_quality, high_quality):
	# 以 0.9 的概率作数据增强
	if (random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate', 'brighten']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if (cur_state == 'flip'):
				# 0.5 概率水平翻转
				if (random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif (cur_state == 'crop'):
				# 0.5 概率做裁剪
				if (random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.5, 0.99)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif (cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
			elif (cur_state == 'brighten'):
				if (random.random() > 1 - 0.3):
					degree = random.uniform(0.8, 1.25)
					low_quality = low_quality * degree
					# print("把输入变暗或者变亮")

	return low_quality, high_quality



class FiveKPairedDataset(Dataset):
	def __init__(self, images_list, train=False):
		self.images_list = images_list
		self.is_train = train
		# numpy → torch.Tensor
		self.transform  = lambda x: torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor).contiguous()
		# torch.Tensor → numpy
		self.restore    = lambda x: x.detach().cpu().mul(255).permute(0, 2, 3, 1).numpy().astype('uint8')


	def __len__(self):
		return len(self.images_list)


	def __getitem__(self, idx):

		# 读取图像
		input_path, label_path = self.images_list[idx]

		# 范围放缩到 [0-1]
		low_quality = cv2.imread(input_path).astype("float32") / 255.0
		high_quality = cv2.imread(label_path).astype("float32") / 255.0

		# train
		if(self.is_train == True):
			# 数据增强
			low_quality, high_quality = make_augment(low_quality, high_quality)

		# 返回
		return self.transform(low_quality), \
			   self.transform(high_quality), \
			   os.path.split(input_path)[-1]