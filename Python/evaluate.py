# Python
import math
# 3rd party
import cv2
import numpy
# tensorflow
import torch
from torch.autograd import Variable
import torch.nn.functional as F


###########################################################################
#                                Metrics
###########################################################################

class ImageEnhanceEvaluator():
	def __init__(self, intensity_loss_fn, color_loss_fn, loss_weights=[1.0, 0.5], metrics=["psnr"]):
		# 损失函数
		self.intensity_loss_fn   = intensity_loss_fn
		self.color_loss_fn       = color_loss_fn
		self.loss_weights        = loss_weights
		# 统计平均损失
		self.mean_intensity_loss = 0
		self.mean_color_loss     = 0
		self.mean_loss 			 = 0
		# 一些衡量图像拟合程度的客观度量
		self.metrics      = metrics
		self.mse_loss_fn  = torch.nn.MSELoss()
		self.compute_psnr = lambda mse: 10 * math.log10(1. / mse) if(mse > 1e-5) else 50
		self.mean_psnr    = 0
		self.mean_ssim    = 0
		# ssim
		self.compute_ssim = SSIM(window_size=11)
		# 统计次数, 方便取平均
		self.count = 0
		

	def update(self, pred_image, label_image):

		# 计算亮度损失
		intensity_loss = self.intensity_loss_fn(pred_image, label_image)
		color_loss     = self.color_loss_fn(pred_image, label_image)
		total_loss     = self.loss_weights[0] * intensity_loss + self.loss_weights[1] * color_loss
		self.mean_intensity_loss += intensity_loss.item()
		self.mean_color_loss     += color_loss.item()
		self.mean_loss           += total_loss.item()
		
		# 计算 PSNR
		if ("psnr" in self.metrics):
			mse_loss_value = self.mse_loss_fn(pred_image, label_image).item()
			psnr_value     = self.compute_psnr(mse_loss_value)
			self.mean_psnr += psnr_value
		# 计算 SSIM
		if ("ssim" in self.metrics):
			ssim_value     = self.compute_ssim(label_image, pred_image).item()
			self.mean_ssim += ssim_value

		# 计数 + 1
		self.count += 1

		# 返回损失
		return total_loss


	def get(self):
		# 如果没有计数, 返回 0
		if(self.count == 0):
			return (0, 0, 0) if (self.psnr_only) else (0, 0, 0)
		# 返回累计的均值
		return self.mean_intensity_loss / self.count, \
			   self.mean_color_loss / self.count, \
			   self.mean_loss / self.count, \
			   self.mean_psnr / self.count, \
			   self.mean_ssim / self.count

	def get_psnr_ssim(self):
		return self.mean_psnr / self.count, self.mean_ssim / self.count

	# 清空
	def clear(self):
		self.count 				 = 0
		self.mean_intensity_loss = 0
		self.mean_color_loss     = 0
		self.mean_loss           = 0
		self.mean_psnr           = 0
		self.mean_ssim           = 0





###########################################################################
#                        SSIM, 来自 pytorch-msssim
###########################################################################
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
	def __init__(self, window_size = 11, size_average = True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)
			
			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)
			
			self.window = window
			self.channel = channel


		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)
	
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)
	
	return _ssim(img1, img2, window, window_size, channel, size_average)