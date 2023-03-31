import torch



# 根据两张图在 [r,g,b] 向量上的余弦距离作为颜色损失函数
class CosineLoss(torch.nn.Module):
	def __init__(self, dim=1):
		super(CosineLoss, self).__init__()
		self.loss = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)

	# 给定两张图像
	def forward(self, lhs, rhs):
		# B, C, H, W
		return 1 - self.loss(lhs, rhs).mean()


class CosineLossMultiScale(torch.nn.Module):
	def __init__(self, alpha=1.0, dim=1):
		super(CosineLossMultiScale, self).__init__()
		self.loss  = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
		self.alpha = alpha

	# 给定两张图像
	def forward(self, lhs, rhs):
		# B, C, H, W
		# 首先把图像切分成很多小块
		b, c, h, w = lhs.shape
		h_2, w_2 = int(h / 2), int(w / 2)
		h_4, w_4 = int(h / 4), int(w / 4)
		# 整图
		total_loss = 1 - self.loss(lhs, rhs).mean()
		# 四个角
		total_loss += 1 - self.loss(lhs[..., :h_2, :w_2], rhs[..., :h_2, :w_2]).mean()
		total_loss += 1 - self.loss(lhs[..., :h_2, w_2:], rhs[..., :h_2, w_2:]).mean()
		total_loss += 1 - self.loss(lhs[..., h_2:, :w_2], rhs[..., h_2:, :w_2]).mean()
		total_loss += 1 - self.loss(lhs[..., h_2:, w_2:], rhs[..., h_2:, w_2:]).mean()
		# 最中间
		total_loss += 1 - self.loss(lhs[..., h_4: h_4 + h_2, w_4: w_4 + w_2], rhs[..., h_4: h_4 + h_2, w_4: w_4 + w_2]).mean()
		return total_loss * self.alpha



# 根据梯度来约束颜色信息
class GradientColorLoss(torch.nn.Module):
	def __init__(self, loss_type="torch.nn.L1Loss"):
		super(GradientColorLoss, self).__init__()
		self.loss = eval(loss_type)()

	# 给定两张图像
	def forward(self, lhs, rhs):
		# B, C, H, W
		# 计算 lhs 的梯度
		lhs_gradient = lhs[:, 1:] - lhs[:, :-1]
		# 计算 rhs 的梯度
		rhs_gradient = rhs[:, 1:] - rhs[:, :-1]
		# 两张图在 C 通道上的
		return self.loss(lhs_gradient, rhs_gradient)



# 一维 LUT 的单调性损失
class MonotonicityLoss(torch.nn.Module):
	def __init__(self):
		super(MonotonicityLoss, self).__init__()
	
	def forward(self, curves):
		loss_value =  torch.nn.functional.relu(curves[..., :-1] - curves[..., 1:])
		return loss_value.mean()



if __name__ == '__main__':
	
	lhs = torch.randn(1, 3, 512, 341)
	rhs = torch.randn(1, 3, 512, 341)

	loss = CosineLoss()

	loss_value = loss(lhs, rhs)
	print("loss_value  ", loss_value)

	loss = CosineLoss(dim=0)

	lhs = torch.tensor([0.2, -4, 1.5])
	rhs = torch.tensor([0.4, -5.6, 2])
	print(loss(lhs, rhs))


	loss = CosineLossMultiScale(dim=1, alpha=0.2)
	lhs = torch.randn(1, 3, 512, 341)
	rhs = torch.randn(1, 3, 512, 341)
	print("loss  ", loss(lhs, rhs))


	loss = GradientColorLoss()
	print(loss(lhs, rhs))