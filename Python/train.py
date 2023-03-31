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
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
# self
import utils
import evaluate
import pipeline
import loss_layer
import architectures


# ------------------------------- 定义超参等 --------------------------------------

# 参数
opt                  = lambda: None
# 训练参数
opt.seed             = 210919              # 随机种子
opt.gpu_id           = "0"                 # gpu
opt.use_cuda         = True                # 是否使用 cuda
opt.optimizer        = "torch.optim.AdamW" # 优化器
opt.lr               = 1e-3                # 学习率
opt.total_epochs     = 200                 # 训练的轮次
opt.train_batch_size = 1                   # 训练的 batch_size
opt.valid_batch_size = 1                   # 验证的 batch_size
opt.test_batch_size  = 1                   # 测试的 batch_size
opt.use_scheduler    = True  			   # 是否使用调节学习率
opt.intensity_alpha  = 1.0   			   # 约束亮度接近
opt.color_alpha      = 1.0   			   # 约束颜色接近
# 实验参数
opt.save             = True                # 是否保存权重
opt.valid_interval   = 1                   # 验证准确性的轮次间隔
opt.exp_name         = "Ada1DLUT"          # 当前实验的名字
opt.activate_func    = "torch.nn.SiLU"     # 特征提取网络中的激活函数名称
opt.small_size       = (256, 256)          # 网络可以处理的小分辨率
opt.checkpoints_dir  = os.path.join("./checkpoints/", opt.exp_name) # 保存权重的目录
# 数据集相关的设置
opt.input_dir        = "input" 
import platform
is_windows       = bool(platform.system() == "Windows")
opt.dataset_dir  = "F:/liuchang/code/2023/MIT-Adobe-FiveK" if (is_windows) else "/home/dx/usrs/liuchang/fivek"
opt.gt_dir       = "expertC_gt" if (is_windows) else "C"
# 可视化参数
opt.visualize_size   = 1
opt.visualize_batch  = 100
opt.visualize_dir    = os.path.join(opt.checkpoints_dir, 'train_phase') 
# 创建一些文件夹
for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.visualize_dir, exist_ok=True)
# 把配置文件保存到 checkpoints_dir
with open(os.path.join(opt.checkpoints_dir, "config.txt"), "w") as config_writer:
	for l, r in vars(opt).items():
		config_writer.write("{}  :  {}\n".format(l, r))


# 设置随机种子
utils.set_seed(opt.seed)
# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.set_default_tensor_type(torch.FloatTensor)


# ------------------------------- 定义网络结构 --------------------------------------
network = architectures.Adaptive1DLUTNet(
		base_lut_count=9,
		lut_nodes=256,
		activate_func=eval(opt.activate_func))

if(opt.use_cuda):
	network = network.cuda()

# ------------------------------- 定义优化器和损失函数等 --------------------------------------


# 优化器
optimizer = eval(opt.optimizer)(filter(lambda p: p.requires_grad, network.parameters()), lr=opt.lr, weight_decay=1e-5) 

# 学习率调整策略
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

# 损失函数
train_evaluator = evaluate.ImageEnhanceEvaluator(
	intensity_loss_fn=torch.nn.L1Loss(),
	color_loss_fn=loss_layer.GradientColorLoss(loss_type="torch.nn.L1Loss"),
	loss_weights=[opt.intensity_alpha, opt.color_alpha],
	metrics=["psnr"])


# 记录 valid 最佳的模型
best_psnr = -1e9
best_checkpoint = ""


# ------------------------------- 定义数据读取 --------------------------------------
make_pair = lambda x: (os.path.join(opt.dataset_dir, opt.input_dir, x), os.path.join(opt.dataset_dir, opt.gt_dir, x))
import dill as pickle
data_split = pickle.load(open("./options.pkl", "rb"))
train_images_list = [make_pair(os.path.split(it[0])[-1]) for it in data_split["train_images_list"]]
valid_images_list = [make_pair(os.path.split(it[0])[-1]) for it in data_split["valid_images_list"]]
test_images_list  = [make_pair(os.path.split(it[0])[-1]) for it in data_split["test_images_list"]]
print('\ntrain  :  {}\nvalid  :  {}\ntest  :  {}'.format(len(train_images_list), len(valid_images_list), len(test_images_list)))
print(train_images_list[:3])
print(valid_images_list[:3])


# train
train_dataset = pipeline.FiveKPairedDataset(train_images_list, train=True)
train_loader = DataLoader(
	train_dataset, 
	batch_size=opt.train_batch_size, 
	shuffle=True,
	pin_memory=True,
	worker_init_fn=utils.worker_init_fn)
# valid
valid_dataset = pipeline.FiveKPairedDataset(valid_images_list, train=False)
valid_loader = DataLoader(
	valid_dataset,
	batch_size=opt.valid_batch_size,
	shuffle=False,
	pin_memory=True)
# test
test_dataset = pipeline.FiveKPairedDataset(test_images_list, train=False)
test_loader = DataLoader(
	test_dataset,
	batch_size=opt.test_batch_size,
	shuffle=False,
	pin_memory=True)


# ------------------------------- 开始训练 --------------------------------------
for ep in range(1, opt.total_epochs + 1):
	# try:
	print()
	# 计时验证的时间
	with utils.Timer() as time_scope:
		network.train()
		train_evaluator.clear()
		# 迭代 batch
		for train_batch, (low_quality, high_quality, image_name) in enumerate(train_loader, 1):
			# 清空梯度
			optimizer.zero_grad()
			# 数据送到 GPU
			if(opt.use_cuda):
				low_quality, high_quality = low_quality.cuda(non_blocking=True), high_quality.cuda(non_blocking=True)
				
			# 经过网络
			enhanced = network(low_quality)

			# 评估损失
			loss_value = train_evaluator.update(enhanced, high_quality)
			
			# 损失回传
			loss_value.backward()

			# 更新学习率
			if (opt.use_scheduler): 
				scheduler.step(ep + train_batch / len(train_loader))

			# w -= lr * gradient
			optimizer.step()

			# 输出信息
			output_infos = '\rTrain===> [epoch {}/{}] [batch {}/{}] [loss {:.4f} {:.4f} {:.4f}]  [PSNR: {:.4f}db | SSIM: {:.4f}] [lr {:.5f}]'.format(
				ep, opt.total_epochs, train_batch, len(train_loader), 
				*train_evaluator.get(), optimizer.state_dict()['param_groups'][0]['lr'])
			sys.stdout.write(output_infos)

			# 可视化一些图像
			# if(train_batch % opt.visualize_batch == 0 and opt.train_batch_size % opt.visualize_size == 0):
			# 	utils.visualize_a_batch(enhanced, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_enhanced.png".format(ep, train_batch)))
			# 	utils.visualize_a_batch(low_quality, save_path=os.path.join(opt.visualize_dir, "ep_{}_batch_{}_low_quality.png".format(ep, train_batch)))
		
	# --------------------------- validation ------------------------
	# 验证
	if(ep % opt.valid_interval == 0):
		with utils.Timer() as time_scope:
			network.eval()
			valid_evaluator = evaluate.ImageEnhanceEvaluator(
				intensity_loss_fn=torch.nn.L1Loss(),
				color_loss_fn=loss_layer.GradientColorLoss(loss_type="torch.nn.L1Loss"),
				loss_weights=[opt.intensity_alpha, opt.color_alpha],
				metrics=["psnr", "ssim"])
			with torch.no_grad():
				for valid_batch, (low_quality, high_quality, image_name) in enumerate(valid_loader, 1):
					# 清空梯度
					optimizer.zero_grad()
					# 数据送到 GPU
					if(opt.use_cuda):
						low_quality, high_quality = low_quality.cuda(non_blocking=True), high_quality.cuda(non_blocking=True)
						
					# 经过网络
					enhanced = network(low_quality)

					# 评估损失
					loss_value = valid_evaluator.update(enhanced, high_quality)
					
					# 输出信息
					output_infos = '\rValid===> [epoch {}/{}] [batch {}/{}] [loss {:.5f} {:.3f} {:.3f}] [PSNR: {:.4f}db | SSIM: {:.4f}] [lr {:.5f}]'.format(
						ep, opt.total_epochs, valid_batch, len(valid_loader),
						 *valid_evaluator.get(), optimizer.state_dict()['param_groups'][0]['lr'])
					sys.stdout.write(output_infos)

				# 保存网络
				save_path  = os.path.join(opt.checkpoints_dir, 
					'epoch_{}_train_{:.3f}_{:.4f}_valid_{:.3f}_{:.4f}.pth'.format(
						ep, *train_evaluator.get_psnr_ssim(), *valid_evaluator.get_psnr_ssim()))
				torch.save(network.state_dict(), save_path)
				print("\nnetwork saved to===>  {}".format(save_path))

				if (valid_evaluator.get_psnr_ssim()[0] > best_psnr):
					best_psnr = valid_evaluator.get()[0]
					best_checkpoint = save_path

	# except Exception as e:
	# 	# 默认在终端打印异常信息, 默认色彩是红色
	# 	traceback.print_exc()
	# 	# 使用变量接收错误信息
	# 	err = traceback.format_exc()
	# 	continue


				
# 开始测试
print("训练结束, 开始加载 valid 性能最好的模型权重做测试")
network.load_state_dict(torch.load(best_checkpoint))
print("loaded weights from {}".format(best_checkpoint))

with utils.Timer() as time_scope:
	network.eval()
	test_evaluator = evaluate.ImageEnhanceEvaluator(
		intensity_loss_fn=torch.nn.L1Loss(),
		color_loss_fn=loss_layer.GradientColorLoss(loss_type="torch.nn.L1Loss"),
		loss_weights=[opt.intensity_alpha, opt.color_alpha],
		metrics=["psnr", "ssim"])
	with torch.no_grad():
		for test_batch, (low_quality, high_quality, image_name) in enumerate(test_loader, 1):
			# 清空梯度
			optimizer.zero_grad()
			# 数据送到 GPU
			if(opt.use_cuda):
				low_quality, high_quality = low_quality.cuda(non_blocking=True), high_quality.cuda(non_blocking=True)
				
			# 经过网络
			enhanced = network(low_quality)

			# 评估损失
			loss_value = test_evaluator.update(enhanced, high_quality)
			
			# 输出信息
			output_infos = '\rtest===> [epoch {}/{}] [batch {}/{}] [loss {:.5f} {:.3f} {:.3f}] [PSNR: {:.4f}db | SSIM: {:.4f}] [lr {:.5f}]'.format(
				ep, opt.total_epochs, test_batch, len(test_loader),
				 *test_evaluator.get(), optimizer.state_dict()['param_groups'][0]['lr'])
			sys.stdout.write(output_infos)