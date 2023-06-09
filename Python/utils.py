import os
import cv2
import sys
import math
import time
import numpy
import random
import torch


# 为 torch 数据随机做准备
def set_seed(seed):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 是否可复现
    torch.backends.cudnn.benchmark     = False # 静态输入的情况下开启, 可以搜索更优的 Kernel
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

GLOBAL_SEED = 19980212
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)



# 计时函数
class Timer:
    def __enter__(self):
        self.start = time.process_time()

    def __exit__(self, type, value, trace):
        _end = time.process_time()
        print('耗时  :  {}'.format(_end - self.start))


# 可视化
def visualize_a_batch(batch_images, save_path, total_size=16):
    row = int(math.sqrt(total_size))
    # tensor -> numpy
    batch_images = batch_images.detach().cpu().permute(0, 2, 3, 1).mul(255).numpy().astype('uint8')
    # (16, 512, 512, 3) -> [4 * 512, 4 * 512, 3]
    composed_images = numpy.concatenate([numpy.concatenate([batch_images[row * i + j] for j in range(row)], axis=1) for i in range(row)], axis=0)
    cv2.imwrite(save_path, composed_images)


