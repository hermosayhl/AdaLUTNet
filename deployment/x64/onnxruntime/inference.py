# Python
import os
# 3rd party
import cv2
import numpy
import onnxruntime

def cv_show(image):
	cv2.imshow("crane", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 加载 onnx 模型
onnx_path = "../../IR/Ada1DLUT_dynamic_sim.onnx"
task = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# 读取图像
image_path = "IMG_20230315_124509.png"
image      = cv2.imread(image_path)

# numpy → tensor
def convert_to_tensor(x):
	x = x.astype("float32") / 255
	x = numpy.ascontiguousarray(x.transpose(2, 0, 1))
	print(x[0, 0, :100])
	return x[None]

# 执行推理
# 输入: image
# 输出: weight
# 注意: 图像需要扩张第一维度
[weight] = task.run(["weight"], {"image": convert_to_tensor(image)})
print(weight.shape)
print("weight  ", weight.min(), weight.max(), weight.mean())

# 后处理, 后续考虑添加 cpp 实现
print(weight.flatten()[:100].round(5))

