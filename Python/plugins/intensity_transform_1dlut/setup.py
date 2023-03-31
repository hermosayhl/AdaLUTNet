from setuptools import setup
from setuptools.command.build_ext import build_ext
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension



ext_name = "intensity_transform_1dlut"

if(torch.cuda.is_available()):
	extension_type = CUDAExtension
	define_macros = [("WITH_CUDA", None), ("WITH_ASSERT", None)]
	source_files = [
		"csrc/intensity_transform_1dlut_forward.cu",
		"csrc/intensity_transform_1dlut_backward.cu",
		"csrc/intensity_transform_1dlut.cpp"
	]
	print(f"Compiling with CUDA support")
else:
	extension_type = CppExtension
	define_macros = []
	source_files = [
		"csrc/intensity_transform_1dlut.cpp"
	]


setup(
	name=ext_name, 
	version="0.1",
	author="bupt-liuchang",
	ext_modules=[
		extension_type(
			name=ext_name,
			sources=source_files,
			define_macros=define_macros
		)
	],
	cmdclass={
		"build_ext": BuildExtension
	}
)


# 1. Windows Visual Studio 2019 + CUDA10.1, 倘若发生错误 
# 	error: function "torch::OrderedDict<Key, Value>::Item::operator=(const torch::OrderedDict<std::string, at::Tensor>::Item &) [with Key=std::string, Value=at::Tensor]" (declared implicitly) cannot be referenced -- it is a deleted function
# 参考, 修改源码即可编译过去 https://github.com/pytorch/pytorch/pull/55275/files/bc4867bb5a074d810c6c5f233b47275322d875d0