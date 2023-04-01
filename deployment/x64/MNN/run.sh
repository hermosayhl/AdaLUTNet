# 编译 MNN
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=true -DCMAKE_INSTALL_PREFIX=E:/environments/C++/mnn/MNN/install_gcc
# 转换模型
MNNConvert --framework ONNX --modelFile ../../IR/Ada1DLUT_dynamic_sim.onnx --MNNModel Ada1DLUT_dynamic.mnn --bizCode MNN