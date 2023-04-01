使用 MNN 推理引擎，对一个含有简单自定义算子的模型做 C++ 推理

# 环境

- MNN 2.4.0
- C++17(TDM GCC 10.3.0)
- Xmake 2.7.4
- Windows 11
- OpenCV 4.5.5（可选）
- stb_image（可选）

# 步骤

- 将 ONNX 模型转换成 MNN

    ```bash
    MNNConvert --framework ONNX --modelFile ../../IR/Ada1DLUT_dynamic_sim.onnx --MNNModel Ada1DLUT_dynamic.mnn --bizCode MNN
    ```

- 看下 MNN 模型的信息(可选）

    ```bash
    GetMNNInfo Ada1DLUT_dynamic.mnn
    ```

- 验证 MNN 精度和 ONNX 是否对齐(可选, 需要进入 MNN 目录)

    ```bash
    python ../tools/script/testMNNFromOnnx.py Ada1DLUT_dynamic.mnn
    ```

- Xmake 构建，做推理

    ```bash
    xmake f -p mingw
    xmake build
    xmake run
    ```

# 结果

- float32、动态输入

    ![](images/float32_dynamic.gif)