# MNIST-numpy

- 本项目通过纯python代码，对MNIST数据集只用numpy及gzip库实现手写数字识别  
- 利用GPU加速训练过程，实现了完整的正向传播、反向传播和Adam优化算法
- 在使用时，请将所有文件放入同一文件夹目录下
## 环境
- Python 3.13
- CUDA 12.8
- CuPy (GPU加速的NumPy兼容库)

## 网络架构与算法
- 输入层 784个神经元
- 第一隐藏层 100个神经元 (ReLU激活函数)
- 第二隐藏层 200个神经元 (ReLU激活函数)  
- 输出层 10个神经元 (Softmax激活函数)
- 优化算法 adam


### 安装依赖
- 安装CuPy（CUDA 12.x版本）
- pip install cupy-cuda12x -i https://pypi.tuna.tsinghua.edu.cn/simple
- 若无法使用cupy，将import cupy as np 改成 import numpy as np 即可
- 
### 数据来源与感谢
该数据集由 Yann LeCun 的官方网站 http://yann.lecun.com/exdb/mnist/



 # MNIST-numpy 

- This project implements handwritten digit recognition on the MNIST dataset using pure Python code, relying solely on the numpy and gzip libraries
 -Utilizing GPU to accelerate the training process, we have implemented the complete forward propagation, backward propagation, 和 Adam optimization algorithm
- When using, please place all files in the same folder directory

environment
- Python 3.13
- CUDA 12.8
- CuPy (GPU-accelerated NumPy compatible library)

## Network Architecture and Algorithms
- Input layer: 784 neurons
- The first hidden layer has 100 neurons (with ReLU activation function)
- Second hidden layer: 200 neurons (ReLU activation function)
- Output layer: 10 neurons (with Softmax activation function)
- Optimization algorithm: Adam

### Install dependencies
- Install CuPy (CUDA version 12.x)
- pip install cupy-cuda12x
- If cupy cannot be used, simply replace "import cupy as np" with "import numpy as np"



### Data Source and Acknowledgment 
- This dataset is sourced from Yann LeCun's official website at http://yann.lecun.com/exdb/mnist/

### 训练结果
### training results
- 可以看到，仅需220次迭代，训练集准确率就接近100%，是个不错的训练结果，有利于对手写数字进行识别，同时，测试集稳定在97%左右，模型较为稳定
- It can be seen that only 220 iterations are needed for the training set accuracy to approach 100%, which is a good training result conducive to the recognition of handwritten digits. Meanwhile, the test set remains stable at around 97%, indicating that the model is relatively stable.

<img width="551" height="499" alt="image" src="https://github.com/user-attachments/assets/111fb064-1d83-41b0-ab8c-eb064af40a3e" />
