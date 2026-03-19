# MNIST-numpy

本项目通过纯python代码，对MNIST数据集只用numpy及gzip库实现手写数字识别  
利用GPU加速训练过程，实现了完整的正向传播、反向传播和Adam优化算法

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
安装CuPy（CUDA 12.x版本）
pip install cupy-cuda12x
若无法使用cupy，将import cupy as np 改成 import numpy as np 即可
