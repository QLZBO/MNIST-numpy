# MNIST-numpy

本项目通过纯python代码，对MNIST数据集只用numpy及gzip库实现手写数字识别
利用GPU加速训练过程，实现了完整的正向传播、反向传播和Adam


环境配置
Python 3.13
CUDA12.8
CuPy

网络架构与算法
100个第一隐藏层，200个第二隐藏层
前两个隐藏层使用relu 输出层使用softmax函数以达到更好的效率
使用adam进行参数更新

