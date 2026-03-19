# MNIST-numpy/n

本项目通过纯python代码，对MNIST数据集只用numpy及gzip库实现手写数字识别/n
利用GPU加速训练过程，实现了完整的正向传播、反向传播和Adam/n


环境配置/n
Python 3.13/n
CUDA12.8/n
CuPy

网络架构与算法/n
100个第一隐藏层，200个第二隐藏层/n
前两个隐藏层使用relu 输出层使用softmax函数以达到更好的效率/n
使用adam进行参数更新

