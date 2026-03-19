import cupy as np
import gzip

# onehot 编码
def one_hot(labels):
    one_hot = np.zeros((10,len(labels)))
    for i in range((len(labels))):
        one_hot[labels[i],i] = 1
    return one_hot


# 激活函数relu sigmoid，,激活函数的导数
# 784个输入，第一个隐藏层100个，第二隐藏层200个 输出层10个x

def relu(x):
    return np.where(x>0,x,0)
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_relu(x):
    return np.where(x > 0 , 1, 0 )

# 输出层函数 sofmax
"""
直接return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
有溢出问题
根据书籍修改
"""

def softmax(x):
    c = np.max(x,axis=0,keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x,axis=0,keepdims=True)
    y = exp_x / sum_exp_x
    return y


# 神经网络，前向传播，反向传播
class Neural_Network:

# 随机初始化参数 3.11
    def init_network(self):
        self.w1=np.random.randn(100,784) * 0.01
        self.b1=np.zeros((100,1))
        self.w2=np.random.randn(200,100) * 0.01
        self.b2=np.zeros((200,1))
        self.w3=np.random.randn(10,200) * 0.01
        self.b3=np.zeros((10,1))
        return self.w1,self.b1,self.w2,self.b2,self.w3,self.b3

# 前向传播
    def forward_network(self,x):
        # 第一层
        z1 = self.w1 @ x +self.b1
        a1=relu(z1)
        # 第二层
        z2 = self.w2 @ a1 + self.b2
        a2=relu(z2)
        # 第三层
        z3 = self.w3 @ a2 + self.b3
        a3=softmax(z3)
        return a1,a2,a3,z1,z2,z3

# 反向传播
    def backward_network(self,a1,a2,a3,z1,z2,z3,Y,m,x):
        # 第三层
        dz3= a3 - Y
        dw3=(1/m) * (dz3 @ a2.T)
        db3=(1/m) * (np.sum(dz3,axis=1,keepdims=True))
        # 第二层
        dz2= self.w3.T @ dz3 * d_relu(z2)
        dw2=(1/m) * (dz2 @ a1.T)
        db2=(1/m) * (np.sum(dz2,axis=1,keepdims=True))
        # 第一层
        dz1= self.w2.T @ dz2 * d_relu(z1)
        dw1=(1/m) * (dz1 @ x.T)
        db1=(1/m) * (np.sum(dz1,axis=1,keepdims=True))
        return dw1,dw2,dw3,db1,db2,db3

if __name__ == '__main__':
            # 学习率
            alpha=0.01
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 10 ** -8
            vdw1 = 0
            sdw1 = 0
            vdb1 = 0
            sdb1 = 0
            vdw2 = 0
            sdw2 = 0
            vdb2 = 0
            sdb2 = 0
            vdw3 = 0
            sdw3 = 0
            vdb3 = 0
            sdb3 = 0
            t = 0
            """
            0.01 恒定为11.24
            0.3以上梯度爆炸，变为恒定9.87
            综上 使用 0.3
            以上为使用梯度下降的学习率
            使用adam后，0.01更为合适
            """
            # 读取数据
            # 读取训练集数据
            # 读取训练集图片
            with gzip.open('train-images-idx3-ubyte.gz', 'rb') as train_images:
                content = train_images.read()
                train_image = np.frombuffer(content, np.uint8, offset=16)
            # 读取训练集标签
            with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as train_labels:
                content = train_labels.read()
                train_label = np.frombuffer(content, np.uint8, offset=8)
            # 读取测试集图片
            with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as test_images:
                content = test_images.read()
                test_image = np.frombuffer(content, np.uint8, offset=16)
            # 读取测试集标签
            with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as test_labels:
                content = test_labels.read()
                test_label = np.frombuffer(content, np.uint8, offset=8)


            # 归一化数据，标签和样本
            # 训练集
            train_images_guiyi = train_image.reshape(60000,784).T / 255.0
            train_Y = one_hot(train_label)
            train_M = train_images_guiyi.shape[1]
            # 测试集
            test_images_guiyi = test_image.reshape(10000,784).T /255.0
            test_Y = one_hot(test_label)
            test_M = test_images_guiyi.shape[1]

            nn = Neural_Network()
            nn.init_network()

            # for循环
            for i in range(2000):
                # 前向传播
                a1,a2,a3,z1,z2,z3 = nn.forward_network(train_images_guiyi)
                # 反向传播
                dw1,dw2,dw3,db1,db2,db3 = nn.backward_network(a1,a2,a3,z1,z2,z3,train_Y,train_M,train_images_guiyi) #x转变为train_images_guiyi
                t += 1
                # Adam
                vdw1 = beta1 * vdw1 + (1 - beta1) * dw1
                vdb1 = beta1 * vdb1 + (1 - beta1) * db1
                sdw1 = beta2 * sdw1 + (1 - beta2) * (dw1 ** 2)
                sdb1 = beta2 * sdb1 + (1 - beta2) * (db1 ** 2)

                vdw2 = beta1 * vdw2 + (1 - beta1) * dw2
                vdb2 = beta1 * vdb2 + (1 - beta1) * db2
                sdw2 = beta2 * sdw2 + (1 - beta2) * (dw2 ** 2)
                sdb2 = beta2 * sdb2 + (1 - beta2) * (db2 ** 2)

                vdw3 = beta1 * vdw3 + (1 - beta1) * dw3
                vdb3 = beta1 * vdb3 + (1 - beta1) * db3
                sdw3 = beta2 * sdw3 + (1 - beta2) * (dw3 ** 2)
                sdb3 = beta2 * sdb3 + (1 - beta2) * (db3 ** 2)

                # 参数更新
                vdw1_c = vdw1 / (1 - beta1 ** t)
                sdw1_c = sdw1 / (1 - beta2 ** t)
                vdb1_c = vdb1 / (1 - beta1 ** t)
                sdb1_c = sdb1 / (1 - beta2 ** t)

                vdw2_c = vdw2 / (1 - beta1 ** t)
                sdw2_c = sdw2 / (1 - beta2 ** t)
                vdb2_c = vdb2 / (1 - beta1 ** t)
                sdb2_c = sdb2 / (1 - beta2 ** t)

                vdw3_c = vdw3 / (1 - beta1 ** t)
                sdw3_c = sdw3 / (1 - beta2 ** t)
                vdb3_c = vdb3 / (1 - beta1 ** t)
                sdb3_c = sdb3 / (1 - beta2 ** t)

                nn.w1 -= alpha * vdw1_c / (np.sqrt(sdw1_c) + epsilon)
                nn.b1 -= alpha * vdb1_c / (np.sqrt(sdb1_c) + epsilon)

                nn.w2 -= alpha * vdw2_c / (np.sqrt(sdw2_c) + epsilon)
                nn.b2 -= alpha * vdb2_c / (np.sqrt(sdb2_c) + epsilon)

                nn.w3 -= alpha * vdw3_c / (np.sqrt(sdw3_c) + epsilon)
                nn.b3 -= alpha * vdb3_c / (np.sqrt(sdb3_c) + epsilon)

                # nn.w1 = nn.w1 - alpha * dw1
                # nn.w2 = nn.w2 - alpha * dw2
                # nn.w3 = nn.w3 - alpha * dw3
                # nn.b1 = nn.b1 - alpha * db1
                # nn.b2 = nn.b2 - alpha * db2
                # nn.b3 = nn.b3 - alpha * db3
                # 原来的参数更新，不用了

                if i % 10 == 0:
                    train_predictions = np.argmax(a3, axis=0)
                    train_accuracy = np.mean(train_predictions == train_label)

                    a1,a2,a3_test,z1,z2,z3_test = nn.forward_network(test_images_guiyi)
                    test_predictions = np.argmax(a3_test, axis=0)
                    test_accuracy = np.mean(test_predictions == test_label)
                    accuracy = np.mean(test_predictions == test_label)
                    print(f"训练集迭代次数: {i}, 训练集准确率: {train_accuracy * 100:.2f}%\n测试集迭代次数:{i},测试集准确率{test_accuracy * 100:.2f}%")

