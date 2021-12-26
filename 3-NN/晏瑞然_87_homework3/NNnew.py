import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 文件路径：
####################################
# insets文件由于数据错误已删除
# dataset = './insets/insets2-training.txt'
# dataset = './insets/insets-training.txt'
# testset = './insets/insets-testing.txt'
# testset = './insets/insets-2-testing.txt'
####################################
# dataset = './data2/Standard2.txt'
# testset = './data2/Standard2_test.txt'
# dataset = './data2/Noise2.txt'
# testset = './data2/Noise2_test.txt'
dataset = './points/points-training.txt'
testset = './points/points-testing.txt'
# dataset = './points/points-2-training.txt'
# testset = './points/points-2-testing.txt'

# 设置全局变量
learningRate = 0.05
iterations = 1000


class Net(torch.nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(
            in_dim, n_hidden_1), torch.nn.BatchNorm1d(n_hidden_1), torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(
            n_hidden_1, n_hidden_2), torch.nn.BatchNorm1d(n_hidden_2), torch.nn.ReLU(True))
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


start = datetime.datetime.now()
net = Net(2, 64, 36, 20)
optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)
loss_func = torch.nn.CrossEntropyLoss()
lossList = []
tList = []

# 获取数据
data = np.loadtxt(dataset, dtype=np.float)
x = data[:, 0:2]
x = torch.from_numpy(x).type(torch.FloatTensor)
y = data[:, 2:].astype(int).T
y = torch.from_numpy(y).type(torch.LongTensor)[0]

testdata = np.loadtxt(testset, dtype=np.float)
xtest = testdata[:, 0:2]
xtest = torch.from_numpy(xtest).type(torch.FloatTensor)
ytest = testdata[:, 2:].astype(int).T
ytest = torch.from_numpy(ytest).type(torch.LongTensor)[0]
# 学习部分==============================
plt.ion()  # 动态学习过程展示
plt.show()


for t in range(iterations):
    out = net(x)  # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, y)  # 计算两者的误差
    lossList.append(loss)
    tList.append(t)
    # print(loss)
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 10 == 0:
        # print(loss)
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(out, 1)[1]
        plt.scatter(x[:, 0], x[:, 1], c=prediction, lw=0, cmap='RdYlGn', s=10)
        # accuracy = sum((prediction == y).numpy()) / 200.  # 预测中有多少和真实值一样
        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

end = datetime.datetime.now()
print(f'programming time is {end - start}')

plt.ioff()
plt.show()

plt.scatter(np.array(tList), np.array(lossList), s=5)
plt.show()

testout = net(xtest)
# print(testout)

ypridict = []
for result in testout:
    ypridict.append(torch.argmax(result))
# print(ypridict)
ypridict = torch.tensor(ypridict)

# 画结果对比图
plt.figure(1)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)

plt.sca(ax1)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='RdYlGn', s=5)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)

plt.sca(ax2)
plt.scatter(testdata[:, 0], testdata[:, 1],
            c=testdata[:, 2], cmap='RdYlGn', s=5)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)

plt.sca(ax3)
plt.scatter(xtest[:, 0], xtest[:, 1], c=ypridict, cmap='RdYlGn', s=5)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)

# 计算准确率
TFTable = torch.eq(ytest, ypridict)
# print(TFTable)
correct = 0
for aa in TFTable:
    if aa == True:
        correct += 1
print(f'correct number is {correct},accuracy is {correct/len(TFTable)*100}%')
plt.show()
