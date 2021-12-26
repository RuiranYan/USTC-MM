import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 文件路径：
####################################
# insets文件由于数据错误已删除
# dataset = './insets/insets2-training.txt'
# dataset = './insets/insets-training.txt'
# testset = './insets/insets-testing.txt'
# testset = './insets/insets-2-testing.txt'
####################################

dataset = './data2/Standard2.txt'
testset = './data2/Standard2_test.txt'
# dataset = './data2/Noise2.txt'
# testset = './data2/Noise2_test.txt'
# dataset = './points/points-training.txt'
# testset = './points/points-testing.txt'
# dataset = './points/points-2-training.txt'
# testset = './points/points-2-testing.txt'


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # .Liner()，“直”（线性）部分，相当与y=f(Wx)中Wx
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        # self.hidden2 = torch.nn.Sequential(torch.nn.Linear(20,20))
        # self.hidden3 = torch.nn.Sequential(torch.nn.Linear(20,20))
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # 激励函数
        # x = F.dropout(x,p=0.5)
        # x = F.relu(self.hidden2(x))
        # x = F.dropout(x,p=0.5)
        # x = F.relu(self.hidden3(x))
        # x = F.dropout(x, p=0.5)
        # y = self.predict(x)  # 输出值
        y = F.softmax(self.predict(x), dim=-1)  # 输出值
        return y


net = Net(n_feature=2, n_hidden=10, n_output=3)
# print(net)
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()

# 获取数据


data = np.loadtxt(dataset, dtype=np.float)
# print(data)


# 学习部分==============================
plt.ion()  # 动态学习过程展示
plt.show()

x = data[:, 0:2]
x = torch.from_numpy(x).type(torch.FloatTensor)
y = data[:, 2:].astype(int).T
y = torch.from_numpy(y).type(torch.LongTensor)[0]


for t in range(1000):
    out = net(x)  # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y)  # 计算两者的误差

    # print(loss)

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 10 == 0:
        print(loss)
        plt.cla()
        prediction = torch.max(out, 1)[1]
        plt.scatter(x[:, 0], x[:, 1], c=prediction, lw=0, cmap='RdYlGn', s=15)
        # accuracy = sum((prediction == y).numpy()) / 200.  # 预测中有多少和真实值一样
        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


# 测试过程
testdata = np.loadtxt(testset, dtype=np.float)
plt.figure(1)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
plt.sca(ax1)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='RdYlGn', s=10)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)

plt.sca(ax2)

plt.scatter(testdata[:, 0], testdata[:, 1],
            c=testdata[:, 2], cmap='RdYlGn', s=10)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)


xtest = testdata[:, 0:2]
xtest = torch.from_numpy(xtest).type(torch.FloatTensor)
ytest = testdata[:, 2:].astype(int).T
ytest = torch.from_numpy(ytest).type(torch.LongTensor)[0]


testout = net(xtest)
# loss = loss_func(testout, ytest)
# print(testout)
# print(ytest)
# print(loss)
ypridict = []
for result in testout:
    ypridict.append(torch.argmax(result))
# print(ypridict)
ypridict = torch.tensor(ypridict)
plt.sca(ax3)

plt.scatter(xtest[:, 0], xtest[:, 1], c=ypridict, cmap='RdYlGn', s=10)
plt.ylim(0, 2.5)
plt.xlim(0, 2.5)
aaa = torch.eq(ytest, ypridict)
# print(aaa)
correct = 0
for aa in aaa:
    if aa == True:
        correct += 1
print(f'correct number is {correct},accuracy is {correct/len(aaa)*100}%')
plt.show()
