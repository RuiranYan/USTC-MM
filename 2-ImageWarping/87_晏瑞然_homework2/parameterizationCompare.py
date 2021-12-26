import matplotlib.pyplot as plt
import numpy as np
import math

exp = 10 #设定回归多项式次数

colordic = {
    'm': 'red',
    'd': 'purple',
    'c': 'green',
    'f': 'yellow'
}

worddic = {
    'm': 'mean',
    'd': 'distance',
    'c': 'center',
    'f': 'foley'
}

X = []
Y = []
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])


# 求两点距离
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 均匀参数化
def parameterization1(x, y):
    T = []
    for i in range(len(x)):
        T.append(i)
    return T


# 弦长参数化
def parameterization2(x, y):
    t = []
    temp = 0
    for i in range(len(x)):
        temp += distance(x[max(0, i - 1)], y[max(0, i - 1)], x[i], y[i])
        t.append(temp)
    return t


# 中心参数化
def parameterization3(x, y):
    t = []
    temp = 0
    for i in range(len(x)):
        temp += math.sqrt(distance(x[max(0, i - 1)], y[max(0, i - 1)], x[i], y[i]))
        t.append(temp)
    return t


# Foley参数化
def deltap(i, x, y):
    if i == -1 or i == len(x) - 1:
        return 0
    else:
        return distance(x[i], y[i], x[i + 1], y[i + 1])


def theta(i, x, y):
    if i == 0:
        return 0
    if i == len(x) - 1:
        return 0
    a = math.acos(
        np.dot([x[i + 1] - x[i], y[i + 1] - y[i]], [x[i] - x[i - 1], y[i] - y[i - 1]]) / deltap(i, x, y) / deltap(i - 1,
                                                                                                                  x, y))
    return min(math.pi - a, math.pi / 2)


def K(i, x, y):
    temp1 = deltap(i - 2, x, y) * theta(i - 1, x, y) / (deltap(i - 1, x, y) + deltap(i - 2, x, y))
    temp2 = deltap(i, x, y) * theta(i, x, y) / (deltap(i - 1, x, y) + deltap(i, x, y))
    return 1 + 3 / 2 * (temp1 + temp2)


def parameterization4(x, y):
    t = [0]
    temp = 0
    for i in range(1, len(x)):
        temp += K(i, x, y) * deltap(i - 1, x, y)
        t.append(temp)
    return t


def DrawExpCurve(event):
    e = event.key
    global X
    global Y
    if e == 'm':
        T = parameterization1(X, Y)
    elif e == 'd':
        T = parameterization2(X, Y)
    elif e == 'c':
        T = parameterization3(X, Y)
    else:
        T = parameterization4(X, Y)
    t = np.linspace(-0.1, T[len(T) - 1] + 0.1, 1000)
    px = np.polyfit(T, X, exp)
    px = np.poly1d(px)
    xvals = px(t)
    py = np.polyfit(T, Y, exp)
    py = np.poly1d(py)
    yvals = py(t)
    plt.plot(xvals, yvals, color=colordic[e], linewidth=2, label=e)
    plt.legend()

    fig.canvas.draw()


def onclick(event):
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       (event.button, event.x, event.y, event.xdata, event.ydata))
    print('%f %f' %
          (event.xdata, event.ydata))
    global X
    global Y
    X.append(event.xdata)
    Y.append(event.ydata)
    plt.plot(event.xdata, event.ydata, 'o', color='red')
    fig.canvas.draw()


def drawstreightline(event):
    plt.plot(X, Y, color='red')
    fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', DrawExpCurve)
plt.show()
