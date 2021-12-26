import cv2
import numpy.matlib
import numpy as np
import matplotlib.pylab as plt
import math

# # 图片路径
# imageName = 'warp_test.png'
imageName = 'Monalisa.png'

u = 2  # 设置全局变量u
xold = []
yold = []
xnew = []
ynew = []


class PImage:
    def __init__(self, ImagePath):
        self.im = plt.imread(ImagePath)
        self.Image_Width, self.Image_Height, self.RGBLength = self.im.shape

    def showImg(self):
        plt.imshow(self.im, interpolation="none")
        plt.axis('off')  # 去掉坐标轴
        plt.show()  # 弹窗显示图像

    def GetWidth(self):
        return self.Image_Width

    def GetHeight(self):
        return self.Image_Height

    def GetPixelAt(self, x, y):
        return self.im[x, y][0:3]

    def SetPixelAt(self, x, y, c):
        self.im[x, y][0:3] = c

    def GetPixelAverage(self):
        aver = [0] * 3
        sum = [0] * 3
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                for i in range(3):
                    sum[i] = sum[i] + PColor[i]
        for i in range(3):
            aver[i] = sum[i] / (self.Image_Width * self.Image_Height)
        return aver

    def GetPixelDeviation(self):
        devi = [] * 3
        color0 = []
        color1 = []
        color2 = []
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                color0.append(PColor[0])
                color1.append(PColor[1])
                color2.append(PColor[2])
        devi[0] = np.std(color0)
        devi[1] = np.std(color1)
        devi[2] = np.std(color2)
        return devi

    im = []
    Image_Width = 0
    Image_Height = 0
    RGBLength = 3


# 鼠标选点
def getPoints():
    img = cv2.imread(imageName)

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global xold, yold, xnew, ynew
        if event == cv2.EVENT_LBUTTONDOWN:
            xold.append(x)
            yold.append(y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=5)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
            #             1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
            print(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            xy = "%d,%d" % (x, y)
            xnew.append(x)
            ynew.append(y)
            cv2.circle(img, (x, y), 1, (0, 255, 0), thickness=5)
            # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
            #             1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)
            print(x, y)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    keyval = cv2.waitKey(0)
    return keyval


def float2int(a):
    return list(map(int, a))


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def countSigma(x0, y0, x, y):
    global u
    sigma = []
    for i in range(len(x0)):
        dis = distance(x0[i], y0[i], x, y)
        if dis == 0:
            sigma.append(0.0)
        else:
            sigma.append(1 / (dis ** u))
    return sigma


def countOmega(sigma):
    sumSigma = sum(sigma)
    omega = []
    for x in sigma:
        omega.append(x / sumSigma)
    return omega


# IDW算法实现
def IDW(y0, x0, y1, x1):
    img = PImage(imageName)
    img2 = PImage(imageName)
    x0 = float2int(x0)
    y0 = float2int(y0)
    x1 = float2int(x1)
    y1 = float2int(y1)
    x0y0 = list(zip(x0, y0))
    delta_x = np.array(x1) - np.array(x0)
    delta_y = np.array(y1) - np.array(y0)
    for x in range(img2.Image_Width):
        for y in range(img2.Image_Height):
            img2.SetPixelAt(x, y, [1, 1, 1])
    flags = [[0 for col in range(img2.Image_Height)] for row in range(img2.Image_Width)]
    for x in range(img.Image_Width):
        for y in range(img.Image_Height):
            if (x, y) in x0y0:
                pcolor = img.GetPixelAt(x, y)
                xafter = x1[x0y0.index((x, y))]
                yafter = y1[x0y0.index((x, y))]
                flags[xafter][yafter] = 1
                img2.SetPixelAt(xafter, yafter, pcolor)
            else:
                pcolor = img.GetPixelAt(x, y)
                sigma = countSigma(x0, y0, x, y)
                omega = countOmega(sigma)
                xafter = x + int(np.array(omega) @ np.array(delta_x))
                yafter = y + int(np.array(omega) @ np.array(delta_y))
                if xafter in range(img.Image_Width) and yafter in range(img.Image_Height):
                    img2.SetPixelAt(xafter, yafter, pcolor)
                    flags[xafter][yafter] = 1
    for x in range(1, img2.Image_Width):
        for y in range(img2.Image_Height):
            # if all(img2.GetPixelAt(x, y) == [1, 1, 1]):
            if flags[x][y] == 0:
                pcolor = img2.GetPixelAt(x - 1, y)
                img2.SetPixelAt(x, y, pcolor)
    img2.showImg()


# RBF算法实现
def RBF(y0, x0, y1, x1):
    img = PImage(imageName)
    img2 = PImage(imageName)
    x0 = float2int(x0)
    y0 = float2int(y0)
    x1 = float2int(x1)
    y1 = float2int(y1)
    delta_x = np.array(x1) - np.array(x0)
    delta_y = np.array(y1) - np.array(y0)
    dis = np.matlib.empty((len(x0), len(x0)))
    phi = np.matlib.empty((len(x0), len(x0)))
    r = []
    for i in range(len(x0)):
        for j in range(len(x0)):
            dis[i, j] = distance(x0[i], y0[i], x0[j], y0[j])
    for i in range(len(x0)):
        r.append(min(dis[i, :].tolist()[0]))
    for i in range(len(x0)):
        for j in range(len(x0)):
            phi[i, j] = (dis[i, j] ** 2 + r[i] ** 2) ** (u / 2)
    wx = np.linalg.solve(np.mat(phi), np.mat(delta_x).T).T
    wy = np.linalg.solve(np.mat(phi), np.mat(delta_y).T).T
    for x in range(img2.Image_Width):
        for y in range(img2.Image_Height):
            img2.SetPixelAt(x, y, [1, 1, 1])
    flags = [[0 for col in range(img2.Image_Height)] for row in range(img2.Image_Width)]
    for x in range(img2.Image_Width):
        for y in range(img2.Image_Height):
            pcolor = img.GetPixelAt(x, y)
            phi2 = [0] * len(x0)
            for i in range(len(x0)):
                phi2[i] = (distance(x, y, x0[i], y0[i]) ** 2 + r[i] ** 2) ** (u / 2)
            xafter = x + int(wx @ phi2)
            yafter = y + int(wy @ phi2)
            if xafter in range(img.Image_Width) and yafter in range(img.Image_Height):
                img2.SetPixelAt(xafter, yafter, pcolor)
                flags[xafter][yafter] = 1
            phi2.clear()
    # for x in range(1, img2.Image_Width):
    #     for y in range(img2.Image_Height):
    #         # if all(img2.GetPixelAt(x, y) == [1, 1, 1]):
    #         if flags[x][y] == 0:
    #             pcolor = img2.GetPixelAt(x - 1, y)
    #             img2.SetPixelAt(x, y, pcolor)
    img2.showImg()


if __name__ == '__main__':
    if getPoints() == ord('i'):
        IDW(xold, yold, xnew, ynew)
    else:
        RBF(xold, yold, xnew, ynew)
