import matplotlib.pylab as plt
import numpy as np
import math


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
        devi = [0] * 3
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

    def MMFunction1(self):
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)

                PColor = (PColor[0] + PColor[1] + PColor[2]) / 3

                self.SetPixelAt(x, y, PColor)

    def MMFunction2(self):
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)

                PColor = (0.299 * PColor[0] + 0.587 * PColor[1] + 0.114 * PColor[2])

                self.SetPixelAt(x, y, PColor)

    def MMFunction3(self):
        aver = [0] * 3
        aver = self.GetPixelAverage()
        im_temp = np.zeros((self.Image_Height * self.Image_Width, 3))
        i = 0
        # 去中心化
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                im_temp[i] = PColor - aver
                i = i + 1
        # U, SIGMA, VT = np.linalg.svd(im_temp)
        C = np.transpose(im_temp) @ im_temp

        eigenvalue, featurevector = np.linalg.eig(C)
        vecx = featurevector[0]

        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                PColor = PColor - aver
                PColor = np.vdot(PColor, vecx)
                PColor = PColor + aver
                self.SetPixelAt(x, y, PColor[0])

    def RGB2lab(self):
        RGB2LMS = np.array([[0.3881, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1228, 0.8444]])
        LogLMS2lab = np.diag([1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(2)]) @ np.array([[1, 1, 1],
                                                                                                 [1, 1, -2],
                                                                                                 [1, -1, 0]])
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                PColor = RGB2LMS @ PColor
                PColor = np.log10(PColor)
                PColor = LogLMS2lab @ PColor
                self.SetPixelAt(x, y, PColor)

    def lab2RGB(self):
        lab2LMS = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]]) @ np.diag(
            [1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(2)])
        LMS2RGB = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])
        for x in range(self.Image_Width):
            for y in range(self.Image_Height):
                PColor = self.GetPixelAt(x, y)
                PColor = lab2LMS @ PColor
                PColor = [10 ** PColor[0], 10 ** PColor[1], 10 ** PColor[2]]
                PColor = LMS2RGB @ PColor
                self.SetPixelAt(x, y, PColor)

    def SaveImage(self):
        plt.imshow(self.im, interpolation="none")
        plt.axis('off')  # 去掉坐标轴
        plt.savefig("3.png")

    im = []
    Image_Width = 0
    Image_Height = 0
    RGBLength = 3


#########################################################################
def color_transform(impath1, impath2):  # 将im2中的颜色风格转移到im1中
    im1 = PImage(impath1)
    im1.showImg()
    im2 = PImage(impath2)
    im2.showImg()
    im1.RGB2lab()
    im2.RGB2lab()
    im1aver = im1.GetPixelAverage()
    im2aver = im2.GetPixelAverage()
    im1devi = im1.GetPixelDeviation()
    im2devi = im2.GetPixelDeviation()
    for x in range(im1.Image_Width):
        for y in range(im1.Image_Height):
            PColor = im1.GetPixelAt(x, y)
            PColor = np.array(PColor) - np.array(im1aver)
            PColor = PColor * np.array(im2devi) / np.array(im1devi)
            PColor = np.array(PColor) + np.array(im2aver)
            im1.SetPixelAt(x, y, PColor)
    im1.lab2RGB()
    im2.lab2RGB()
    im1.showImg()

def colorGrey(impath1, impath2):  # 将im2中的颜色风格为im1上色
    im1 = PImage(impath1)
    im1.showImg()
    im2 = PImage(impath2)
    im2.showImg()
    im1aver = im1.GetPixelAverage()
    im2aver = im2.GetPixelAverage()
    im1devi = im1.GetPixelDeviation()
    im2devi = im2.GetPixelDeviation()
    for x in range(im1.Image_Width):
        for y in range(im1.Image_Height):
            PColor = im1.GetPixelAt(x, y)
            PColor = np.array(PColor) - np.array(im1aver)
            PColor = PColor * np.array(im2devi) / np.array(im1devi)
            PColor = np.array(PColor) + np.array(im2aver)
            im1.SetPixelAt(x, y, PColor)
    im1.showImg()

#测试
if __name__ == "__main__":
    MMImage = PImage("6.png")
    MMImage.showImg()
    MMImage.MMFunction3()
    MMImage.showImg()
    # color_transform("5.png", "2.png")
    # colorGrey("5.png", "2.png")
