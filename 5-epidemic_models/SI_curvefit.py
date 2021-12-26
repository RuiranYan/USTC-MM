# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:10:50 2021

@author: lenovo
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from scipy.optimize import curve_fit
bound = 60000

data = pd.read_csv(r'data/time-series-19-covid-combined.csv')
# 在数据中找出中国湖北的数据
chinese_data = data[(data['Country/Region'] == 'China') & (data['Province/State'] == 'Hubei')]
chinese_data = chinese_data[chinese_data['Confirmed'] < bound]
chinese_data['Confirmed'].plot()
x = [i for i in range(len(chinese_data))]
y = (chinese_data['Confirmed']/59270000).tolist()

#拟合指定的函数func，参数为I,k
def func(t, I, k):
    return 1 / (1 + (1 / I - 1) * np.exp(-k * t))
# popt拟合得到的参数
popt, pcov = curve_fit(func,x,y,[500,1])
I0 = popt[0]
k0 = popt[1]
str_I0 = "{:.7f}".format(I0)
str_k0 = "{:.3f}".format(k0)
yval = func(np.array(x), I0, k0)
plt.cla()
plt.scatter(x, y)    # 输出原数据
plt.plot(x, yval, 'g', label='I0=%5.3f,k0=%5.3f' % tuple(popt)) # 输出拟合数据
plt.title("SI bound=" + str(bound) + " I0=" + str_I0 + " k0=" + str_k0)
plt.xlabel("time")
plt.ylabel("Infectious")
plt.show()