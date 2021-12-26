# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from scipy.optimize import curve_fit

data = pd.read_csv(r'data/countries-aggregated.csv')
chinese_data = data[data['Country'] == 'China']
print(chinese_data)
plt.plot(chinese_data['Confirmed'][chinese_data['Confirmed'] < 80000])
# chinese_data['Confirmed'][chinese_data['Confirmed']<80000].plot()
# 取中国疫情在达到80000之前的数据
x = [i for i in range(len(chinese_data[chinese_data['Confirmed'] < 80000]))]
y = (chinese_data['Confirmed'][chinese_data['Confirmed'] < 80000]/1400000000).tolist()

#指定拟合的函数func
def SI_func(t, I, k):
    return 1 / (1 + (1 / I - 1) * np.exp(-k * t))

# 进行拟合，popt为拟合后的参数
popt, pcov = curve_fit(SI_func, x, y, [500, 1])
I0 = popt[0]
k0 = popt[1]
yval = SI_func(np.array(x), I0, k0)
plt.cla()
plt.scatter(x, y)   # 输出原始数据
plt.plot(x, yval, 'g', label='I0=%5.3f,k0=%5.3f' % tuple(popt)) # 输出拟合后数据
plt.show()