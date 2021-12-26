# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:57:50 2021

@author: lenovo
"""

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
T = 150
a = [0 for i in range(10)]

def funcSEIR(inivalue, _, beta, N, Te, gamma):
    Y = np.zeros(4)
    X = inivalue
    # 易感个体变化(S)
    Y[0] = - (beta * X[0] * (X[2] + X[1])) / N
    # 潜伏个体变化(E)
    Y[1] = (beta * X[0] * (X[2] + X[1])) / N - X[1] / Te
    # 感染个体变化(I)
    Y[2] = X[1] / Te - gamma * X[2]
    # 治愈个体变化(R)
    Y[3] = gamma * X[2]
    return Y

def qiujie(t, beta, N, Te, gamma, INI):
    tspan = np.hstack([[0], np.hstack([t])])
    return spi.odeint(funcSEIR, INI, tspan, args=(beta, N, Te, gamma))[1:, 0]

popt, pcov = curve_fit(qiujie, a, a)    #popt拟合得到的参数
print(popt)