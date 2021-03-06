import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import math

# N为人群总数
N = 60000000
# d为死亡率
delta = 0.03 / 30
# r为政府措施干预，小于1
r1 = 0.3
r2 = 0.1
# β为传染率系数
beta1 = 0.78735 * r1  # 15*0.05249 = 0.78735
beta2 = 0.15747 * r2  # 3*0.05249 = 0.15747
# gamma为恢复率系数
gamma = 0.154
# Te为疾病潜伏期
Te = 10
# I_0为感染者的初始人数
I_0 = 1
# E_0为潜伏者的初始人数
E_0 = 0
# R_0为治愈者的初始人数
R_0 = 0
# D_0为死亡者的初始人数
D_0 = 0
# S_0为易感者的初始人数
S_0 = N - I_0 - E_0 - R_0 - D_0
# T为传播时间
T = 500

# INI为初始状态下的数组
INI = (S_0, E_0, I_0, R_0, D_0)

def funcSEIRD(inivalue, _):
 Y = np.zeros(5)
 X = inivalue
 # 易感个体变化
 Y[0] = - (beta1 * X[0] * X[1]) / N - (beta2 * X[0] * X[2]) / N
 # 潜伏个体变化
 Y[1] = (beta1 * X[0] * X[1]) / N + (beta2 * X[0] * X[2]) / N - X[1] / Te
 # 感染个体变化
 Y[2] = X[1] / Te - gamma * X[2] - delta * X[2]
 # 治愈个体变化
 Y[3] = gamma * X[2]
 # 死亡变化
 Y[4] = delta * X[2]
 return Y

T_range = np.arange(0, T + 1)

RES = spi.odeint(funcSEIRD, INI, T_range)

plt.plot(RES[:, 0], color='darkblue', label='Susceptible')
plt.plot(RES[:, 1], color='orange', label='Exposed')
plt.plot(RES[:, 2], color='red', label='Infection')
plt.plot(RES[:, 3], color='green', label='Recovery')
plt.plot(RES[:, 4], color='black', label='Death')

plt.title('SEIRD Model(Te=10)')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number')
plt.show()