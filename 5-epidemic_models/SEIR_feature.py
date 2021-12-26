import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
N = 60000000          # 湖北省为6000 0000
E_0 = 0
I_0 = 1
R_0 = 0
S_0 = N - E_0 - I_0 - R_0
beta1 = 0.78735     # 真实数据拟合得出
beta2 = 0.15747
Te = 14
sigma = 1 / Te      # 感染期的倒数
gamma = 0.154
alpha = 1           # 政府干预措施决定
T = 150

#ode求解
INI = [S_0, E_0, I_0, R_0]
def SEIR(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (alpha * beta1 * X[0] * X[2]) / N - (alpha * beta2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (alpha * beta1 * X[0] * X[2]) / N + (alpha * beta2 * X[0] * X[1]) / N - sigma * X[1]
    # I数量
    Y[2] = sigma * X[1] - gamma * X[2]
    # R数量
    Y[3] = gamma * X[2]
    return Y

T_range = np.arange(0, T + 1)
Res = spi.odeint(SEIR, INI, T_range)
S_t = Res[:, 0]
E_t = Res[:, 1]
I_t = Res[:, 2]
R_t = Res[:, 3]

plt.plot(S_t, color='blue', label='Susceptibles')   #, marker='.')
plt.plot(E_t, color='grey', label='Exposed')
plt.plot(I_t, color='red', label='Infected')
plt.plot(R_t, color='green', label='Recovered')
plt.xlabel('Day')
plt.ylabel('Number')
plt.title('SEIR Model(modified)_Te=' + str(Te))
plt.legend()
plt.show()