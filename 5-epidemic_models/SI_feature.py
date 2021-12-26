import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

I0 = 1 / 60000000
k = 0.1
t = [i for i in range(0, 300)]
I = np.zeros(300)
for t0 in t:
    I[t0] = 1 / (1 + (1 / I0 - 1) * np.exp(-k * t0))
plt.plot(t, I)
plt.title("SI k=0.1")
plt.xlabel("days")
plt.ylabel("Infectious")
plt.show()