import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
num_of_clients = 100

actual_pdf = [0.1] * 10
alpha_list = [j * 0.01 + 0.01 for j in range(500)]

EMD_list = []
for alpha in alpha_list:
    EMD = 0
    dirichlet_pdf = np.random.dirichlet([alpha / 10] * 10, num_of_clients)
    for i in range(num_of_clients):
        EMD += np.sum(np.square(np.subtract(dirichlet_pdf[i], actual_pdf))) ** 0.5
    EMD /= num_of_clients
    EMD_list.append(EMD)

y = [0]*500
for i in range(500):
    y[i] = 1/np.sqrt(13*alpha_list[i]/10+1)

#plt.plot(alpha_list, y, label=r'$y=\frac{1}{\sqrt{13\alpha/10}}$')
plt.plot(alpha_list, EMD_list, label=r'mapping curve')
plt.ylim((0, 1))
plt.xlim((0, 5))
plt.legend()

plt.ylabel("EMD")
plt.xlabel(r'$\alpha$ - Dirichlet concentration parameter')
plt.savefig("./result/EMD_equition-" + format(time.time()) + ".png")
