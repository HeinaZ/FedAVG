import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

alpha = 0.01
num_of_clients = 100
shard_size = 300

actual_pdf = [0.1] * 10
dirichlet_pdf = np.random.dirichlet([alpha / 10] * 10, num_of_clients)

shared_portion = math.exp(-alpha)

shared_size = shard_size*shared_portion/4
shard_size = shard_size*(1 - shared_portion/4)

for i in range(num_of_clients):
    local_pdf = np.floor(dirichlet_pdf[i] * shard_size + np.multiply(actual_pdf, shared_size)).astype('int64')
    pdf_list = []
    start_index = 0
    for k in range(10):
        pdf_list.append((start_index, local_pdf[k]))
        start_index += local_pdf[k]

    plt.broken_barh(pdf_list,
                (i*4, 3),
                facecolors=("b", "#FF7F50", "g", "r", "purple", "#8B4513", "#FFC0CB", "#808080", "#FFD700", "#00FFFF"),
                alpha=1)
plt.xlim((0, shard_size+shared_size-5))
plt.axis('off')
plt.savefig("./result/alpha_" + format(alpha) + '-' + format(time.time()) + ".png")
