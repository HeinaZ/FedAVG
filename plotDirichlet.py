import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

isIID = 0.06
num_of_clients = 100
shard_size = 300

dirichlet_pdf = np.random.dirichlet([isIID/10] * 10, num_of_clients)
actual_pdf = [0.1] * 10
for i in range(num_of_clients):
    local_pdf = np.floor(dirichlet_pdf[i] * 300).astype('int64')
    pdf_list = []
    start_index = 0
    for k in range(10):
        pdf_list.append((start_index, local_pdf[k]))
        start_index += local_pdf[k]

    plt.broken_barh(pdf_list,
                (i*4, 3),
                facecolors=("b", "#FF7F50", "g", "r", "purple", "#8B4513", "#FFC0CB", "#808080", "#FFD700", "#00FFFF"),
                alpha=1)
plt.xlim((0, 295))
plt.axis('off')
plt.savefig("./result/alpha_" + format(isIID) + '-' + format(time.time()) + ".png")
