import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
from sklearn.manifold import TSNE
from time import time


dataset = 'Citeseer'
search_mode = 'Zeng'
submanager_log_file = 'hop_8_gran_16.txt'
with open(dataset + "_" + search_mode + submanager_log_file, "r") as f:
    lines = f.readlines()

results = []
best_val_score = "0"
for line in lines:
    actions = line[:line.index(";")]
    val_score = line.split(";")[-1]

    results.append((actions, val_score))
results.sort(key=lambda x: x[-1], reverse=True)  # sort

# print(results)
C_dim = []
VAL = []
for c, val in results:
    c_s = c.split('[')
    c_s = c_s[1].split(']')
    c_s = c_s[0].split(',')
    c_num = [int(i) for i in c_s]

    C_dim.append(c_num)

    val_s = val.split('\n')
    val_num = val_s[0].split('.')
    val_num = int(val_num[1])
    VAL.append(val_num/100)
    print(val)
C_dim = np.array(C_dim)
best_structure = ""
best_score = 0



sns.set()
f,ax=plt.subplots()
sns.heatmap(C_dim, vmin=1, vmax=1024, cmap=None, center=1, robust=False, annot=False, fmt='f', annot_kws=None,
            linewidths=0.05, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False,
            xticklabels=False, yticklabels=False, mask=None, ax=None)  # 画热力图

plt.show()