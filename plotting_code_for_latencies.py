import numpy as np
from matplotlib import pyplot as plt


################################################## 2-layer  ##################################################
labels_CCGamp   = np.load('first_10mice_combined_CCG_2layer_5std_labels.npy')
ttr_intact    = np.load('ttr_2layer.npy')
plotting_order = np.array([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
label_order = np.array([4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3])
color_list = ['r', 'r', 'brown', 'brown', '#E3CF57', '#E3CF57', 'green', 'green', 'purple', 'purple', 'blue', 'blue']



################################################## 3-layer  ##################################################
labels_CCGamp = np.load('first_10mice_combined_CCG_3layer_5std_labels.npy')
ttr_intact    = np.load('ttr.npy')
plotting_order = np.array([12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
label_order = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5])
color_list = ['r', 'r', 'r', 'brown', 'brown', 'brown', '#E3CF57', '#E3CF57', '#E3CF57', 'green', 'green', 'green',
              'purple', 'purple', 'purple', 'blue', 'blue', 'blue']



num_nodes = len(labels_CCGamp)


plt.figure(figsize=(8, 6))
plt.scatter(plotting_order, ttr_intact, c=color_list)
plt.xticks(np.arange(num_nodes), labels_CCGamp[label_order], rotation=90)
plt.xlabel("Area")
plt.ylabel("Time-to-Respond")
plt.title('Stimulated area: V1m')
plt.tight_layout()
# plt.show()
