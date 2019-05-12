import numpy as np
from matplotlib import pyplot as plt

from matplotlib import pylab


params = {'legend.fontsize': 22,
          'axes.labelsize': 22,
          'axes.titlesize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22}
pylab.rcParams.update(params)


AdjMat = np.load('adjacency_matrix/ccMat_rowNorm.npy')
areas = np.array(['POR', 'LI', 'LM', 'V1', 'AL', 'RL', 'AM', 'PM', 'ACA'])

# plt.matshow(AdjMat)
out_degrees = np.nansum(AdjMat, axis = 1)
in_degrees  = np.nansum(AdjMat, axis = 0)
ratio = in_degrees/out_degrees

inds = np.array([3, 2, 4, 5, 7, 6])
print inds
print areas[inds]
print ratio[inds]
color_list = ['#E3CF57', 'green', 'purple', 'blue', 'brown', 'red']

fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(AdjMat[2:8][:,2:8], cmap = 'magma')
plt.xticks(np.arange(len(inds)), areas[2:8])
plt.yticks(np.arange(len(inds)), areas[2:8])
plt.xlabel('Target')
plt.ylabel('Source')
plt.subplot(1,2,2)
plt.scatter(np.arange(len(inds)), ratio[inds], color = color_list, lw = 10)
plt.xticks(np.arange(len(inds)), areas[inds])
plt.ylabel('in-degree/out-degree')
plt.tight_layout()
plt.savefig("adjacency_matrix/anatomical_degree_quantification.pdf", format = 'pdf', dpi = 1000)

plt.show()
