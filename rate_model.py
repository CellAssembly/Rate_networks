import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 30,
          'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30}
pylab.rcParams.update(params)



def simulate_4node_rate_model(AdjMat, tStart, tEnd, tStep):

    delay = 0.5
    time = np.arange(tStart, tEnd, tStep)
    numSteps = len(time)
    beta = 4.0


    pops = np.zeros((4,numSteps))

    for i, t in enumerate(time):
        if i == numSteps - 1:
            break

        if t >= 5.0 and t <=7.0:
            t_relative = (t - 5.0)
            pops[0, i] =  pops[0, i] + 0.1*t_relative*np.exp(-t_relative/0.2)

        if t >= 15.0 and t <=17.0:
            t_relative = (t - 15.0)
            pops[3, i] = pops[3, i] + 0.1*t_relative*np.exp(-t_relative/0.2)

        pops[0, i+1] = pops[0, i] + tStep * (np.sum(AdjMat[:,0]*pops[:,i -int(delay/tStep)]) - pops[0, i] * beta)
        pops[1, i+1] = pops[1, i] + tStep * (np.sum(AdjMat[:,1]*pops[:,i- int(delay/tStep)]) - pops[1, i] * beta)
        pops[2, i+1] = pops[2, i] + tStep * (np.sum(AdjMat[:,2]*pops[:,i- int(delay/tStep)]) - pops[2, i] * beta)
        pops[3, i+1] = pops[3, i] + tStep * (np.sum(AdjMat[:,3]*pops[:,i- int(delay/tStep)]) - pops[3, i] * beta)


    plt.figure(figsize=(20,20))
    plt.plot(time, pops[0, :], 'r', lw = 3, label='V1-s')
    plt.plot(time, pops[1, :], 'forestgreen', lw = 3, label = 'V1-d')
    plt.plot(time, pops[2, :], 'darkorchid', lw = 3, label = 'HVA-s')
    plt.plot(time, pops[3, :], 'goldenrod', lw = 3, label = 'HVA-d')
    plt.xlabel('Time (unitless)')
    plt.ylabel('Rate (unitless)')
    plt.legend()


tStart = 0
tStep  = 0.001
tEnd   = 25.0

scalar = 3.0
AdjMat = np.ones((4,4)) * 0.5
AdjMat[0, 1] += scalar
AdjMat[1, 2] += scalar
AdjMat[2, 3] += scalar
AdjMat = AdjMat

######## Test case #############################################
plt.figure()
plt.imshow(AdjMat, interpolation='none')
plt.colorbar()
simulate_4node_rate_model(AdjMat, tStart, tEnd, tStep)
################################################################

# combined_CCGamp = np.load('combined_CCGamp_2layer.npy')
# labels_CCGamp   = np.load('combined_CCGamp_2layer_labels.npy')
# combined_CCGamp = np.transpose(combined_CCGamp) #transpose so source is rows and target is columns
#
# plt.figure()
# plt.imshow(combined_CCGamp, interpolation='none')
# plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
# plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
# plt.colorbar()
#
# print labels_CCGamp
#
#
# v1D_index = np.where(labels_CCGamp == 'V1-d')[0][0]
# v1S_index = np.where(labels_CCGamp == 'V1-s')[0][0]
#
# hvaD_index = np.where(labels_CCGamp == 'AM-d')[0][0]
# hvaS_index = np.where(labels_CCGamp == 'AM-s')[0][0]
#
# # Note that the deep had lower numbers than the superficial which is why there is a +1 for V1S and hvaS
# a = combined_CCGamp[v1D_index:v1S_index + 1, v1D_index:v1S_index + 1]
# b = combined_CCGamp[v1D_index:v1S_index + 1, hvaD_index:hvaS_index + 1]
# c = combined_CCGamp[hvaD_index:hvaS_index + 1, v1D_index:v1S_index + 1]
# d = combined_CCGamp[hvaD_index:hvaS_index + 1, hvaD_index:hvaS_index + 1]
#
# ### Will use transpose and flips to flip the matrix along the diagonals to have the superficial first and then
# ### the deep. Run the code below with temp to see the effect.
# #temp = np.array([[1,2], [3,4]])
# # np.transpose(np.transpose(temp[::-1])[::-1])
# a = np.transpose(np.transpose(a[::-1])[::-1])
# b = np.transpose(np.transpose(b[::-1])[::-1])
# c = np.transpose(np.transpose(c[::-1])[::-1])
# d = np.transpose(np.transpose(d[::-1])[::-1])
#
#
# matrix_subset = np.concatenate([np.concatenate([a,c]), np.concatenate([b,d])], axis = 1)*100000
# matrix_subset = (scalar/np.max(matrix_subset)) * matrix_subset
# np.fill_diagonal(matrix_subset, 0)
# simulate_4node_rate_model(matrix_subset, tStart, tEnd, tStep)


plt.show()


