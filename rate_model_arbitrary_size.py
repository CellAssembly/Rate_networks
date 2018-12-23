import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 12,
          'axes.labelsize': 14,
          'axes.titlesize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}
pylab.rcParams.update(params)



def simulate_rate_model(AdjMat, delayMat, num_nodes, tStart, tEnd, tStep, stim_times = ((5.0, 7.0), (15.0, 17.0)), stim_node = 0, labels = [], plot = True):

    AdjMat[delayMat < 0] = 0
    delayMat[delayMat < 0] = 0
    delay = 0.005
    tau = 0.05
    time = np.arange(tStart, tEnd, tStep)
    numSteps = len(time)
    beta = num_nodes


    pops = np.zeros((num_nodes,numSteps))

    for i, t in enumerate(time):
        if i == numSteps - 1:
            break

        if t >= stim_times[0][0] and t <=stim_times[0][1]:
            t_relative = (t - stim_times[0][0])
            pops[stim_node, i] =  pops[stim_node, i] + tau*t_relative*np.exp(-t_relative/tau)

        if t >= stim_times[1][0] and t <=stim_times[1][1]:
            t_relative = (t - stim_times[1][0])
            pops[3, i] = pops[3, i] + tau*t_relative*np.exp(-t_relative/tau)


        for j in range(num_nodes):
            pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)]) - pops[j, i] * beta)
            # Need to accound for delay mat and not use fixed delay
            # Can use the code below
            # pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delayMat[:,j]/tStep)]) - pops[j, i] * beta)



    if plot:
        plt.figure(figsize=(20,20))

        if len(labels) == 0:
            labels = range(num_nodes)

        for j in range(num_nodes):
            plt.plot(time, pops[j, :], lw = 3, label = labels[j])
        # plt.plot(time, pops[1, :], 'forestgreen', lw = 3, label = 'V1-d')
        # plt.plot(time, pops[2, :], 'darkorchid', lw = 3, label = 'HVA-s')
        # plt.plot(time, pops[3, :], 'goldenrod', lw = 3, label = 'HVA-d')
        plt.xlabel('Time (unitless)')
        plt.ylabel('Rate (unitless)')
        plt.legend()

    return np.sum(pops, axis=1)


tStart = 0
tStep  = 0.001
tEnd   = 25.0

scalar = 3.0
num_nodes = 18
delayMat = (np.random.random((num_nodes, num_nodes)) - 1)*0.01
AdjMat = np.random.random((num_nodes, num_nodes))# * 0.5
AdjMat[0, 1] += scalar
AdjMat[1, 2] += scalar
AdjMat[2, 3] += scalar
AdjMat[9, :] =  0
AdjMat[:, 9] =  0
AdjMat = AdjMat

######## Test case #############################################
plt.figure()
plt.imshow(AdjMat, interpolation='none')
plt.colorbar()
simulate_rate_model(AdjMat, delayMat, num_nodes, tStart, tEnd, tStep)
################################################################
# exit()

combined_CCGamp_all = np.load('first_7mice_combined_CCG_3layer_6std.npy')
combined_CCGamp = combined_CCGamp_all[0,:,:] / np.max(combined_CCGamp_all[0,:,:])
labels_CCGamp   = np.load('first_7mice_combined_CCG_3layer_6std_labels.npy')
combined_CCGamp = np.transpose(combined_CCGamp) #transpose so source is rows and target is columns

plt.figure()
plt.imshow(combined_CCGamp, interpolation='none')
plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.colorbar()


# tStart = 0
# tStep  = 0.001
# tEnd   = 12.0
# rates_intact = simulate_rate_model(combined_CCGamp, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=7, labels = labels_CCGamp)
#
# num_nodes = len(labels_CCGamp)
# ModulationInds = np.zeros((num_nodes, num_nodes))
# for i in range(num_nodes):
#     CCG_temp = np.copy(combined_CCGamp)
#     CCG_temp[i, :] =  0
#     CCG_temp[:, i] =  0
#     rates_perturbed = simulate_rate_model(CCG_temp, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=7, plot = False)
#     ModulationInds[:, i] = (rates_perturbed - rates_intact) / (rates_perturbed + rates_intact)
#
# plt.figure()
# ModulationInds[ModulationInds == -1] = np.nan
# plt.imshow(ModulationInds,
#            interpolation='none',
#            # cmap='seismic',
#            vmin=np.unique(ModulationInds)[1], vmax=0,
#            aspect='auto')
#
# plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
# plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
# plt.xlabel("Deleted node")
# plt.colorbar()
#
# # print labels_CCGamp
# #
# #
# # v1D_index = np.where(labels_CCGamp == 'V1-d')[0][0]
# # v1S_index = np.where(labels_CCGamp == 'V1-s')[0][0]
# #
# # hvaD_index = np.where(labels_CCGamp == 'AM-d')[0][0]
# # hvaS_index = np.where(labels_CCGamp == 'AM-s')[0][0]
# #
# # # Note that the deep had lower numbers than the superficial which is why there is a +1 for V1S and hvaS
# # a = combined_CCGamp[v1D_index:v1S_index + 1, v1D_index:v1S_index + 1]
# # b = combined_CCGamp[v1D_index:v1S_index + 1, hvaD_index:hvaS_index + 1]
# # c = combined_CCGamp[hvaD_index:hvaS_index + 1, v1D_index:v1S_index + 1]
# # d = combined_CCGamp[hvaD_index:hvaS_index + 1, hvaD_index:hvaS_index + 1]
# #
# # ### Will use transpose and flips to flip the matrix along the diagonals to have the superficial first and then
# # ### the deep. Run the code below with temp to see the effect.
# # #temp = np.array([[1,2], [3,4]])
# # # np.transpose(np.transpose(temp[::-1])[::-1])
# # a = np.transpose(np.transpose(a[::-1])[::-1])
# # b = np.transpose(np.transpose(b[::-1])[::-1])
# # c = np.transpose(np.transpose(c[::-1])[::-1])
# # d = np.transpose(np.transpose(d[::-1])[::-1])
# #
# #
# # matrix_subset = np.concatenate([np.concatenate([a,c]), np.concatenate([b,d])], axis = 1)*100000
# # matrix_subset = (scalar/np.max(matrix_subset)) * matrix_subset
# # np.fill_diagonal(matrix_subset, 0)
# # simulate_4node_rate_model(matrix_subset, tStart, tEnd, tStep)
#
#
# plt.show()
#
#
