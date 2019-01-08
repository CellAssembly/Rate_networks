import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 12,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}
pylab.rcParams.update(params)



def simulate_rate_model(AdjMat, delayMat, num_nodes, tStart, tEnd, tStep, stim_times = ((5.0, 7.0), (15.0, 17.0)), stim_node = 0, include_delayMat = False, labels = [], plot = True):
    '''

    :param AdjMat: Input adjaceny matrix calculated from experimental data
    :param delayMat: Time delay between different regions
    :param num_nodes: Size of adjacency matrix also
    :param tStart: start time
    :param tEnd: end time
    :param tStep: timestep
    :param stim_times: Time when an input pulse is given to stim_node
    :param stim_node: Node that gets stimulated by a pulse
    :param labels: Labels for plotting
    :param plot: Flag to determine whether or not to plot time traces
    :return: Response trace of every population due to input perturbation
    '''

    if include_delayMat:
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
            # For when there is a single delay to be used holistically.
            # pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)]) - pops[j, i] * beta)
            # Need to accound for delay mat and not use fixed delay
            pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[range(num_nodes),i -(delayMat[:,j]/tStep).astype(int)]) - pops[j, i] * beta)



    if plot:
        plt.figure(figsize=(16,16))

        if len(labels) == 0:
            labels = range(num_nodes)

        for j in range(num_nodes):
            plt.plot(time, pops[j, :], lw = 3, label = labels[j])
        plt.xlabel('Time (unitless)')
        plt.ylabel('Rate (unitless)')
        plt.legend()

    return np.sum(pops, axis=1)


tStart = 0
tStep  = 0.001
tEnd   = 25.0

scalar = 3.0
num_nodes = 18
delayMat = (np.random.random((num_nodes, num_nodes)) - 0.5)*0.01
AdjMat = np.random.random((num_nodes, num_nodes))# * 0.5
AdjMat[0, 1] += scalar
AdjMat[1, 2] += scalar
AdjMat[2, 3] += scalar
delayMat[0, 1] = 0.005
delayMat[1, 2] = 0.005
delayMat[2, 3] = 0.005
AdjMat[9, :] =  0
AdjMat[:, 9] =  0
AdjMat = AdjMat

######## Test case #############################################
plt.figure()
plt.imshow(AdjMat, interpolation='none')
plt.colorbar()
simulate_rate_model(AdjMat, delayMat, num_nodes, tStart, tEnd, tStep, include_delayMat = True)
################################################################

combined_CCGamp_all = np.load('first_7mice_combined_CCG_3layer_6std.npy')
combined_CCGamp = combined_CCGamp_all[0,:,:] / np.max(combined_CCGamp_all[0,:,:])
labels_CCGamp   = np.load('first_7mice_combined_CCG_3layer_6std_labels.npy')
combined_CCGamp = np.transpose(combined_CCGamp) #transpose so source is rows and target is columns

plt.figure()
plt.imshow(combined_CCGamp, interpolation='none')
plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.colorbar()


tStart = 0
tStep  = 0.001
tEnd   = 12.0

stim_node = 7
print "Stimulation input pulse at: ", labels_CCGamp[stim_node]

rates_intact = simulate_rate_model(combined_CCGamp, delayMat, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=stim_node, labels = labels_CCGamp)

num_nodes = len(labels_CCGamp)
ModulationInds = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    CCG_temp = np.copy(combined_CCGamp)
    CCG_temp[i, :] =  0
    CCG_temp[:, i] =  0
    rates_perturbed = simulate_rate_model(CCG_temp, delayMat, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=stim_node, plot = False)
    ModulationInds[:, i] = (rates_perturbed - rates_intact) / (rates_perturbed + rates_intact)

plt.figure(figsize=(12,12))
ModulationInds[ModulationInds == -1] = np.nan
plt.imshow(ModulationInds,
           interpolation='none',
           # cmap='seismic',
           vmin=np.unique(ModulationInds)[1], vmax=0,
           aspect='auto')

plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.xlabel("Deleted node")
plt.colorbar()





