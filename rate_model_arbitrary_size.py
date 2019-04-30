import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 20,
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20}
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
        # AdjMat[delayMat < 0] = 0
        delayMat[delayMat < 0] = 0.005
    delay = 1.0
    tau = 0.1
    time = np.arange(tStart, tEnd, tStep)
    numSteps = len(time)
    beta = num_nodes/2   #1 over tau of biophysical leak current of neurons
    K_s = 1./1


    pops = np.zeros((num_nodes,numSteps))

    for i, t in enumerate(time):
        if i == numSteps - 1:
            break

        # temp = pops[stim_node, i]
        # pops[:, i] = pops[:, i] + 0.01*np.random.random(num_nodes)
        if t >= stim_times[0][0] and t <=stim_times[0][1]:
            t_relative = (t - stim_times[0][0])
            # pops[stim_node, i] = temp
            pops[stim_node, i] =  pops[stim_node, i] + tau*t_relative*np.exp(-t_relative/tau)
            # pops[stim_node, i] =  0.5


            # For a grating stimulus
            f = 2    # Frequency in Hz
            # pops[stim_node, i] =  pops[stim_node, i] + np.sin(2*np.pi*f*t_relative)


        if t >= stim_times[1][0] and t <=stim_times[1][1]:
            t_relative = (t - stim_times[1][0])
            pops[3, i] = pops[3, i] + tau*t_relative*np.exp(-t_relative/tau)

        for j in range(num_nodes):
            # For when there is a single delay to be used holistically.
            # No normalization
            pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)]) - pops[j, i] * beta)

            # Normalization (Divisive) - Ohshiro et. al, Nat. Neuro 2011
            # pops[j, i + 1] = (pops[j, i] + tStep * (np.sum(AdjMat[:, j] * pops[:, i - int(delay / tStep)]))) / (1 + np.mean(pops[j, i - int(delay / tStep)]))
            # pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)])/(1 + np.sum(pops[j, i-int(delay/tStep)])) - pops[j, i] * beta)



            # Normalization (linear) - Issa et. al, eLife 2018
            # activity_all_other_nodes = np.copy(pops[:, i - int(delay / tStep)])
            # activity_all_other_nodes[j] = 0
            # activity_all_other_nodes = np.sum(activity_all_other_nodes)
            # pops[j, i + 1] = pops[j, i] + tStep * (np.sum(AdjMat[:, j] * pops[:, i - int(delay / tStep)]) - pops[j, i] * beta - K_s*activity_all_other_nodes)

            # Normalization (nopnlinear) - Issa et. al, eLife 2018
            # activity_all_other_nodes = np.sum(pops[:, i - int(delay / tStep)]) - pops[j, i - int(delay / tStep)]
            # pops[j, i + 1] = pops[j, i] + tStep * (np.sum(AdjMat[:, j] * pops[:, i - int(delay / tStep)]) - pops[j, i] * beta * (1/np.sqrt(1 - K_s*activity_all_other_nodes)))

            # For accounting for delay mat and not use fixed delay
            # pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[range(num_nodes),i -(delayMat[:,j]/tStep).astype(int)]) - pops[j, i] * beta)

            # Consider linear divisive normalization (March 2nd Cosyne poster)
            # Ri = ... divide by (1 + w*sum(Rj))
            # pops[j, i+1] = pops[j, i] + tStep *



    if plot:
        plt.figure(figsize=(16,16))

        if len(labels) == 0:
            labels = range(num_nodes)

        for j in range(num_nodes):
            plt.plot(time, pops[j, :], lw = 3, label = labels[j])
        plt.xlabel('Time (arb. u.)')
        plt.ylabel('Rate (arb. u.)')
        plt.legend()

    min_amplitude = np.min(np.max(pops, axis = 1))  #0.005052069650848851 (cluster 2), 0.0007585100825505153 (cluster 3)
    response_time = np.zeros(num_nodes)
    for i in range(num_nodes):
        try:
            response_time[i] = np.where(pops[i,:] > min_amplitude*0.5)[0][0]
        except IndexError:
            response_time[i] = 0.0
    print min_amplitude
    # return (np.sum(pops, axis=1), (np.argmax(pops, axis = 1)*tStep - stim_times[0][0]))
    # np.save('psth_cluster3.npy', pops)
    return (np.sum(pops, axis=1), response_time*tStep - stim_times[0][0])



tStart = 0
tStep  = 0.001
tEnd   = 14.0

scalar = 2.0
num_nodes = 18
delayMat = (np.random.random((num_nodes, num_nodes)) - 0.5)*0.01
AdjMat = np.random.random((num_nodes, num_nodes)) * 0.5
AdjMat[0, 1] += scalar
AdjMat[1, 2] += scalar
AdjMat[2, 3] += scalar
delayMat[0, 1] = 0.005
delayMat[1, 2] = 0.005
delayMat[2, 3] = 0.005
# AdjMat[9, :] =  0
# AdjMat[:, 9] =  0
AdjMat = ( AdjMat /np.mean(AdjMat) )
AdjMat[1, 0] += scalar*2


######## Test case #############################################
plt.figure()
plt.imshow(AdjMat, interpolation='none')
plt.colorbar()
simulate_rate_model(AdjMat, delayMat, num_nodes, tStart, tEnd, tStep, include_delayMat = False)
################################################################


combined_CCGamp_all = np.load('adjacency_matrix/first_10mice_combined_CCG_3layer_5std.npy')
labels_CCGamp   = np.load('adjacency_matrix/first_10mice_combined_CCG_3layer_5std_labels.npy')
combined_CCGamp_all = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std.npy')
labels_CCGamp   = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_labels.npy')

combined_CCGamp_all = np.load('first_7mice_combined_CCG_3layer_6std.npy')
labels_CCGamp   = np.load('first_7mice_combined_CCG_3layer_6std_labels.npy')

combined_CCGamp_all = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_2cluster.npy')
labels_CCGamp   = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_2cluster_labels.npy')
combined_CCGamp_all = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_3cluster.npy')
labels_CCGamp   = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_3cluster_labels.npy')
combined_CCGamp = combined_CCGamp_all[0,:,:]

###### Remove PM and RL ######
# labels_CCGamp = np.delete(labels_CCGamp, [3,4,5,15,16,17])
# combined_CCGamp = np.delete(combined_CCGamp, [3,4,5,15,16,17], 0)
# combined_CCGamp = np.delete(combined_CCGamp, [3,4,5,15,16,17], 1)
##############################

combined_CCGamp = combined_CCGamp / np.max(combined_CCGamp)
combined_CCGamp = np.transpose(combined_CCGamp) #transpose so source is rows and target is columns
np.fill_diagonal(combined_CCGamp, 0)

delayMat = combined_CCGamp_all[1] / 1000.       # To convert to seconds
delayMat = np.transpose(delayMat)

plt.figure()
plt.imshow(combined_CCGamp, interpolation='none')
plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.colorbar()

plt.figure()
plt.imshow(delayMat, interpolation='none', cmap = 'bwr')
plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.colorbar()


tStart = 0
tStep  = 0.001
tEnd   = 12.0

num_nodes = len(labels_CCGamp)
for stim_node in np.arange(5, 6, 1):
    print "Stimulation input pulse at: ", labels_CCGamp[stim_node]

    rates_intact, ttp_intact = simulate_rate_model(combined_CCGamp, delayMat, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=stim_node, labels = labels_CCGamp, plot = True)
    plt.figure(figsize = (8, 6))
    # temp_labels = np.delete(labels_CCGamp, stim_node)
    # ttp_intact = np.delete(t# Consider linear divisive normalization (March 2nd Cosyne poster)
            # Ri = ... divide by (1 + w*sum(Rj))
            # pops[j, i+1] = pops[j, i] + tStep *tp_intact, stim_node)
    # 3-layer
    # plotting_order = np.array([12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # label_order = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5])
    # color_list = ['r','r','r','brown','brown','brown', '#E3CF57', '#E3CF57', '#E3CF57', 'green','green','green','purple','purple','purple','blue','blue','blue']
    # 2-layer
    plotting_order = np.array([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    label_order = np.array([4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3])
    color_list = ['r','r','brown','brown', '#E3CF57', '#E3CF57', 'green','green','purple','purple','blue','blue']
    plt.scatter(plotting_order, ttp_intact, c = color_list)
    # plt.plot([0, num_nodes - 1], [0, 0], 'k')
    plt.xticks(np.arange(num_nodes), labels_CCGamp[label_order], rotation = 90)
    plt.xlabel("Area")
    plt.ylabel("Time-to-Respond")
    plt.title('Stimulated area: ' + labels_CCGamp[stim_node])
    plt.tight_layout()
    plt.savefig("adjacency_matrix/TimeToPeak_Stim_node_" + labels_CCGamp[stim_node] + "_regularNormalization.pdf", format = 'pdf', dpi = 1000)
    # plt.savefig("Stim_node_V1m_regularNormalization.png")
    ModulationInds = np.zeros((num_nodes, num_nodes))

    # for i in range(num_nodes):
    #     CCG_temp = np.copy(combined_CCGamp)
    #     CCG_temp[i, :] =  0
    #     CCG_temp[:, i] =  0
    #     rates_perturbed, ttp_perturbed = simulate_rate_model(CCG_temp, delayMat, num_nodes, tStart, tEnd, tStep, stim_times=((5.0, 7.0), (None, None)), stim_node=stim_node, plot = False)
    #     ModulationInds[:, i] = (rates_perturbed - rates_intact) / (rates_perturbed + rates_intact)

    plt.figure(figsize=(8,6))
    np.fill_diagonal(ModulationInds, np.nan)  # Set diagonals to NAN as meaningless to measure from deleted node
    ModulationInds[:, stim_node] = np.nan     # If you remove the stimulated node, then everything is -1 and so meaningless also
    # palette = plt.cm.bwr
    # palette.set_bad('black', 0.6)
    # ModulationInds = np.ma.array(ModulationInds, mask=np.isnan(ModulationInds))
    net_modulation = np.nanmean(ModulationInds, axis=0)
    net_modulation = np.delete(net_modulation, stim_node)
    # plt.scatter(plotting_order, net_modulation, c = color_list)
    # plt.plot([0, num_nodes - 1], [0, 0], 'k')
    # plt.xticks(np.arange(num_nodes - 1), temp_labels[label_order], rotation = 90)
    # plt.xlabel("Deleted Area")
    # plt.ylabel("Total Modulation")
    # plt.title('Stimulated Area: ' + labels_CCGamp[stim_node])
    # plt.ylim(-0.15, 0.005)

    # plt.imshow(ModulationInds,
    #            interpolation='none',
    #            cmap='bwr',
    #            # vmin=np.unique(ModulationInds)[0], vmax=-1*np.unique(ModulationInds)[0],
    #            vmin=-0.15, vmax=0.15,
    #            aspect='auto')

    # plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
    # plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
    # plt.xlabel("Deleted node")
    # plt.ylabel("Node Response")
    # plt.colorbar()
    plt.tight_layout()
    # plt.savefig("adjacency_matrix/Modulation_Stim_node_" + labels_CCGamp[stim_node] + "_regularNormalization.png")
    # plt.savefig("adjacency_matrix/Net_Modulation_Stim_node_" + labels_CCGamp[stim_node] + "_regularNormalization.png")
    plt.savefig("adjacency_matrix/Modulation_stim_" + labels_CCGamp[stim_node] + "_regularNormalization.pdf", format = 'pdf', dpi = 1000)








