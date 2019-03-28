import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 30,
          'axes.labelsize': 30,
          'axes.titlesize': 30,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30}
pylab.rcParams.update(params)


#simulate_rate_model(AdjMat,
#                   delayMat,
#                   num_nodes,
#                   tStart, tEnd, tStep,
#                   stim_times = ((5.0, 7.0), (15.0, 17.0)),
#                   stim_node = 0,
#                   labels = [], plot = True):

def simulate_4node_rate_model(AdjMat, tStart, tEnd, tStep, stim_node = 0, area = 'HVA'):

    delay = 1.
    time = np.arange(tStart, tEnd, tStep)
    numSteps = len(time)
    num_nodes = len(AdjMat)
    beta = num_nodes/2
    K_s = 0.5


    pops = np.zeros((num_nodes, numSteps))

    for i, t in enumerate(time):
        if i == numSteps - 1:
            break

        if t >= 2.0 and t <=4.:
            t_relative = (t - 2.0)
            pops[stim_node, i] =  pops[stim_node, i] + 0.1*t_relative*np.exp(-t_relative/0.1)

        if t >= 15.0 and t <=17.:
            t_relative = (t - 15.0)
            pops[3, i] = pops[3, i] + 0.1*t_relative*np.exp(-t_relative/0.1)

        # One line for loop replacement
        # From rate_model_arbitrary_size
        for j in range(len(AdjMat)):
            # No normalization - regular leak
            pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)]) - pops[j, i] * beta)

            # Normalization (Divisive) - Ohshiro et. al, Nat. Neuro 2011
            # pops[j, i+1] = pops[j, i] + tStep * ((np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)])))/(1 + np.sum(pops[j, i-int(delay/tStep)]))# - pops[j, i] * beta)
            # pops[j, i+1] = pops[j, i] + tStep * (np.sum(AdjMat[:,j]*pops[:,i -int(delay/tStep)])/(1 + np.sum(pops[j, i-int(delay/tStep)])) - pops[j, i] * beta)

            # Consider linear divisive normalization (March 2nd Cosyne poster)
            # Ri = ... divide by (1 + w*sum(Rj))
            # pops[j, i+1] = pops[j, i] + tStep *



    plt.figure(figsize=(20,20))
    plt.plot(time, pops[0, :], 'r', lw = 3, label='V1-s')
    plt.plot(time, pops[1, :], 'forestgreen', lw = 3, label = 'V1-d')
    plt.plot(time, pops[2, :], 'darkorchid', lw = 3, label = area + '-s')
    plt.plot(time, pops[3, :], 'goldenrod', lw = 3, label = area + '-d')
    plt.xlabel('Time (arb. u.)')
    plt.ylabel('Relative Rate (arb. u.)')
    plt.legend()


tStart = 0
tStep  = 0.001
tEnd   = 10.0

scalar = 1.0
AdjMat = np.ones((4,4)) * 0.25
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



combined_CCGamp = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std.npy')
labels_CCGamp   = np.load('adjacency_matrix/first_10mice_combined_CCG_2layer_5std_labels.npy')
combined_CCGamp = np.transpose(combined_CCGamp[0,:,:]) #transpose so source is rows and target is columns

plt.figure()
plt.imshow(combined_CCGamp, interpolation='none')
plt.xticks(range(len(labels_CCGamp)), list(labels_CCGamp), rotation = 90.0)
plt.yticks(range(len(labels_CCGamp)), list(labels_CCGamp))
plt.colorbar()

print labels_CCGamp


v1D_index = np.where(labels_CCGamp == 'V1-d')[0][0]
v1S_index = np.where(labels_CCGamp == 'V1-s')[0][0]

area = 'LM'
hvaD_index = np.where(labels_CCGamp == area + '-d')[0][0]
hvaS_index = np.where(labels_CCGamp == area + '-s')[0][0]

# Note that the deep had lower numbers than the superficial which is why there is a +1 for V1S and hvaS
a = combined_CCGamp[v1D_index:v1S_index + 1, v1D_index:v1S_index + 1]
b = combined_CCGamp[v1D_index:v1S_index + 1, hvaD_index:hvaS_index + 1]
c = combined_CCGamp[hvaD_index:hvaS_index + 1, v1D_index:v1S_index + 1]
d = combined_CCGamp[hvaD_index:hvaS_index + 1, hvaD_index:hvaS_index + 1]

### Will use transpose and flips to flip the matrix along the diagonals to have the superficial first and then
### the deep. Run the code below with temp to see the effect.
#temp = np.array([[1,2], [3,4]])
# np.transpose(np.transpose(temp[::-1])[::-1])
a = np.transpose(np.transpose(a[::-1])[::-1])
b = np.transpose(np.transpose(b[::-1])[::-1])
c = np.transpose(np.transpose(c[::-1])[::-1])
d = np.transpose(np.transpose(d[::-1])[::-1])


matrix_subset = np.concatenate([np.concatenate([a,c]), np.concatenate([b,d])], axis = 1)
np.fill_diagonal(matrix_subset, 0)
matrix_subset = (scalar/np.max(matrix_subset)) * matrix_subset


plt.figure()
plt.imshow(matrix_subset, interpolation='none')
plt.colorbar()
simulate_4node_rate_model(matrix_subset, tStart, tEnd, tStep, area = area)
# simulate_4node_rate_model(matrix_subset, tStart, tEnd, tStep, stim_node=3)

plt.tight_layout()
plt.savefig("adjacency_matrix/area_" + area + ".pdf", format = 'pdf', dpi = 1000)
plt.show()

#### Plot Alpha Function ####
plt.figure()
x = np.arange(0,5,0.001)
y = 0.5*x*np.exp(-x/0.5)
plt.plot(x,y, lw = 10.0, color = 'royalblue')
plt.show()

