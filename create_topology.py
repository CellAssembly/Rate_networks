import numpy as np
### SECOND MATLAB script to convert
#function[weightsEE, weightsEI, weightsIE, weightsII] = create_EI_topology(EneuronNum, numClusters, PARAMS)
# def create_FF_topology (neuronNum, numClusters, PARAMS_DICTRIONARY = None):
#
#     EneuronNum = 0.8 * neuronNum
#     IneuronNum = 0.2 * neuronNum
#
#     if PARAMS_DICTRIONARY == None:
#         PARAMS_DICTRIONARY = {}
#         PARAMS_DICTRIONARY['factorEI'] = 3
#         PARAMS_DICTRIONARY['factorIE'] = 3
#         PARAMS_DICTRIONARY['pfactorEI'] = 1.8
#         PARAMS_DICTRIONARY['pfactorIE'] = 1.8
# 
#         PARAMS_DICTRIONARY['wEI'] = 0.042
#         PARAMS_DICTRIONARY['pEI'] = .5
#
#         PARAMS_DICTRIONARY['wIE'] = 0.0105
#         PARAMS_DICTRIONARY['pIE'] = .5
#
#         PARAMS_DICTRIONARY['wEE'] = 0.022
#         PARAMS_DICTRIONARY['pEE'] = .2
#
#         PARAMS_DICTRIONARY['wII'] = 0.042
#         PARAMS_DICTRIONARY['pII'] = .5



###############################################################################
# CREATE WEIGHT MATRIX
###############################################################################

neuronNum = 1000
EneuronNum = 0.8 * neuronNum
IneuronNum = 0.2 * neuronNum

WRatio  = 1.0               #Ratio of Win/Wout (synaptic weight of within group to neurons outside of the group)
REE = 3.5                   #Ratio of pin/pout (probability of connection withing group to outside the group)
numClusters = 6

mult = 1
f = 1/np.sqrt(mult)       #Factor to scale by synaptic weight parameters by network size

wEI     = f*.042          #Average weight of inhibitroy to excitatory cells
wIE     = f*0.0105        #Average weight of excitatory to inhibitory cells
wEE     = f*.022          #Average weight of excitatory to excitatory cells
wII     = f*0.042         #Average weight of inhibitory to inhibitory cells



wEEsub = wEE/(1/numClusters+(1-1/numClusters)/WRatio);          #Average weight for sub-clusters
wEE    = wEEsub/WRatio;

p1  = 0.2/(1/numClusters+(1-1/numClusters)/REE);                #Average probability for sub-clusters
pEE = p1/REE;

weightsEI = np.random('binom',1,0.5,[EneuronNum,IneuronNum]);      #Weight matrix of inhibioty to excitatory LIF cells
weightsEI = wEI.* weightsEI;

weightsIE = np.random('binom',1,0.5,[IneuronNum, EneuronNum]);     #Weight matrix of excitatory to inhibitory cells
weightsIE = wIE.* weightsIE;

weightsII = np.random('binom',1,0.5,[IneuronNum, IneuronNum]);     #Weight matrix of inhibitory to inhibitory cells
weightsII = wII.* weightsII;

weightsEE = np.random('binom',1,pEE,[EneuronNum, EneuronNum]);     #Weight matrix of excitatory to excitatory cells
weightsEE = wEE.* weightsEE;


#Create the group weight matrices and update the total weight matrix
for i in range (numClusters):
    weightsEEsub = random('binom',1,p1,[EneuronNum/numClusters, EneuronNum/numClusters]);
    weightsEEsub = wEEsub.* weightsEEsub;
    weightsEE((i-1)*EneuronNum/numClusters+1:i*EneuronNum/numClusters,(i-1)*EneuronNum/numClusters+1:i*EneuronNum/numClusters) = weightsEEsub;


#Ensure the diagonals are zero
weightsII = weightsII - np.diag(np.diag(weightsII));
weightsEE = weightsEE - np.diag(np.diag(weightsEE));