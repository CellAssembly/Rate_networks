import numpy as np
from matplotlib import pyplot as plt


def create_matrices():
    ###############################################################################
    # CREATE WEIGHT MATRIX
    ###############################################################################

    neuronNum = 2000
    EneuronNum = 0.8 * neuronNum
    IneuronNum = 0.2 * neuronNum

    WRatio  = 1.0               #Ratio of Win/Wout (synaptic weight of within group to neurons outside of the group)
    REE = 2.0                   #Ratio of pin/pout (probability of connection withing group to outside the group)
    numClusters = 4

    mult = 1
    f = 1/np.sqrt(mult)       #Factor to scale by synaptic weight parameters by network size

    wEI     = f*.042          #Average weight of inhibitroy to excitatory cells
    wIE     = f*0.0105        #Average weight of excitatory to inhibitory cells
    wEE     = f*.011          #Average weight of excitatory to excitatory cells
    wII     = f*0.042         #Average weight of inhibitory to inhibitory cells



    wEEsub = wEE/(1/numClusters+(1-1/numClusters)/WRatio);          #Average weight for sub-clusters
    wEE    = wEEsub/WRatio;

    p1  = 0.2/(1/numClusters+(1-1/numClusters)/REE);                #Average probability for sub-clusters
    pEE = p1/REE;

    weightsEI = np.random.binomial(1, 0.5, (EneuronNum,IneuronNum));      #Weight matrix of inhibioty to excitatory LIF cells
    weightsEI = wEI* weightsEI;

    weightsIE = np.random.binomial(1,0.5,(IneuronNum, EneuronNum));     #Weight matrix of excitatory to inhibitory cells
    weightsIE = wIE* weightsIE;

    weightsII = np.random.binomial(1,0.5,(IneuronNum, IneuronNum));     #Weight matrix of inhibitory to inhibitory cells
    weightsII = wII* weightsII;

    weightsEE = np.random.binomial(1,pEE,(EneuronNum, EneuronNum));     #Weight matrix of excitatory to excitatory cells
    weightsEE = wEE* weightsEE;


    #Create the group weight matrices and update the total weight matrix
    for i in range (numClusters):
        weightsEEsub = np.random.binomial(1,p1,[EneuronNum/numClusters, EneuronNum/numClusters]);
        weightsEEsub = wEEsub* weightsEEsub
        weightsEE[i*EneuronNum/numClusters:(i+1)*EneuronNum/numClusters,i*EneuronNum/numClusters:(i+1)*EneuronNum/numClusters] = weightsEEsub


    #Ensure the diagonals are zero
    weightsII = weightsII - np.diag(np.diag(weightsII))
    weightsEE = weightsEE - np.diag(np.diag(weightsEE))

    # Plotting to check
    plt.imshow(weightsEE)
    plt.show()

    return (weightsEE, weightsEI, weightsIE, weightsII)