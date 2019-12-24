import numpy as np
from matplotlib.pyplot import plot as plt
# Create LIF network
# Make it feedforwad

# Below is MATLAB code that needs to be converted
def simulate_LIF_network(weightsEE, weightsEI, weightsIE, weightsII,tEnd = 1000):
    '''
    :param weightsEE:
    :param weightsEI:
    :param weightsIE:
    :param weightsII:
    :param tEnd: length of simulation (in millisecond)
    :return: spike times and neuron numbers
    '''

    ################################################################################
    # Simulation time parameters
    ################################################################################
    tStart = 0
    tStep = 0.1

    ################################################################################
    # Setting Parameters For Neuron Number
    ################################################################################
    #Number of excitatory neurons in the network
    EneuronNum    = size(weightsEE,1);
    #Number of inhibitory neurons in the network
    IneuronNum    = size(weightsII,1);
    #Total number of neurons
    neuronNum = EneuronNum + IneuronNum;



    ################################################################################
    # Parameters for the LIF neurons
    ################################################################################
    Vthres = 1;      %Threshold voltage for both exc and inh neurons
    Vreset = 0;      %Reset voltage for both exc and inh neurons

    %Excitatory neuron params
    Etm = 15;        %Membrane Time Constant
    Etr = 5;       %Refractory period

    %Inhibitory neuron params
    Itm = 10;        %Membrane Time Constant
    Itr = 5;       %Refractory period

    ################################################################################
    ## Solving Network with LIF neurons
    ################################################################################
    #time constant for excitatory to excitatory synapses
    t_EE = 3;
    #time constant for excitatory to inhibitory synapses
    t_IE = 3;
    #time constant for inhibitory to excitatory synapses
    t_EI = 2;
    t_II = 2;

    #conductance for excitatory to excitatory synapses
    gEE = np.zeros(1,EneuronNum);
    #conductance for excitatory to inhibitory synapses
    gIE = np.zeros(1,IneuronNum);
    #conductance for inhibitory to excitatory synapses
    gEI = np.zeros(1,EneuronNum);
    #conductance for inhibitory to inhibitory synapses
    gII = np.zeros(1,IneuronNum);

    gBE = 1.1 + .1*rand(1,EneuronNum);
    gBI = 1 + .05* rand(1,IneuronNum);

    #Matrix storing spike times for raster plots
    rast = zeros(neuronNum,(tEnd - tStart)/tStep + 1);

    #last action potential for refractor period calculation (just big number)
    lastAP  = -50 * ones(1,neuronNum);

    #inital membrane voltage is random
    memVol = rand(neuronNum,(tEnd - tStart)/tStep + 1);

    # np.random.seed(292892)

    for i  in range(0, (tEnd - tStart)/tStep):
        for j in range (neuronNum):

            ####################################
            #CONNCECTIVITY CALCULATIONS
            #################################

            if (j <= EneuronNum):
                gEE(j) = gEE(j) - gEE(j)*tStep/t_EE;
                gEI(j) = gEI(j) - gEI(j)*tStep/t_EI;
            else:
                gIE(j-EneuronNum) = gIE(j-EneuronNum) - gIE(j-EneuronNum)*tStep/t_IE;
                gII(j-EneuronNum) = gII(j-EneuronNum) - gII(j-EneuronNum)*tStep/t_II;


            #If excitatory neuron fired
            if (rast(j,i-1) ~= 0 && j <= EneuronNum):
                gEE = gEE + weightsEE(:,j)';
                gIE = gIE + weightsIE(:,j)';


            if (rast(j,i-1) ~= 0 && j > EneuronNum):
                gEI = gEI + weightsEI(:,j-EneuronNum)';
                gII = gII + weightsII(:,j-EneuronNum)';


            #################################
            #EXCITATORY NEURONS
            #################################

            if(j <= EneuronNum)

                gE= gEE(j);
                gI= gEI(j);


                v = memVol(j,i-1) + (tStep/Etm)*(-memVol(j,i-1) + gBE(j)) + tStep*(gE -gI);

                %Refractory Period
                if ((lastAP(j) + Etr/tStep)>=i)
                    v = Vreset;
                end

                %Fire if exceed threshold
                if (v > Vthres)
                    v = Vthres;
                    lastAP(j) = i;
                    rast(j,i) = j;
                end

                memVol(j,i) = v;
            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %INHIBITORY NEURONS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if (j > EneuronNum)
                gE = gIE(j - EneuronNum);
                gI = gII(j - EneuronNum);


                v = memVol(j,i-1) + (tStep/Itm)*(-memVol(j,i-1) + gBI(j-EneuronNum)) + tStep*(gE -gI);

                %Refractory Period
                if ((lastAP(j) + Itr/tStep)>=i)
                    v = Vreset;
                end

                %Fire if exceed threshold
                if (v > Vthres)
                    v = Vthres;
                    lastAP(j) = i;
                    rast(j,i) = j;
                end

                memVol(j,i) = v;
            end

        end
    end

%end of function
end