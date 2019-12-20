### SECOND MATLAB script to convert
function[weightsEE, weightsEI, weightsIE, weightsII] = create_EI_topology(EneuronNum, numClusters, PARAMS)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% WEIGHT
MATRIX
HERE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
if nargin < 1
    EneuronNum = 800;
end
IneuronNum = round(.25 * EneuronNum);

if nargin < 2
    numClusters = 10;
end

if nargin < 3
    PARAMS.factorEI = 3;
PARAMS.factorIE = 3;
PARAMS.pfactorEI = 1.8; % 2.5;
PARAMS.pfactorIE = 1.8; % 2.5;

PARAMS.wEI = 0.042; % .015
PARAMS.pEI = .5;

PARAMS.wIE = 0.0105; % .015
PARAMS.pIE = .5;

PARAMS.wEE = 0.022; % .015
PARAMS.pEE = .2;

PARAMS.wII = 0.042; % .057
PARAMS.pII = .5;

end

% calculation
of
adjusted
weights
for EI
    wEIsub = PARAMS.wEI / (1 / numClusters + (1 - 1 / numClusters) * PARAMS.factorEI);
PARAMS.wEI = wEIsub * PARAMS.factorEI;
pEIsub = PARAMS.pEI / (1 / numClusters + (1 - 1 / numClusters) * PARAMS.pfactorEI);
PARAMS.pEI = pEIsub * PARAMS.pfactorEI;

% calculation
of
adjusted
weights
for IE
    wIEsub = PARAMS.wIE / (1 / numClusters + (1 - 1 / numClusters) / PARAMS.factorIE);
PARAMS.wIE = wIEsub / PARAMS.factorIE;
pIEsub = PARAMS.pIE / (1 / numClusters + (1 - 1 / numClusters) / PARAMS.pfactorIE);
PARAMS.pIE = pIEsub / PARAMS.pfactorIE;

% Weight
matrix
of
inhibitory
to
excitatory
LIF
units
weightsEI = random('binom', 1, PARAMS.pEI, [EneuronNum, IneuronNum]);
weightsEI = PARAMS.wEI. * weightsEI;

% Weight
matrix
of
excitatory
to
inhibitory
cells
weightsIE = random('binom', 1, PARAMS.pIE, [IneuronNum, EneuronNum]);
weightsIE = PARAMS.wIE. * weightsIE;

weightsII = random('binom', 1, PARAMS.pII, [IneuronNum, IneuronNum]);
weightsII = PARAMS.wII. * weightsII;

weightsEE = random('binom', 1, PARAMS.pEE, [EneuronNum, EneuronNum]);
weightsEE = PARAMS.wEE. * weightsEE;

for i = 1:numClusters
weightsIEsub = random('binom', 1, pIEsub, [IneuronNum / numClusters, EneuronNum / numClusters]);
weightsIEsub = wIEsub. * weightsIEsub;
kk = (i - 1) * IneuronNum / numClusters + 1:i * IneuronNum / numClusters;
jj = (i - 1) * EneuronNum / numClusters + 1:i * EneuronNum / numClusters;
weightsIE(kk, jj) = weightsIEsub;
end

for i = 1:numClusters
weightsEIsub = random('binom', 1, pEIsub, [EneuronNum / numClusters, IneuronNum / numClusters]);
weightsEIsub = wEIsub. * weightsEIsub;
kk = (i - 1) * IneuronNum / numClusters + 1:i * IneuronNum / numClusters;
jj = (i - 1) * EneuronNum / numClusters + 1:i * EneuronNum / numClusters;
weightsEI(jj, kk) = weightsEIsub;
end
weightsEI = circshift(weightsEI, [0 - IneuronNum / numClusters]);

weightsEE = weightsEE - diag(diag(weightsEE));
weightsII = weightsII - diag(diag(weightsII));
end