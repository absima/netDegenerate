import numpy as np
import itertools
    
netfold = '../01_preSimulation/ordered'
netfold = '/home/sima/projects/nest/n3611/02_currently/ordered'
qntfold = 'qntfold'

NI, NE = [ 680, 2931]
N0 = NI+NE
Inrn = np.arange(NI)
Enrn = np.arange(NI,N0)


all_network_types = [
    'emp',
    'erb',
    'sfo',
    'ero',
    'sfr',
    'swr'
]

nReal = 10
nNet = 6
nDegen = 2 
nIndex = 10 
nPrun = 5
nStage = 10

del_frac = 0.1

J_unit = 0.005
Je = 0.1
mije = Je/J_unit
p_rate = 3000.  # the rate of the poisson generator
J_bg = Je/J_unit # 20.83 #4.4



delay = 1.5

simulation_time = 11. # in s
start_record_time = 1. # in s
dt = 0.1 # time resolution in ms

duration_ms = int((simulation_time-start_record_time)*1000)+1

wk0 = np.array([-3., -3., 1.5, 1])

# scalezz = np.load('empLscale_rndBscale_2nets_10reals_20scales.npy')
empscales = np.arange(0.5, 10.5, 0.5)
scalers0 = np.load('scalers_6nets_10reals_20weights.npy')
scalers = scalers0[...,::2]
nWeight = scalers.shape[-1]






# sorting for parent and degnerates
list_pruning_types = [
    range(1), # parent
    range(nPrun) # children
]


list_pruning_stages  = [
    range(1),
    range(1,nStage)
]

list_paramList = [list(
    itertools.product(
        range(nNet),
        range(nDegen),
        range(nIndex),
        list_pruning_types[ips],
        list_pruning_stages[ips],
        range(nWeight))
    ) for ips in range(2)]

    
    
    