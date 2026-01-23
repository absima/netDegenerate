# parameters.py
from __future__ import annotations

import numpy as np

# Paths
parfold = "/parent_dir"
netfold = parfold
qntfold = "qntfold"

# Population sizes
N = 100
NI = N // 5
NE = N - NI

inrn = np.arange(NI)
enrn = np.arange(NI, N)

# Network families (Stage I)
netnames = [
    "emp",
    "erb",
    "ero",
    "swr",
    "sfo",
    "sfr",
]

# Pruning strategies (Stage I)  NOTE: kept as-is to avoid breaking upstream code
edge_prunings = [
    "out",
    "in",
    "rnd",
    "ord",
    "res",
]
node_prunings = [
    "iout",
    "ideg",
    "nrnd",
    "dout",
    "ddeg",
]
n_net = len(netnames)

# Experiment grid sizes
n_real = 10
n_degen = 2
n_index = 10
n_prun = len(edge_prunings)
n_stage = 10

del_frac = 0.1

# Synaptic scaling and background input
j_unit = 0.013  # 0.0048
je = 0.1
p_rate = 8000.0  # rate of the poisson generator
j_bg = je / j_unit

# Synapse and simulation timing
tau_syn = 2.0
delay = 1.5

s_simtime = 11.0
s_recstart = 1.0
s_nettime = s_simtime - s_recstart

dt = 0.1  # ms

ms_simtime = s_simtime * 1000.0
ms_recstart = s_recstart * 1000.0
ms_nettime = int(s_nettime * 1000.0)
ms_duration = ms_nettime + 1

ms_startI = (s_nettime - 2.0) * 1000.0
ms_durI = ms_nettime - ms_startI

# Time grid used in current-based reconstructions
t_grid = np.arange(0, ms_durI - 1)

# Weight presets (kept as-is)
wk0 = np.array([-3.0, -3.0, 1.5, 1.0])
wg0 = np.array([-5.0, -5.0, 1.0, 1.0])

# Weight scalers
wscales = np.logspace(np.log10(.1), np.log10(15.), 10)







