# WARNING: This is to run all the simulation for all the parameters (networks, realization, prunings, stages, etc.) available 
import itertools
import time
from joblib import Parallel, delayed

from parameters import (
    n_real, n_net, n_degen, n_prun, n_stage, n_scales
)


def _run_one(paramss):
    from simulating import simulateAndStore
    return simulateAndStore(paramss)

def build_param_list(parent: bool):
    # parent run: stage=0 only, pruning index/type fixed to 0
    # child run: stage=1..n_stage-1, pruning index varies, type varies (0/1)
    if parent:
        idtyp_range = range(1)          # only 0
        idxprun_range = range(1)        # only 0
        istage_range = range(1)         # only 0
    else:
        idtyp_range = range(n_degen)    # 0..1
        idxprun_range = range(n_prun)   # 0..4
        istage_range = range(1, n_stage)

    # ilb is weight-scheme index {0,1}, do this:
    ilb_range = range(2)

    return list(itertools.product(
        range(n_real),     # cp_index
        ilb_range,         # ilb
        range(n_net),      # inet
        idtyp_range,       # idtyp
        idxprun_range,     # idxprun
        istage_range,      # istage
        range(n_scales),   # iscale
    ))

if __name__ == "__main__":
    start_time = time.strftime("%H:%M:%S", time.localtime())

    for parent in (True, False):
        param_list = build_param_list(parent)
        Parallel(n_jobs=-1, backend="loky")(
            delayed(_run_one)(p) for p in param_list
        )

    finish_time = time.strftime("%H:%M:%S", time.localtime())
    print("started at:", start_time)
    print("stopped at:", finish_time)



# import time, itertools
# import numpy as np
# from joblib import Parallel, delayed
#
# from simulating import *
# from parameters import *
#
# n_scales = len(wscales)
#
# list_pruning_types = [
# range(1), # parent
# range(n_degen) # children
# ]
#
# list_pruning_indices = [
# range(1), # parent
# range(n_prun) # children
# ]
#
# list_pruning_stages  = [
#     range(1),
#     range(1,n_stage)
# ]
#
# list_paramList = [list(
#     itertools.product(
#         range(n_real),
#         range(n_scales),
#         range(n_net),
#         list_pruning_types[ips], # range(nDegen),
#         list_pruning_indices[ips],
#         list_pruning_stages[ips],
#         range(nScale))    ) for ips in range(2)]
#
#
#
# if __name__ == '__main__':
#     __spec__ = None
#     start_time = time.strftime("%H:%M:%S", time.localtime())
#     for ips in range(2):
#         paramList = list_paramList[ips]
#         Parallel(n_jobs=-1)(delayed(simulateAndStore)(param) for param in paramList)
#         finish_time = time.strftime("%H:%M:%S", time.localtime())
#         print('started at: ', start_time)
#         print('stopped at: ', finish_time)
