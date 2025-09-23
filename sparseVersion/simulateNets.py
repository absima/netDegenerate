import time, itertools
import numpy as np
import scipy.io 
from joblib import Parallel, delayed

from funcNest import *
from parameters import *



if __name__ == '__main__':
    __spec__ = None
    start_time = time.strftime("%H:%M:%S", time.localtime())
    for ips in range(2):
        paramList = list_paramList[ips]
        Parallel(n_jobs=-1)(delayed(simulateAndStore)(param) for param in paramList)
        finish_time = time.strftime("%H:%M:%S", time.localtime())
        print('started at: ', start_time)
        print('stopped at: ', finish_time)
        

# prm = list_paramList[0][0]
#
# simulateAndStore(prm)





                    
    
