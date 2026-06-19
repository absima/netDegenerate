"""
Stack qnt_*.npz simulation outputs into tuned/untuned multidimensional arrays.

Input filename convention:
    qnt_<ilb>_<cp_index>_<inet>_<idtyp>_<idxprun>_<istage>_<iscale>.npz

Output convention:
    Each stacked array has axes:
        realization, network, degeneration type, pruning type, stage,
        weight scale, variable, population category.
"""

import numpy as np
import scipy.io
import os

def organizing(qnt, reorderI=False):
    """
    Convert one qnt_*.npz payload into stacked observable blocks.

    Args:
        qnt: loaded qnt_*.npz object.
        reorderI: if True, reorder synaptic-current rows for legacy data.

    Returns:
        List of observable blocks ready to be vertically stacked.
    """
    # dgsh = qnt['dgsh']
    esw  = qnt['esw']
    eff  = qnt['mneff'] 
    radi = np.reshape(qnt['radius'], (1,))
    mnBlock, sdBlock = qnt['mnsdeff']
    
    synI = qnt['synI']
    if reorderI:#due to the way the data was collected.
        synI = synI[[2,0,1]] #### ---------------------------------
    
    rvc  = qnt['rvc']
    sync = qnt['sync']
    
    ccff = qnt['ccff']
    ccc = ccff[:12].reshape(3,4)
    
    # fake placeholder as we reduced binwidth for postprocessing
    ccc = np.concatenate((ccc, ccc[-1:,:]), axis = 0)
    # 
    ccs = ccc[:,[0,2,3]]
    ffs = ccff[12:].reshape(3,3)
    
    # fake placeholder
    ffs = np.concatenate((ffs, ffs[-1:,:]), axis =0)
    cc_ei = ccc[:,1]
    

    a1 = esw #6 
    a2 = np.repeat(eff, 3).reshape(1,3) #1
    a3 = np.repeat(radi, 3).reshape(1,3) #1 
    a4 = np.repeat(mnBlock[:,None], 3, 1) #4 mean blocks
    a5 = np.repeat(sdBlock[:,None], 3, 1) #4 sigma blocks
    a6 = synI #2 inh, exc # 3....
    a7 = rvc #3 mn, sd, cv
    a8 = sync.reshape(1,3) #1
    a9 = ffs #4 bins
    a10 = ccs #4 bins
    a11 = np.repeat(cc_ei[:,None], 3, 1) #4 bins
    x = [ a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]    
    return x
    
    
def create_big_array(
    ilb, 
    qfold,
    reorderI=False,
    nNet = 6,
    nDtyp = 2,
    nPrun = 5,
    nStage = 10,
    nVars = 42,
    nPop = 3,
    nReal = 10,
    nWeight = 10
    ):
    """
    Build one stacked output tensor from a qntfold directory.

    Args:
        ilb: weight-scheme index used in qnt filenames.
        qfold: directory containing qnt_*.npz files.
        reorderI: if True, reorder synaptic-current rows before stacking.
        nNet: number of network families.
        nDtyp: number of degeneration types.
        nPrun: number of pruning strategies.
        nStage: number of degeneration stages.
        nVars: number of stacked variables.
        nPop: number of population categories.
        nReal: number of realizations.
        nWeight: number of weight scales.

    Returns:
        Stacked array with shape
        (nReal, nNet, nDtyp, nPrun, nStage, nWeight, nVars, nPop).
    """
    
    if ilb:
        ilb=1
    qarr = np.full((nReal, nNet, nDtyp, nPrun, nStage, nWeight, nVars, nPop), np.nan)
    for itrial in range(nReal):
        print(itrial)
        for iname in range(nNet):
            for idtyp in range(nDtyp):
                for idxprun in range(nPrun):
                    for istage in range(nStage):
                        for iweight in range(nWeight):

                            #  """ NAN case skipping"""
                            # if ilb==0 and iname==3 and iweight==4:
                            #     continue
                            # """ NAN case skipping"""
                            idtyp2 = idtyp
                            if istage==0:
                                idxprun2 = 0
                                if idtyp: # to fill the synaptic parents for neuronal pruning stage0
                                    idtyp2 = 0
                            else:
                                idxprun2 = idxprun

                            params = [ilb, itrial, iname, idtyp2, idxprun2, istage, iweight]


                            qstrng = tuple([qfold]+list(params))
                            fname = '%s/qnt_%d_%d_%d_%d_%d_%d_%d.npz'%qstrng

                            qnt = np.load(fname)
                            if iweight:
                                dgsh = qarr[itrial, iname, idtyp2, idxprun2, istage, 0, :7,:]

                            else:
                                dgsh = qnt['dgsh']
                            combined = organizing(qnt, reorderI)
                            combo = np.vstack(combined)
                            comb = np.vstack((dgsh, combo))

                            qarr[itrial, iname, idtyp, idxprun, istage, iweight] = comb
    print('')                        
    print('======================')
    print('done')
    print('======================')   
    print('')                     
    return qarr
    
    

parfold = '/Users/sima/Projects/work/nestZone'


""" 1. tuned - linscale """
#lin - l and b g3, g5, g10
tlinfold = f'{parfold}/22_bgd3_noNest/02_tuned'
tlinsfolds = ['landau', 'brunel_g10']
ilbs = [0, 1]
lb_lin_g10 = np.array([
    create_big_array(ilbs[k], f'{tlinfold}/{tlinsfolds[k]}/qntfold') 
    for k in range(len(ilbs))])

""" 2. untuned - linscale """
#lin - l and b g3, g5, g10
ulinfold = f'{parfold}/22_bgd3_noNest/01_untuned/02_linscale'
ulinsfolds = ['landau', 'brunel_g3', 'brunel_g5', 'brunel_g10']
ilbs = [0, 1, 1, 1]
lb_lin_g3g5g10 = np.array([
    create_big_array(ilbs[k], f'{ulinfold}/{ulinsfolds[k]}/qntfold', reorderI=True) 
    for k in range(len(ilbs))])

""" 3. untuned - logscale """
#log - l and b g5
logfold = f'{parfold}/22_bgd3_noNest/01_untuned/01_logscale'
logsfolds = ['landau', 'brunel_g5', 'brunel_g10']
ilbs = [0,1,1]
lb_log_g5g10 = np.array([
    create_big_array(ilbs[k], f'{logfold}/{logsfolds[k]}/qntfold', reorderI=True) 
    for k in range(len(ilbs))])




np.savez_compressed(
    'data/tuned_untuned_42vars_3cats.npz',
    llin_t  = lb_lin_g10[0],
    b10lin_t = lb_lin_g10[1],
    llin_u  = lb_lin_g3g5g10[0],
    b3lin_u = lb_lin_g3g5g10[1],
    b5lin_u = lb_lin_g3g5g10[2],
    b10lin_u= lb_lin_g3g5g10[3],  
    llog_u  = lb_log_g5g10[0],
    b5log_u = lb_log_g5g10[1],
    b10log_u= lb_log_g5g10[2],
    readme = """ b5 means brunel with g5, lin means linscale weight scalers; t means tuned"""      
    )






