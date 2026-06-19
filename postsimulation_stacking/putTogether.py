"""
Append weight, connection-density, and network-size features to stacked arrays.

This script reads the 42-variable tuned/untuned arrays created from qnt_*.npz
files, selects structural and dynamic variables, appends w/p/N features, and
writes a flattened feature dataset for downstream analysis.
"""

import numpy as np
import scipy.io
from pathlib import Path

from params import *

############## to append the w_p_n 10 feature vectors: 
def network_size():
    """
    Compute remaining E/I population sizes for each degeneration stage.

    Returns:
        Array with shape (nDtyp, nStage, 2), storing [NE, NI].
    """
    NI, NE = [ 680, 2931]
    del_frac = 0.1
    nEnI = np.zeros((2, 10, 2))
    for idtyp in range(2):
        for istage in range(10):
            newNI = NI-idtyp*int(del_frac*istage*NI)
            newNE = NE-idtyp*int(del_frac*istage*NE)
            nEnI[idtyp, istage] = [newNE, newNI]
    return nEnI
    
def block_conn_density(nreal=None):
    """
    Load block connection densities used as appended structural features.

    Args:
        nreal: optional number of realizations to keep.

    Returns:
        Connection-density array.
    """
    parfold = '/Users/sima/Projects/work/nestZone'
    rhofold = '20_tunedWeight_run3/merging'
    pconns0 = np.load(f'{parfold}/{rhofold}/pConnBlocks.npy')
    pconns = pconns0[0,...,0,:]
    if nreal is not None:
        pconns = pconns[:nreal]
    return pconns
    
    
    
def cleanData(lbarr0, nScale=None): 
    """
    Select and flatten structural/current/dynamic variables.

    Args:
        lbarr0: stacked tuned/untuned array.
        nScale: optional number of weight scales to keep.

    Returns:
        xydata: concatenated structural and target variables.
        vxydata: variable names for xydata.
        mvxydata: math-formatted variable names.
        indices: grouped variable index lists.
    """
    
    if nScale is None:
        nScale = lbarr0.shape[-3]
    lbarr = lbarr0[..., ivarnames,:]
    lbarr = lbarr[...,:nScale,:,:]

    snew_shape = list(lbarr.shape[:-2]) + [len(dxstr1)*3]
    xdata3 = lbarr[...,dxstr1,:].reshape(snew_shape)
    
    xdata1 = lbarr[...,dxstr2,0]#.reshape(-1, len(dxstr2))
    xdata = np.concatenate((xdata3, xdata1), axis=-1)
    
    
    ydx3 = dxdyn1 + dxcurr
    dnew_shape =list(lbarr.shape[:-2]) + [len(ydx3)*3] 
    ydata = lbarr[...,ydx3,:].reshape(dnew_shape)

    vxdata =  np.concatenate((
        nvarnames[dxstr1].flatten(),
        nvarnames[dxstr2,0]
        ))
    vydata = nvarnames[ydx3].flatten()

    mvxdata = np.concatenate((
        mvarnames[dxstr1].flatten(),
        mvarnames[dxstr2,0]
        ))
    mvydata = mvarnames[ydx3].flatten()

    xydata = np.concatenate((xdata, ydata), axis=-1)
    xydata = np.swapaxes(xydata, 0,1)
    vxydata = np.concatenate((vxdata, vydata))
    mvxydata = np.concatenate((mvxdata, mvydata))
    
    indices = [dxstr1, dxstr2, dxcurr, dxdyn1, dxdyn2]
    return xydata, vxydata, mvxydata, indices 
    

def expand_axes(
    x, 
    shp,
    nRealx=1, 
    nNetx=1, 
    nDegenx=1, 
    nPrunx=1, 
    nStagex=1, 
    nScalex=1, 
    nVarx=1,
    ):
    
    nNet1, nReal1, nDegen1, nPrun1, nStage1, nScale1 = shp[:-1]
        
    shpe = (nNetx, nRealx, nDegenx, nPrunx, nStagex, nScalex, nVarx)
    # print('------')
    # print(x.shape)
    # print(shpe)
    y = np.reshape(x, shpe)
    shpe2 = np.array(shpe[:])
    dxnot1 = np.where(shpe2!=1)[0]
    # dxones = np.where(shpe2==1)[0]
    shpe3 = np.array([nNet1, nReal1, nDegen1, nPrun1, nStage1, nScale1, nVarx])
    # print(shpe3)
    # print(dxnot1)
    shpe3[dxnot1] = 1
    z = np.tile(y, shpe3)
    
    return z
    
# def v76_wpn(source, dname, lb, reorder=True):
def v76_wpn(key, reorder=True):
    y = bdata[key]
    
    #sign the block means
    dxs = [varnames.index(i) for i in ['mnwEI', 'mnwII']]
    print("**************************")
    print(y.shape)
    ycopy = y[..., dxs,:]
    y[..., dxs, :] = -ycopy
    
    t_or_u = key[-1] # tuned / untuned
    l_or_b = key[0] # landau/ brunel
    
    # scales
    if t_or_u == 'u':
        if 'lin' in key:
            scalers = np.linspace(0.1, 15., 10)
        elif 'log' in key:
            scalers = np.logspace(np.log10(.1), np.log10(15.), 10)
        else:
            raise ValueError("untuned has log or lin variants only")
        scalez = np.repeat(scalers[None, :], 6, 0)
    elif t_or_u == 't':
        if l_or_b=='l':
            scalez = tscales_l
        elif l_or_b=='b':
            scalez = tscales_b
        else:
            raise ValueError("we currently have l or b weight-scheme only")    
    else:
        raise ValueError("It should be tuned(t) or untuned(u)") 

    # weights
    if l_or_b=='l':
        wgt = np.array([1., -3, 1.5, -3.]) # in the right order of ee e<-i, i<-e, ii      
    elif l_or_b=='b':
        g = int(key[1])
        if g == 1:
            g = 10
        wgt = np.array([1., -g, 1., -g])
    else:
        raise ValueError("we currently have l or b weight-scheme only")
    

    x, variables, mvariables, indices = cleanData(y)
    shape = x.shape
    print(shape)
    print("--------------------------")
    print(key)
    print(wgt)
    

    nNet, nReal, nDegen, nPrun, nStage, nScale = shape[:-1]


    wcore = scalez[...,None] * wgt
    w = expand_axes(
        wcore,
        shp = shape,
        # nLBx=1,
        nNetx=nNet,
        nScalex=nScale,
        nVarx=4,
        )

    pcore = block_conn_density(nReal)
    pcore = pcore.transpose(1, 0, 2, 3, 4, 5)
    p = expand_axes(
        pcore,
        shp = shape,
        nRealx=nReal,
        nNetx=nNet,
        nDegenx=nDegen,
        nPrunx = nPrun,
        nStagex=nStage,
        nVarx=4)

    ncore = network_size()
    n = expand_axes(
        ncore,
        shp = shape,
        nDegenx=nDegen,
        nStagex=nStage,
        nVarx=2)

    _wpn = np.concatenate((w, p, n), axis=-1)
    #
    _merged = np.concatenate((x, _wpn[:,...]), axis=-1)
    
    variables = np.concatenate((variables, vwpn))
    mvariables = np.concatenate((mvariables, mvwpn))
    if not reorder:
        return _merged, variables, mvariables, indices

    ### new orders
    z0 = np.take(_merged, dxnets, axis=0) # new network orders
    z = np.take(z0, dxnptypes, axis=3) # new deletion orders
    
    return z, variables, mvariables, indices
    
    


bdata = np.load('data/tuned_untuned_42vars_3cats.npz')

### untuned
# untuned -- logspaced
tuned_lin_l, variables, mvariables, _ = v76_wpn('llin_t')
tuned_lin_b5, variables, mvariables, _ = v76_wpn('b10lin_t')

untuned_lin_l, variables, mvariables, _ = v76_wpn('llin_u')
untuned_lin_b3, variables, mvariables, _ = v76_wpn('b3lin_u')
untuned_lin_b5, variables, mvariables, _ = v76_wpn('b5lin_u')
untuned_lin_b10, variables, mvariables, _ = v76_wpn('b10lin_u')

untuned_log_l, variables, mvariables, _ = v76_wpn('llog_u')
untuned_log_b5, variables, mvariables, _ = v76_wpn('b5log_u')
untuned_log_b10, variables, mvariables, _ = v76_wpn('b10log_u')



save_payload = dict(
    untuned_log_l=untuned_log_l,
    untuned_log_b5=untuned_log_b5,
    untuned_log_b10=untuned_log_b10,
    untuned_lin_l=untuned_lin_l,
    untuned_lin_b5=untuned_lin_b5,
    untuned_lin_b3=untuned_lin_b3,
    untuned_lin_b10=untuned_lin_b10,
    tuned_lin_l=tuned_lin_l,
    tuned_lin_b10=tuned_lin_b5,
    netnames=netnames,
    sptypes=sptypes,
    nptypes=nptypes,
    dx_str3cat=dxstr1,
    dx_str1cat=dxstr2,
    dx_current=dxcurr,
    dx_dyn3cat=dxdyn1,
    dx_dyn1cat=dxdyn2,
    dx_wpn=np.arange(33, 43),
    varnames=variables,
    mvarnames=mvariables,
)



np.savez_compressed(
    'data/tuned_untuned_flat86vars_with_wpn.npz',
    **save_payload,
)
