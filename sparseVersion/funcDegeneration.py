from scipy.sparse import coo_matrix, load_npz
from parameters import *


def scaledWeight(netname, scale):
    if netname in ['emp', 'erb', 'sfo']:
        weight = Lweight(scale) # wii, wie, wei, wee
    elif netname in ['ero', 'sfr', 'swr']:
        weight = Bweight(scale) # wii, wie, wei, wee
    else:
        raise ValueError(f"Invalid netname: {netname}.")
    return weight
      
def Lweight(scale):
    return scale * np.array([-3., -3., 1.5, 1])
    
def Bweight(scale, g=5):
    return scale * np.array([-5, -5, 1, 1])
    
    
def loadNetwork(netname, index, ndir=None, pdir=None):
    
    # if netname =='xx':
    #     return np.loadtxt('edgezN22.txt').astype(int)
    if ndir is None:
        ndir = netfold
    fname = f'{ndir}/{netname}_ordEdges_{index}.npz'
    edgz = np.load(fname)['ordedges']
    return edgz.astype(int)
    


def weightedFromAdjacency(cmtx, how_many_Ineurons, weight, orderIE=None):

    if orderIE is None:
        orderIE = np.arange(cmtx.shape[0])
    
    nrnI = orderIE[:how_many_Ineurons]
    nrnE = orderIE[how_many_Ineurons:]
    
    j_i2i, j_i2e, j_e2i, j_e2e = weight
    
    cmtx = cmtx.toarray()
    cmtx = cmtx.astype(float)
    cmtx[np.ix_(nrnI, nrnI)] *= j_i2i
    cmtx[np.ix_(nrnE, nrnI)] *= j_i2e
    cmtx[np.ix_(nrnI, nrnE)] *= j_e2i
    cmtx[np.ix_(nrnE, nrnE)] *= j_e2e
    
    return coo_matrix(cmtx)
    
    
    
def trimSynapses(trim_params):
    netname, cp_index, idxprun, istage = trim_params   
    
    edgez = loadNetwork(netname, cp_index)
    nedgez = len(edgez)
    src_indices, tgt_indices = edgez.T
    if idxprun==0:   # out
        sorted_indices = np.argsort(tgt_indices)
    elif idxprun==1: # in 
        sorted_indices = np.argsort(src_indices)
    elif idxprun==2: # rand
        
        sorted_indices = np.random.permutation(nedgez)
    elif idxprun==3: # ord 
        sorted_indices = np.arange(nedgez)
    elif idxprun==4: # res
        sorted_indices = np.arange(nedgez)[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}. Please provide a valid value.")
    
    sedgez = edgez[sorted_indices]
    
    ncutt = int(istage*del_frac*len(sedgez))
    edges_rem = sedgez[ncutt:]
    
    unit = np.ones(len(edges_rem), dtype=int)
    cbmtx = coo_matrix((unit, (edges_rem[:,1], edges_rem[:,0])), shape=(N0,N0))
    
    return cbmtx
    
    
    
            
def trimNeurons(trim_params):
    netname, cp_index, idxprun, istage = trim_params
    
    edgez = loadNetwork(netname, cp_index)
    odeg = np.bincount(edgez[:,0], minlength=N0)
    ideg = np.bincount(edgez[:,1], minlength=N0)
    degg = odeg + ideg
    
    cbmtx = coo_matrix((np.ones(len(edgez)), (edgez[:,1], edgez[:,0])), shape=(N0,N0))
    
    if istage==0:# no pruning
        return cbmtx
    
    prrm = np.arange(N0) # for random nodal attack
    np.random.shuffle(prrm)
    
    odd = np.column_stack((odeg, degg, prrm, np.arange(N0)))    
    if  idxprun==0:  #iout 
        sort = odd[:,3][odd[:,0].argsort()]
    elif idxprun==1: #ideg
        sort = odd[:,3][odd[:,1].argsort()]
    elif idxprun==2: # random
        sort = odd[:,3][odd[:,2].argsort()]
    elif idxprun==3: # ddeg
        sort = odd[:,3][odd[:,1].argsort()]
        sort = sort[::-1]
    elif idxprun==4: # dout
        sort = odd[:,3][odd[:,0].argsort()]
        sort = sort[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}. Please provide a valid value.")
    sort = sort.astype(int)
    
    isort = sort[np.isin(sort, Inrn)]
    esort = sort[np.isin(sort, Enrn)]
    
    nidel = int(del_frac * NI * istage)
    nedel = int(del_frac * NE * istage)
    
    # we make sure Inh are indexed first in order
    to_delete = np.concatenate((isort[:nidel],esort[:nedel]))
    remaining = np.concatenate((isort[nidel:],esort[nedel:])) 
    remaining = remaining.astype(int)
    
    
    cbmtx = cbmtx.toarray()
    cbmtx = cbmtx[np.ix_(remaining, remaining)].astype(int)
    cbmtx = coo_matrix(cbmtx)
    
    return cbmtx
    
        
    
def trimming(params):
    """ 
    netname - network name
    idtyp - degeneration scheme as in edge or node removal
    cp_index - the trial id 
    idxprun - is one of the 5 prunings
    istage - is one of the 10 stages of pruning
    """
    netname, idtyp, cp_index, idxprun, istage = params
    trim_params = (netname, cp_index, idxprun, istage)
    
    if idtyp:
        return trimNeurons(trim_params)
    else:
        return trimSynapses(trim_params)
        




