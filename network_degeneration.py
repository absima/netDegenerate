import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix, issparse

from parameters import *


def weightedFromAdjacency(cmtx, new_NI, weights, order_ie=None, return_sparse=True):
    """
    Apply 4-block weights to a connectivity matrix.

    Matrix semantics:
        M[tgt, src] != 0 encodes src -> tgt (Convention B)

    Args:
        cmtx: sparse or dense adjacency stored as M[tgt,src]
        new_NI: inhibitory count (must satisfy 0 < new_NI < N)
        weights: dict with keys {"II","EI","IE","EE"} (missing keys default to 1.0)
        order_ie: optional permutation where first new_NI indices are inhibitory
        return_sparse: if True return CSR else return dense array

    Returns:
        Weighted connectivity with semantics M[tgt,src]
    """
    if not issparse(cmtx):
        cmtx = csr_matrix(np.asarray(cmtx))
    else:
        cmtx = cmtx.tocsr(copy=False)

    Nloc = cmtx.shape[0]
    if cmtx.shape[0] != cmtx.shape[1]:
        raise ValueError("weightedFromAdjacency expects a square matrix.")
    if not (0 < int(new_NI) < Nloc):
        raise ValueError(f"new_NI must satisfy 0 < new_NI < N (got new_NI={new_NI}, N={Nloc}).")

    if order_ie is None:
        order_ie = np.arange(Nloc, dtype=np.int64)
    else:
        order_ie = np.asarray(order_ie, dtype=np.int64)
        if order_ie.ndim != 1 or order_ie.size != Nloc:
            raise ValueError("order_ie must be a permutation of size N.")

    # Apply symmetric permutation
    cmtx = cmtx[order_ie][:, order_ie]

    w_ii = float(weights.get("II", 1.0))
    w_ei = float(weights.get("EI", 1.0))  # I -> E  (src=I, tgt=E)
    w_ie = float(weights.get("IE", 1.0))  # E -> I  (src=E, tgt=I)
    w_ee = float(weights.get("EE", 1.0))

    coo = cmtx.tocoo(copy=True)
    tgt = coo.row
    src = coo.col
    data = coo.data.astype(np.float32, copy=False)

    tgt_is_i = tgt < new_NI
    src_is_i = src < new_NI

    mask_ii = tgt_is_i & src_is_i
    mask_ie = tgt_is_i & (~src_is_i)
    mask_ei = (~tgt_is_i) & src_is_i
    mask_ee = (~tgt_is_i) & (~src_is_i)

    if w_ii != 1.0:
        data[mask_ii] *= w_ii
    if w_ie != 1.0:
        data[mask_ie] *= w_ie
    if w_ei != 1.0:
        data[mask_ei] *= w_ei
    if w_ee != 1.0:
        data[mask_ee] *= w_ee

    coo.data = data
    coo.sum_duplicates()

    if return_sparse:
        return coo.tocsr()
    return coo.toarray()


def groupEdgesPerNodeNoEIsort(edges, new_NI, inout=0, perms=None):
    """
    Group edges by source or target node after applying an I/E-concatenated permutation.

    Args:
        edges: (m,2) edges as (src, tgt)
        new_NI: inhibitory count (first block)
        inout: 0 -> group by src (out), 1 -> group by tgt (in)
        perms: optional (iperm, eperm) permutations for I and E blocks

    Returns:
        perm: indices that reorder edges
    """
    edges = np.asarray(edges)
    Nloc = int(edges.max()) + 1 if edges.size else 0

    if not (0 < int(new_NI) < Nloc):
        raise ValueError(f"new_NI must satisfy 0 < new_NI < N (got new_NI={new_NI}, N={Nloc}).")

    if inout not in (0, 1):
        raise ValueError(f"inout must be 0 (out/src) or 1 (in/tgt), got {inout}.")

    if perms is None:
        iperm = np.random.permutation(new_NI)
        eperm = np.random.permutation(np.arange(new_NI, Nloc))
    else:
        iperm, eperm = perms

    ieperm = np.concatenate((iperm, eperm))
    ieperm_inv = ieperm.argsort()

    hedge = edges[:, inout]  # 0=src or out, 1=tgt or in
    positions = ieperm_inv[hedge]
    perm = np.argsort(positions)
    return perm


def trimSynapses(trim_params, ndir=None, pdir=None):
    """
    Trim synapses by pruning strategy index and stage.

    Strategy index convention (synapses):
        0 -> out  (group by src)
        1 -> in   (group by tgt)
        2 -> rnd
        3 -> ord
        4 -> res

    Args:
        trim_params: (edges, idxprun, istage)
            edges: (m,2) (src,tgt)
            idxprun: pruning strategy index (0..4)
            istage: stage index (fraction = istage*del_frac)

    Returns:
        COO adjacency matrix stored as M[tgt,src]
    """
    edges, idxprun, istage = trim_params
    edges = np.asarray(edges, dtype=np.int64)

    if idxprun in (0, 1):  # out/in
        synperm = groupEdgesPerNodeNoEIsort(edges, new_NI=NI, inout=idxprun)
        edges = edges[synperm]
    elif idxprun == 2:  # rnd
        prm = np.random.permutation(len(edges))
        edges = edges[prm]
    elif idxprun == 3:  # ord
        pass
    elif idxprun == 4:  # res
        edges = edges[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}.")

    ecut = int(istage * del_frac * len(edges))
    edges = edges[ecut:]

    return coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 1], edges[:, 0])),
        shape=(N, N),
    )



def trimNeurons(trim_params):
    """
    Trim neurons by deleting I and E nodes according to a node-based strategy index.

    Strategy index convention (nodes) — as in your existing code:
        0 -> iout
        1 -> ideg
        2 -> rand
        3 -> ddeg
        4 -> dout

    Args:
        trim_params: (edges, idxprun, istage)
            edges: (m,2) as (src,tgt)
            idxprun: node strategy index (0..4)
            istage: stage index (fraction = istage*del_frac)

    Returns:
        COO adjacency matrix of remaining neurons, stored as M[tgt,src]
    """
    edges, idxprun, istage = trim_params
    edges = np.asarray(edges, dtype=np.int64)

    odeg = np.bincount(edges[:, 0], minlength=N)
    ideg = np.bincount(edges[:, 1], minlength=N)
    degg = odeg + ideg

    prrm = np.arange(N)
    np.random.shuffle(prrm)

    odd = np.column_stack((odeg, degg, prrm, np.arange(N)))

    if idxprun == 0:        # iout
        sort = odd[:, 3][odd[:, 0].argsort()]
    elif idxprun == 1:      # ideg
        sort = odd[:, 3][odd[:, 1].argsort()]
    elif idxprun == 2:      # rand
        sort = odd[:, 3][odd[:, 2].argsort()]
    elif idxprun == 3:      # ddeg
        sort = odd[:, 3][odd[:, 1].argsort()][::-1]
    elif idxprun == 4:      # dout
        sort = odd[:, 3][odd[:, 0].argsort()][::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}.")

    sort = sort.astype(int)

    isort = sort[np.isin(sort, inrn)]
    esort = sort[np.isin(sort, enrn)]

    nidel = int(del_frac * NI * istage)
    nedel = int(del_frac * NE * istage)

    remaining = np.concatenate((isort[nidel:], esort[nedel:])).astype(int)

    a = coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 1], edges[:, 0])),
        shape=(N, N),
    ).tocsr()

    a_kept = a[remaining][:, remaining]
    return a_kept.tocoo()


def trimming(params):
    """
    Load ordered edges and apply trimming.

    Args:
        params: (cp_index, inet, idtyp, idxprun, istage)

    Returns:
        COO adjacency matrix stored as M[tgt,src]
    """
    cp_index, inet, idtyp, idxprun, istage = params

    netname = netnames[inet]
    ename = Path(netfold) / f"{netname}_relabeled_ordered_restored_{cp_index}.npz"

    if not ename.exists():
        raise FileNotFoundError(
            f"Missing ordered-edge file: {ename}. "
            "Run Stage I (generation + ordering) first."
        )

    edges = np.load(ename)["ordedges"].astype(np.int64)

    if istage == 0:
        return coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 1], edges[:, 0])),
            shape=(N, N),
        )

    trim_params = (edges, idxprun, istage)
    if idtyp:
        return trimNeurons(trim_params)
    return trimSynapses(trim_params)