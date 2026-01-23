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


# from __future__ import annotations
#
# import os
# import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix, issparse
#
#
# # Convention
# # ----------------------------------------------------------
# # - Edge list: always (src, tgt)
# # - Connectivity matrix (dense OR sparse): entry at (tgt, src) means src -> tgt
# #   i.e. columns are sources, rows are targets
# #
# # Conversions:
# #   - matrix -> edges: (src, tgt) = (col, row) from M.nonzero() = (row, col)
# #   - edges  -> matrix: place ones at (row=tgt, col=src)
#
#
# def lweight(param):
#     """
#     Build 4-block weights (scheme 1) from a 4-tuple.
#
#     Convention (A[tgt,src]):
#         II: I -> I
#         EI: I -> E
#         IE: E -> I
#         EE: E -> E
#
#     Args:
#         param: (w_ii, w_ei, w_ie, w_ee)
#
#     Returns:
#         dict with keys {"II","EI","IE","EE"}
#     """
#     w_ii, w_ei, w_ie, w_ee = param
#     return {"II": float(w_ii), "EI": float(w_ei), "IE": float(w_ie), "EE": float(w_ee)}
#
#
# def bweight(param, cross=None):
#     """
#     Build weights (scheme 2) from a 2-tuple, optionally overriding cross-blocks.
#
#     Args:
#         param: (w_i, w_e) where:
#             w_i is the II weight
#             w_e is the EE weight
#         cross: optional scalar for EI and IE; if None uses geometric mean
#
#     Returns:
#         dict with keys {"II","EI","IE","EE"}
#     """
#     w_i, w_e = param
#     w_i = float(w_i)
#     w_e = float(w_e)
#
#     if cross is None:
#         cross = float(np.sqrt(w_i * w_e))
#     else:
#         cross = float(cross)
#
#     return {"II": w_i, "EE": w_e, "EI": cross, "IE": cross}
#
#
#
#
#
# def _normalizeToEdges(graph, one_based=False):
#     """
#     Convert input to canonical edge list (src, tgt).
#
#     Interpretation:
#         - If matrix (dense or sparse): M[tgt,src] != 0 means src -> tgt
#         - If edges: already (src, tgt) format
#
#     Args:
#         graph: edges (m,2) OR dense matrix OR sparse matrix
#         one_based: if True and input is edges, subtract 1
#
#     Returns:
#         edges array (m,2) int32 as (src, tgt)
#     """
#     if issparse(graph):
#         mtx = graph.tocsr(copy=False)
#         tgt, src = mtx.nonzero()   # (row, col) = (tgt, src)
#         if src.size == 0:
#             return np.empty((0, 2), dtype=np.int32)
#         return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T
#
#     arr = np.asarray(graph)
#     if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
#         edges = arr.astype(np.int64, copy=False)
#         if one_based: # as in NEST or MATLAB ..
#             edges = edges - 1
#         return edges.astype(np.int32, copy=False)
#
#     mtx = csr_matrix(arr)
#     tgt, src = mtx.nonzero()
#     if src.size == 0:
#         return np.empty((0, 2), dtype=np.int32)
#     return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T
#
#
# def _edgesToCsr(edges, shape=None, dtype=np.int8):
#     """
#     Build CSR connectivity matrix from edges (src, tgt).
#
#     Matrix semantics:
#         M[tgt, src] = 1 encodes src -> tgt
#
#     Args:
#         edges: array (m,2) as (src, tgt)
#         shape: optional (n_rows, n_cols)
#         dtype: dtype for data
#
#     Returns:
#         CSR matrix M where M[tgt,src] indicates src -> tgt
#     """
#     edges = np.asarray(edges, dtype=np.int64)
#     if edges.size == 0:
#         if shape is None:
#             return csr_matrix((0, 0), dtype=dtype)
#         return csr_matrix(shape, dtype=dtype)
#
#     src = edges[:, 0]
#     tgt = edges[:, 1]
#
#     n_rows = int(max(tgt.max() + 1, shape[0] if shape else 0))
#     n_cols = int(max(src.max() + 1, shape[1] if shape else 0))
#
#     data = np.ones(src.size, dtype=dtype)
#     mtx = coo_matrix((data, (tgt, src)), shape=(n_rows, n_cols)).tocsr()
#     mtx.sum_duplicates()
#     return mtx
#
#
#
# def weightedFromAdjacency(adj_or_edges, NI, weights, neuron_order=None, return_sparse=True):
#     """
#     Apply 4-block weights to a connectivity matrix.
#
#     Args:
#         adj_or_edges: edges (src,tgt) or matrix M[tgt,src]
#         NI: number of inhibitory nodes (0..NI-1)
#         weights: dict with keys {"II","EI","IE","EE"}
#         neuron_order: optional permutation applied symmetrically
#         return_sparse: CSR if True else dense array
#
#     Returns:
#         Weighted connectivity matrix with semantics M[tgt,src]
#     """
#     if issparse(adj_or_edges):
#         mtx = adj_or_edges.tocsr(copy=False)
#     else:
#         arr = np.asarray(adj_or_edges)
#         if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
#             edges = _normalizeToEdges(arr)
#             mtx = _edgesToCsr(edges, dtype=np.float32)
#         else:
#             mtx = csr_matrix(arr.astype(np.float32, copy=False))
#
#     N = mtx.shape[0]
#     if N != mtx.shape[1]:
#         raise ValueError("weightedFromAdjacency expects a square matrix.")
#
#     if neuron_order is not None:
#         p = np.asarray(neuron_order, dtype=np.int64)
#         if p.ndim != 1 or p.size != N:
#             raise ValueError("neuron_order must be a permutation of size N.")
#         mtx = mtx[p][:, p]
#
#     i_idx = np.arange(NI, dtype=np.int32)
#     e_idx = np.arange(NI, N, dtype=np.int32)
#
#     mtx = mtx.tolil(copy=True)
#
#     if "II" in weights:
#         mtx[np.ix_(i_idx, i_idx)] = mtx[np.ix_(i_idx, i_idx)].astype(np.float32) * float(weights["II"])
#     if "EI" in weights:
#         mtx[np.ix_(e_idx, i_idx)] = mtx[np.ix_(e_idx, i_idx)].astype(np.float32) * float(weights["EI"])
#     if "IE" in weights:
#         mtx[np.ix_(i_idx, e_idx)] = mtx[np.ix_(i_idx, e_idx)].astype(np.float32) * float(weights["IE"])
#     if "EE" in weights:
#         mtx[np.ix_(e_idx, e_idx)] = mtx[np.ix_(e_idx, e_idx)].astype(np.float32) * float(weights["EE"])
#
#     w = mtx.tocsr()
#     return w if return_sparse else w.toarray()
#
#
#
#
# def loadOrdEdges(netname, index, ndir="."):
#     """
#     Load ordering artifact saved by saveCompressed (edge_ordering.py).
#
#     Args:
#         netname: name tag
#         index: trial index
#         ndir: directory containing the .npz file
#
#     Returns:
#         (edges_base, meta)
#     """
#     path = os.path.join(str(ndir), f"{netname}_trial{int(index)}_ordering.npz")
#     with np.load(path, allow_pickle=True) as z:
#         edges_base = z["edges_base"].astype(np.int32, copy=False)
#         meta = z["meta"].item() if "meta" in z else {}
#     return edges_base, meta
#
#
#
# def _edgeOrderByDegree(mtx, mode="out"):
#     """
#     Order edges by degree-based heuristics.
#
#     Semantics:
#         matrix is M[tgt,src]
#         edges are (src,tgt)
#
#     Args:
#         mtx: sparse connectivity matrix M[tgt,src]
#         mode: "out", "in", or "total"
#
#     Returns:
#         ordered_edges (m,2) as (src,tgt)
#     """
#     mtx = mtx.tocsr(copy=False)
#     tgt, src = mtx.nonzero()
#     if src.size == 0:
#         return np.empty((0, 2), dtype=np.int32)
#
#     # Out-degree of src = column sum (since cols are sources)
#     outdeg = np.asarray(mtx.sum(axis=0)).ravel()
#     # In-degree of tgt = row sum (since rows are targets)
#     indeg = np.asarray(mtx.sum(axis=1)).ravel()
#
#     if mode == "out":
#         keys = (-outdeg[src], tgt)
#     elif mode == "in":
#         keys = (-indeg[tgt], src)
#     elif mode == "total":
#         keys = (-(outdeg[src] + indeg[tgt]), src)
#     else:
#         raise ValueError("mode must be 'out','in','total'.")
#
#     order = np.lexsort(keys[::-1])
#     edges = np.vstack([src, tgt]).T.astype(np.int32)
#     return edges[order]
#
#
#
# def trimSynapses(graph, k, strategy="rand", ordering_dir=None, netname=None, trial_index=None, seed=None):
#     """
#     Remove k edges according to a strategy.
#
#     Semantics:
#         - matrices are M[tgt,src]
#         - edges are (src,tgt)
#
#     Args:
#         graph: edges or connectivity matrix
#         k: number of edges to remove
#         strategy: "rand","out","in","ord","rev"
#         ordering_dir/netname/trial_index: required for "ord"/"rev"
#         seed: RNG seed
#
#     Returns:
#         (mtx_pruned, removed_edges)
#         - mtx_pruned is CSR with semantics M[tgt,src]
#         - removed_edges is (r,2) (src,tgt)
#     """
#     rng = np.random.default_rng(seed)
#
#     edges = _normalizeToEdges(graph)
#     mtx = _edgesToCsr(edges, dtype=np.int8)
#
#     tgt, src = mtx.nonzero()
#     m = src.size
#
#     if k <= 0 or m == 0:
#         return mtx, np.empty((0, 2), dtype=np.int32)
#
#     k = min(k, m)
#
#     if strategy == "rand":
#         idx = rng.choice(m, size=k, replace=False)
#         to_remove = np.vstack([src[idx], tgt[idx]]).T.astype(np.int32)
#
#     elif strategy in ("out", "in"):
#         outdeg = np.asarray(mtx.sum(axis=0)).ravel()
#         indeg = np.asarray(mtx.sum(axis=1)).ravel()
#
#         present = set(zip(src.tolist(), tgt.tolist()))
#         removed = []
#
#         if strategy == "out":
#             order_nodes = np.argsort(-outdeg)  # sources by out-degree
#             for s in order_nodes:
#                 if len(removed) >= k:
#                     break
#                 # outgoing from s are entries in column s => rows are targets
#                 tgts = mtx.getcol(int(s)).indices
#                 for t in tgts:
#                     e = (int(s), int(t))
#                     if e in present:
#                         present.remove(e)
#                         removed.append(e)
#                         if len(removed) == k:
#                             break
#         else:
#             order_nodes = np.argsort(-indeg)  # targets by in-degree
#             for t in order_nodes:
#                 if len(removed) >= k:
#                     break
#                 # incoming to t are entries in row t => cols are sources
#                 srcs = mtx.getrow(int(t)).indices
#                 for s in srcs:
#                     e = (int(s), int(t))
#                     if e in present:
#                         present.remove(e)
#                         removed.append(e)
#                         if len(removed) == k:
#                             break
#
#         to_remove = np.asarray(removed, dtype=np.int32)
#
#     elif strategy in ("ord", "rev"):
#         if ordering_dir is None or netname is None or trial_index is None:
#             raise ValueError("strategy 'ord'/'rev' requires ordering_dir, netname, trial_index.")
#
#         edges_ord, _meta = loadOrdEdges(netname, trial_index, ordering_dir)
#         if strategy == "rev":
#             edges_ord = edges_ord[::-1]
#
#         present = set(zip(src.tolist(), tgt.tolist()))
#         removed = []
#
#         for e in edges_ord:
#             tup = (int(e[0]), int(e[1]))
#             if tup in present:
#                 present.remove(tup)
#                 removed.append(tup)
#                 if len(removed) == k:
#                     break
#
#         to_remove = np.asarray(removed, dtype=np.int32)
#
#     else:
#         raise ValueError("Unknown strategy. Choose from 'rand','out','in','ord','rev'.")
#
#     if to_remove.size == 0:
#         return mtx, to_remove
#
#     mtx_lil = mtx.tolil(copy=True)
#     # matrix stores (tgt,src)
#     mtx_lil[to_remove[:, 1], to_remove[:, 0]] = 0
#     mtx2 = mtx_lil.tocsr()
#     mtx2.eliminate_zeros()
#     return mtx2, to_remove
#
#
#
# def trimNeurons(graph, NI, n_remove_i=0, n_remove_e=0, strategy="dout", seed=None):
#     """
#     Remove neurons (rows & cols) by a node-selection strategy.
#
#     Args:
#         graph: edges or connectivity matrix
#         NI: number of inhibitory nodes
#         n_remove_i/n_remove_e: number of nodes to remove
#         strategy: "iout","ideg","ddeg","dout","rand"
#         seed: RNG seed
#
#     Returns:
#         (kept_idx, mtx_kept)
#         - kept_idx: indices kept in original indexing
#         - mtx_kept: CSR with semantics M[tgt,src]
#     """
#     rng = np.random.default_rng(seed)
#
#     edges = _normalizeToEdges(graph)
#     mtx = _edgesToCsr(edges, dtype=np.int8)
#
#     N = mtx.shape[0]
#     if N != mtx.shape[1]:
#         raise ValueError("trimNeurons expects a square matrix.")
#
#     outdeg = np.asarray(mtx.sum(axis=0)).ravel()
#     indeg = np.asarray(mtx.sum(axis=1)).ravel()
#     totdeg = outdeg + indeg
#
#     i_idx = np.arange(NI, dtype=np.int32)
#     e_idx = np.arange(NI, N, dtype=np.int32)
#
#     def pick(group, how_many):
#         if how_many <= 0:
#             return np.array([], dtype=np.int32)
#
#         if strategy in ("iout", "dout"):
#             order = group[np.argsort(-outdeg[group])]
#         elif strategy == "ideg":
#             order = group[np.argsort(-indeg[group])]
#         elif strategy == "ddeg":
#             order = group[np.argsort(-totdeg[group])]
#         elif strategy == "rand":
#             order = rng.permutation(group)
#         else:
#             raise ValueError("Unknown strategy for trimNeurons.")
#
#         return order[:how_many].astype(np.int32)
#
#     rem_i = pick(i_idx, min(n_remove_i, i_idx.size))
#     rem_e = pick(e_idx, min(n_remove_e, e_idx.size))
#     removed = np.concatenate([rem_i, rem_e]).astype(np.int32)
#
#     mask = np.ones(N, dtype=bool)
#     mask[removed] = False
#     kept_idx = np.flatnonzero(mask).astype(np.int32)
#
#     mtx_kept = mtx[kept_idx][:, kept_idx].tocsr()
#     return kept_idx, mtx_kept
#
#
#
#
# def trimming(
#     graph,
#     NI,
#     syn_k=0,
#     syn_strategy="rand",
#     node_strategy=None,
#     n_remove_i=0,
#     n_remove_e=0,
#     ordering_dir=None,
#     netname=None,
#     trial_index=None,
#     seed=None,
# ):
#     """
#     Trim synapses and/or neurons in one call.
#
#     Args:
#         graph: edges or connectivity matrix (M[tgt,src])
#         NI: number of inhibitory nodes
#         syn_k/syn_strategy: edge trimming options
#         node_strategy/n_remove_i/n_remove_e: node trimming options
#         ordering_dir/netname/trial_index: for ord/rev trimming
#         seed: RNG seed
#
#     Returns:
#         dict with keys {"A","removed_edges","kept_idx"}
#         - A: CSR matrix with semantics M[tgt,src]
#         - removed_edges: (m,2) as (src,tgt)
#         - kept_idx: kept node indices
#     """
#     edges0 = _normalizeToEdges(graph)
#     mtx0 = _edgesToCsr(edges0, dtype=np.int8)
#
#     removed_edges = np.empty((0, 2), dtype=np.int32)
#
#     if syn_k and syn_k > 0:
#         mtx1, removed_edges = trimSynapses(
#             mtx0,
#             k=syn_k,
#             strategy=syn_strategy,
#             ordering_dir=ordering_dir,
#             netname=netname,
#             trial_index=trial_index,
#             seed=seed,
#         )
#     else:
#         mtx1 = mtx0
#
#     if node_strategy is not None and (n_remove_i > 0 or n_remove_e > 0):
#         kept_idx, mtx2 = trimNeurons(
#             mtx1,
#             NI=NI,
#             n_remove_i=n_remove_i,
#             n_remove_e=n_remove_e,
#             strategy=node_strategy,
#             seed=seed,
#         )
#     else:
#         N = mtx1.shape[0]
#         kept_idx = np.arange(N, dtype=np.int32)
#         mtx2 = mtx1
#
#     return {"A": mtx2, "removed_edges": removed_edges, "kept_idx": kept_idx}
#
