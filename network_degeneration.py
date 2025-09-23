
# trimming.py
from __future__ import annotations

import os
from typing import Tuple, Dict, List, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse


# ==========================================================
# ORIENTATION CONTRACT (internal)
# ----------------------------------------------------------
# - We store edges as: A[src, tgt] = 1
#   rows   = sources
#   cols   = targets
# - Canonical edge tuples are (src, tgt).
# - If an external routine expects rows=targets, cols=sources, use M.T at the boundary.
# ==========================================================


# -----------------------------
# Lightweight weighting helpers
# -----------------------------
def Lweight(w_II: float, w_IE: float, w_EI: float, w_EE: float) -> Dict[str, float]:
    """
    Convenience builder for a 4-block weight dictionary:
      - "II": src in I, tgt in I
      - "IE": src in I, tgt in E
      - "EI": src in E, tgt in I
      - "EE": src in E, tgt in E
    """
    return {"II": float(w_II), "IE": float(w_IE), "EI": float(w_EI), "EE": float(w_EE)}


def Bweight(w_I: float, w_E: float, cross: Optional[float] = None) -> Dict[str, float]:
    """
    Binary-style weight builder: give a weight for I-blocks and E-blocks.
    By default, cross-blocks get the geometric mean unless `cross` is provided.

    Examples
    --------
    Bweight(2.0, 1.0) -> {"II": 2.0, "EE": 1.0, "IE": sqrt(2), "EI": sqrt(2)}
    Bweight(2.0, 1.0, cross=1.5) -> {"II":2.0, "EE":1.0, "IE":1.5, "EI":1.5}
    """
    w_I = float(w_I); w_E = float(w_E)
    if cross is None:
        cross = float(np.sqrt(w_I * w_E))
    return {"II": w_I, "EE": w_E, "IE": float(cross), "EI": float(cross)}


def ScaledWeight(base: Dict[str, float], scale: float) -> Dict[str, float]:
    """
    Scale an existing block-weight dictionary by a constant factor.
    """
    scale = float(scale)
    return {k: float(v) * scale for k, v in base.items()}


# -----------------------------
# Core conversions
# -----------------------------
def _NormalizeToEdges(graph, one_based: bool = False) -> np.ndarray:
    """
    Normalize input into a canonical edge list (src, tgt).

    Parameters
    ----------
    graph : (m,2) array-like | sparse | dense
        If edges: assumed to be (src, tgt) possibly 1-based if `one_based=True`.
        If sparse/dense: treated as internal orientation A[src, tgt] (rows=sources, cols=targets).
    one_based : bool
        If True and input is edges, subtract 1 to convert to 0-based.

    Returns
    -------
    edges : (m,2) int32
    """
    if issparse(graph):
        M = graph.tocsr(copy=False)
        src, tgt = M.nonzero()
        if src.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T

    arr = np.asarray(graph)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
        edges = arr.astype(np.int64, copy=False)
        if one_based:
            edges = edges - 1
        return edges.astype(np.int32, copy=False)

    # dense
    M = csr_matrix(arr)
    src, tgt = M.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T


def _EdgesToCSR(edges: np.ndarray, shape: Optional[Tuple[int, int]] = None, dtype=np.int8) -> csr_matrix:
    """
    Build CSR adjacency from canonical edges (src, tgt) using internal orientation.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        if shape is None:
            return csr_matrix((0, 0), dtype=dtype)
        return csr_matrix(shape, dtype=dtype)
    src = edges[:, 0]
    tgt = edges[:, 1]
    n_rows = int(max(src.max() + 1, shape[0] if shape else 0))
    n_cols = int(max(tgt.max() + 1, shape[1] if shape else 0))
    data = np.ones(src.size, dtype=dtype)
    M = coo_matrix((data, (src, tgt)), shape=(n_rows, n_cols)).tocsr()
    M.sum_duplicates()
    return M


# -----------------------------
# Weighted adjacency
# -----------------------------
def WeightedFromAdjacency(adj_or_edges,
                          NI: int,
                          weights: Dict[str, float],
                          neuron_order: Optional[np.ndarray] = None,
                          return_sparse: bool = True):
    """
    Apply 4-block weights to an adjacency using internal orientation.

    Orientation & Blocks
    --------------------
    - Internal orientation: A[src, tgt].
    - Blocks are defined by (src in I/E) × (tgt in I/E):
        "II", "IE", "EI", "EE" keys in `weights`.

    Parameters
    ----------
    adj_or_edges : edges | sparse | dense
        Input adjacency or edge list. If edges, they must be (src, tgt).
    NI : int
        Number of inhibitory neurons (0..NI-1). The rest are excitatory.
    weights : dict
        Keys {"II","IE","EI","EE"} with float values.
    neuron_order : np.ndarray or None
        Optional permutation to apply symmetrically on rows and cols.
    return_sparse : bool, default True

    Returns
    -------
    W : csr_matrix or ndarray
        Weighted adjacency.
    """
    # Normalize to CSR
    if issparse(adj_or_edges):
        A = adj_or_edges.tocsr(copy=False)
    else:
        edges = _NormalizeToEdges(adj_or_edges)
        A = _EdgesToCSR(edges)

    N = A.shape[0]
    assert A.shape[0] == A.shape[1], "WeightedFromAdjacency expects a square adjacency."

    # Optional symmetric relabeling
    if neuron_order is not None:
        p = np.asarray(neuron_order, dtype=np.int64)
        if p.ndim != 1 or p.size != N:
            raise ValueError("neuron_order must be a permutation of size N.")
        A = A[p][:, p]

    # Split index sets
    I = np.arange(NI, dtype=np.int32)
    E = np.arange(NI, N, dtype=np.int32)

    # Extract blocks and scale
    # Note: slicing A[src_idx][:, tgt_idx]
    A = A.tolil(copy=True)
    if "II" in weights:
        A[np.ix_(I, I)] = (A[np.ix_(I, I)]).astype(np.float32) * float(weights["II"])
    if "IE" in weights:
        A[np.ix_(I, E)] = (A[np.ix_(I, E)]).astype(np.float32) * float(weights["IE"])
    if "EI" in weights:
        A[np.ix_(E, I)] = (A[np.ix_(E, I)]).astype(np.float32) * float(weights["EI"])
    if "EE" in weights:
        A[np.ix_(E, E)] = (A[np.ix_(E, E)]).astype(np.float32) * float(weights["EE"])

    W = A.tocsr()
    if return_sparse:
        return W
    return W.toarray()


# -----------------------------
# Ordering artifacts I/O
# -----------------------------
def LoadOrdEdges(netname: str,
                 index: int,
                 ndir: str | os.PathLike[str] = ".") -> Tuple[np.ndarray, Dict]:
    """
    Load ordering artifact saved by `SaveCompressed` (edge_ordering.py).

    Returns
    -------
    edges_base : (m,2) int32
    meta : dict-like with keys such as 'netname', 'itrial', 'orientation'
    """
    path = os.path.join(str(ndir), f"{netname}_trial{int(index)}_ordering.npz")
    with np.load(path, allow_pickle=True) as Z:
        edges_base = Z["edges_base"].astype(np.int32, copy=False)
        meta = Z["meta"].item() if "meta" in Z else {}
    return edges_base, meta


# -----------------------------
# Edge ordering utilities
# -----------------------------
def _EdgeOrderByDegree(A: csr_matrix,
                       mode: str = "out") -> np.ndarray:
    """
    Produce an edge ordering based on degrees.

    Parameters
    ----------
    A : csr_matrix (square or rectangular)
        Internal orientation A[src, tgt].
    mode : {'out','in','total'}
        - 'out': sort by source out-degree (row sum) descending, stable within each src by tgt.
        - 'in' : sort by target in-degree (col sum) descending.
        - 'total': sort by (outdeg(src) + indeg(tgt)) descending.

    Returns
    -------
    ordered_edges : (m,2) int32 in canonical (src, tgt)
    """
    A = A.tocsr(copy=False)
    src, tgt = A.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    outdeg = np.asarray(A.sum(axis=1)).ravel()  # per source
    indeg = np.asarray(A.sum(axis=0)).ravel()   # per target

    if mode == "out":
        keys = (-outdeg[src], tgt)
    elif mode == "in":
        keys = (-indeg[tgt], src)
    elif mode == "total":
        keys = (-(outdeg[src] + indeg[tgt]), src)
    else:
        raise ValueError("mode must be 'out','in','total'.")

    order = np.lexsort(keys[::-1])  # lexsort uses last key as primary; reverse to make first primary
    edges = np.vstack([src, tgt]).T.astype(np.int32)
    return edges[order]


# -----------------------------
# Edge trimming (synapses)
# -----------------------------
def TrimSynapses(graph,
                 k: int,
                 strategy: str = "rand",
                 ordering_dir: Optional[str] = None,
                 netname: Optional[str] = None,
                 trial_index: Optional[int] = None,
                 seed: Optional[int] = None) -> Tuple[csr_matrix, np.ndarray]:
    """
    Remove `k` edges according to the chosen strategy.

    Strategies
    ----------
    - 'rand': remove k random edges uniformly.
    - 'out' : remove edges from sources with highest out-degree first.
    - 'in'  : remove edges to targets with highest in-degree first.
    - 'ord' : remove in the order saved by SaveCompressed (requires ordering_dir, netname, trial_index).
    - 'rev' : remove in the reverse order of 'ord'.

    Returns
    -------
    A_pruned : csr_matrix
    removed_edges : (k,2) int32 (src, tgt) actually removed (may be <k if graph has <k edges).
    """
    rng = np.random.default_rng(seed)
    A = _EdgesToCSR(_NormalizeToEdges(graph))
    src, tgt = A.nonzero()
    m = src.size
    if k <= 0 or m == 0:
        return A, np.empty((0, 2), dtype=np.int32)
    k = min(k, m)

    if strategy == "rand":
        idx = rng.choice(m, size=k, replace=False)
        to_remove = np.vstack([src[idx], tgt[idx]]).T.astype(np.int32)

    elif strategy in ("out", "in"):
        A_csr = A.tocsr()
        outdeg = np.asarray(A_csr.sum(axis=1)).ravel()
        indeg = np.asarray(A_csr.sum(axis=0)).ravel()
        remaining = set(zip(src.tolist(), tgt.tolist()))
        removed = []
        if strategy == "out":
            order_nodes = np.argsort(-outdeg)  # sources by descending outdeg
            for u in order_nodes:
                if len(removed) >= k: break
                nbrs = A_csr.getrow(u).indices
                for v in nbrs:
                    e = (int(u), int(v))
                    if e in remaining:
                        remaining.remove(e)
                        removed.append(e)
                        if len(removed) == k: break
        else:  # 'in'
            order_nodes = np.argsort(-indeg)  # targets by descending indeg
            # build reverse adjacency index for speed
            cols_to_rows = {int(v): [] for v in np.unique(tgt)}
            for u, v in zip(src, tgt):
                cols_to_rows[int(v)].append(int(u))
            for v in order_nodes:
                if len(removed) >= k: break
                for u in cols_to_rows.get(int(v), []):
                    e = (int(u), int(v))
                    if e in remaining:
                        remaining.remove(e)
                        removed.append(e)
                        if len(removed) == k: break
        to_remove = np.asarray(removed, dtype=np.int32)

    elif strategy in ("ord", "rev"):
        if ordering_dir is None or netname is None or trial_index is None:
            raise ValueError("strategy 'ord'/'rev' requires ordering_dir, netname, trial_index.")
        edges_ord, _meta = LoadOrdEdges(netname, trial_index, ordering_dir)
        if strategy == "rev":
            edges_ord = edges_ord[::-1]
        # Remove first k edges from this order that are present
        present = set(zip(src.tolist(), tgt.tolist()))
        removed = []
        for e in edges_ord:
            tup = (int(e[0]), int(e[1]))
            if tup in present:
                present.remove(tup)
                removed.append(tup)
                if len(removed) == k:
                    break
        to_remove = np.asarray(removed, dtype=np.int32)

    else:
        raise ValueError("Unknown strategy. Choose from 'rand','out','in','ord','rev'.")

    # Apply removals
    if to_remove.size == 0:
        return A, to_remove
    A_lil = A.tolil(copy=True)
    A_lil[to_remove[:, 0], to_remove[:, 1]] = 0
    A_csr = A_lil.tocsr()
    A_csr.eliminate_zeros()
    return A_csr, to_remove


# -----------------------------
# Neuron trimming (nodes)
# -----------------------------
def TrimNeurons(graph,
                NI: int,
                n_remove_I: int = 0,
                n_remove_E: int = 0,
                strategy: str = "dout",
                seed: Optional[int] = None) -> Tuple[np.ndarray, csr_matrix]:
    """
    Remove neurons (rows & cols) according to a node-selection strategy.

    Parameters
    ----------
    graph : edges | sparse | dense (square)
    NI : int
        Number of inhibitory neurons (0..NI-1).
    n_remove_I : int
        How many inhibitory neurons to remove.
    n_remove_E : int
        How many excitatory neurons to remove.
    strategy : {'iout','ideg','rand','ddeg','dout'}
        - 'iout': remove by highest OUT-degree (row sums) within each class.
        - 'ideg': remove by highest IN-degree (col sums) within each class.
        - 'dout': alias of 'iout'.
        - 'ddeg': remove by highest TOTAL degree (out+in).
        - 'rand': random within class.
    seed : int or None
        RNG seed.

    Returns
    -------
    kept_idx : (N - n_removed,) int32
        Indices of neurons kept (in original indexing).
    A_kept : csr_matrix
        Induced subgraph on kept_idx, in internal orientation.
    """
    rng = np.random.default_rng(seed)

    A = _EdgesToCSR(_NormalizeToEdges(graph))
    N = A.shape[0]
    assert N == A.shape[1], "TrimNeurons expects a square adjacency."

    outdeg = np.asarray(A.sum(axis=1)).ravel()
    indeg = np.asarray(A.sum(axis=0)).ravel()
    totdeg = outdeg + indeg

    I = np.arange(NI, dtype=np.int32)
    E = np.arange(NI, N, dtype=np.int32)

    def pick(group: np.ndarray, how_many: int) -> np.ndarray:
        if how_many <= 0:
            return np.array([], dtype=np.int32)
        if strategy in ("iout", "dout"):
            order = group[np.argsort(-outdeg[group])]
        elif strategy == "ideg":
            order = group[np.argsort(-indeg[group])]
        elif strategy == "ddeg":
            order = group[np.argsort(-totdeg[group])]
        elif strategy == "rand":
            order = rng.permutation(group)
        else:
            raise ValueError("Unknown strategy for TrimNeurons.")
        return order[:how_many].astype(np.int32)

    rem_I = pick(I, min(n_remove_I, I.size))
    rem_E = pick(E, min(n_remove_E, E.size))
    removed = np.concatenate([rem_I, rem_E]).astype(np.int32)

    mask = np.ones(N, dtype=bool)
    mask[removed] = False
    kept_idx = np.flatnonzero(mask).astype(np.int32)

    A_kept = A[kept_idx][:, kept_idx].tocsr()
    return kept_idx, A_kept


# -----------------------------
# High-level trimming wrapper
# -----------------------------
def Trimming(graph,
             NI: int,
             syn_k: int = 0,
             syn_strategy: str = "rand",
             node_strategy: Optional[str] = None,
             n_remove_I: int = 0,
             n_remove_E: int = 0,
             ordering_dir: Optional[str] = None,
             netname: Optional[str] = None,
             trial_index: Optional[int] = None,
             seed: Optional[int] = None) -> Dict[str, object]:
    """
    High-level API to trim edges and/or neurons in one go.

    Steps
    -----
    1) Optionally trim synapses (edges) first.
    2) Optionally trim neurons (rows & cols) next.

    Returns
    -------
    result : dict with:
      - 'A' : csr_matrix     pruned adjacency
      - 'removed_edges' : (k,2) int32  (may be empty)
      - 'kept_idx' : (N',) int32       (may be full N if no node trimming)
    """
    A0 = _EdgesToCSR(_NormalizeToEdges(graph))
    removed_edges = np.empty((0, 2), dtype=np.int32)

    if syn_k and syn_k > 0:
        A1, removed_edges = TrimSynapses(A0, k=syn_k, strategy=syn_strategy,
                                         ordering_dir=ordering_dir, netname=netname,
                                         trial_index=trial_index, seed=seed)
    else:
        A1 = A0

    if node_strategy is not None and (n_remove_I > 0 or n_remove_E > 0):
        kept_idx, A2 = TrimNeurons(A1, NI=NI, n_remove_I=n_remove_I,
                                   n_remove_E=n_remove_E, strategy=node_strategy,
                                   seed=seed)
    else:
        N = A1.shape[0]; kept_idx = np.arange(N, dtype=np.int32); A2 = A1

    return {"A": A2, "removed_edges": removed_edges, "kept_idx": kept_idx}
