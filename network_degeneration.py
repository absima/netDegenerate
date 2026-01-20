# trimming.py
from __future__ import annotations

import os
from typing import Tuple, Dict, Optional

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
def lweight(w_ii: float, w_ie: float, w_ei: float, w_ee: float) -> Dict[str, float]:
    """
    Convenience builder for a 4-block weight dictionary:
      - "II": src in I, tgt in I
      - "IE": src in I, tgt in E
      - "EI": src in E, tgt in I
      - "EE": src in E, tgt in E
    """
    return {"II": float(w_ii), "IE": float(w_ie), "EI": float(w_ei), "EE": float(w_ee)}


def bweight(w_i: float, w_e: float, cross: Optional[float] = None) -> Dict[str, float]:
    """
    Binary-style weight builder: give a weight for I-blocks and E-blocks.
    By default, cross-blocks get the geometric mean unless `cross` is provided.
    """
    w_i = float(w_i)
    w_e = float(w_e)
    if cross is None:
        cross = float(np.sqrt(w_i * w_e))
    return {"II": w_i, "EE": w_e, "IE": float(cross), "EI": float(cross)}


def scaledWeight(base: Dict[str, float], scale: float) -> Dict[str, float]:
    """
    Scale an existing block-weight dictionary by a constant factor.
    """
    scale = float(scale)
    return {k: float(v) * scale for k, v in base.items()}


# -----------------------------
# Core conversions
# -----------------------------
def _normalizeToEdges(graph, one_based: bool = False) -> np.ndarray:
    """
    Normalize input into a canonical edge list (src, tgt).
    """
    if issparse(graph):
        mtx = graph.tocsr(copy=False)
        src, tgt = mtx.nonzero()
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
    mtx = csr_matrix(arr)
    src, tgt = mtx.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T


def _edgesToCsr(edges: np.ndarray, shape: Optional[Tuple[int, int]] = None, dtype=np.int8) -> csr_matrix:
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
    mtx = coo_matrix((data, (src, tgt)), shape=(n_rows, n_cols)).tocsr()
    mtx.sum_duplicates()
    return mtx


# -----------------------------
# Weighted adjacency
# -----------------------------
def weightedFromAdjacency(
    adj_or_edges,
    ni: int,
    weights: Dict[str, float],
    neuron_order: Optional[np.ndarray] = None,
    return_sparse: bool = True,
):
    """
    Apply 4-block weights to an adjacency using internal orientation.

    Orientation & Blocks
    --------------------
    - Internal orientation: A[src, tgt].
    - Blocks are defined by (src in I/E) × (tgt in I/E):
        "II", "IE", "EI", "EE" keys in `weights`.
    """
    # Normalize to CSR
    if issparse(adj_or_edges):
        a = adj_or_edges.tocsr(copy=False)
    else:
        edges = _normalizeToEdges(adj_or_edges)
        a = _edgesToCsr(edges)

    n = a.shape[0]
    assert a.shape[0] == a.shape[1], "weightedFromAdjacency expects a square adjacency."

    # Optional symmetric relabeling
    if neuron_order is not None:
        p = np.asarray(neuron_order, dtype=np.int64)
        if p.ndim != 1 or p.size != n:
            raise ValueError("neuron_order must be a permutation of size n.")
        a = a[p][:, p]

    # Split index sets
    i_idx = np.arange(ni, dtype=np.int32)
    e_idx = np.arange(ni, n, dtype=np.int32)

    # Extract blocks and scale
    a = a.tolil(copy=True)
    if "II" in weights:
        a[np.ix_(i_idx, i_idx)] = (a[np.ix_(i_idx, i_idx)]).astype(np.float32) * float(weights["II"])
    if "IE" in weights:
        a[np.ix_(i_idx, e_idx)] = (a[np.ix_(i_idx, e_idx)]).astype(np.float32) * float(weights["IE"])
    if "EI" in weights:
        a[np.ix_(e_idx, i_idx)] = (a[np.ix_(e_idx, i_idx)]).astype(np.float32) * float(weights["EI"])
    if "EE" in weights:
        a[np.ix_(e_idx, e_idx)] = (a[np.ix_(e_idx, e_idx)]).astype(np.float32) * float(weights["EE"])

    w = a.tocsr()
    if return_sparse:
        return w
    return w.toarray()


# -----------------------------
# Ordering artifacts I/O
# -----------------------------
def loadOrdEdges(netname: str, index: int, ndir: str | os.PathLike[str] = ".") -> Tuple[np.ndarray, Dict]:
    """
    Load ordering artifact saved by `saveCompressed` (edge_ordering.py).
    """
    path = os.path.join(str(ndir), f"{netname}_trial{int(index)}_ordering.npz")
    with np.load(path, allow_pickle=True) as z:
        edges_base = z["edges_base"].astype(np.int32, copy=False)
        meta = z["meta"].item() if "meta" in z else {}
    return edges_base, meta


# -----------------------------
# Edge ordering utilities
# -----------------------------
def _edgeOrderByDegree(a: csr_matrix, mode: str = "out") -> np.ndarray:
    """
    Produce an edge ordering based on degrees.
    """
    a = a.tocsr(copy=False)
    src, tgt = a.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    outdeg = np.asarray(a.sum(axis=1)).ravel()
    indeg = np.asarray(a.sum(axis=0)).ravel()

    if mode == "out":
        keys = (-outdeg[src], tgt)
    elif mode == "in":
        keys = (-indeg[tgt], src)
    elif mode == "total":
        keys = (-(outdeg[src] + indeg[tgt]), src)
    else:
        raise ValueError("mode must be 'out','in','total'.")

    order = np.lexsort(keys[::-1])
    edges = np.vstack([src, tgt]).T.astype(np.int32)
    return edges[order]


# -----------------------------
# Edge trimming (synapses)
# -----------------------------
def trimSynapses(
    graph,
    k: int,
    strategy: str = "rand",
    ordering_dir: Optional[str] = None,
    netname: Optional[str] = None,
    trial_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Remove `k` edges according to the chosen strategy.
    """
    rng = np.random.default_rng(seed)

    a = _edgesToCsr(_normalizeToEdges(graph))
    src, tgt = a.nonzero()
    m = src.size

    if k <= 0 or m == 0:
        return a, np.empty((0, 2), dtype=np.int32)

    k = min(k, m)

    if strategy == "rand":
        idx = rng.choice(m, size=k, replace=False)
        to_remove = np.vstack([src[idx], tgt[idx]]).T.astype(np.int32)

    elif strategy in ("out", "in"):
        a_csr = a.tocsr()
        outdeg = np.asarray(a_csr.sum(axis=1)).ravel()
        indeg = np.asarray(a_csr.sum(axis=0)).ravel()

        remaining = set(zip(src.tolist(), tgt.tolist()))
        removed: list[tuple[int, int]] = []

        if strategy == "out":
            order_nodes = np.argsort(-outdeg)
            for u in order_nodes:
                if len(removed) >= k:
                    break
                nbrs = a_csr.getrow(u).indices
                for v in nbrs:
                    edge = (int(u), int(v))
                    if edge in remaining:
                        remaining.remove(edge)
                        removed.append(edge)
                        if len(removed) == k:
                            break
        else:  # 'in'
            order_nodes = np.argsort(-indeg)
            cols_to_rows: dict[int, list[int]] = {int(v): [] for v in np.unique(tgt)}
            for u, v in zip(src, tgt):
                cols_to_rows[int(v)].append(int(u))

            for v in order_nodes:
                if len(removed) >= k:
                    break
                for u in cols_to_rows.get(int(v), []):
                    edge = (int(u), int(v))
                    if edge in remaining:
                        remaining.remove(edge)
                        removed.append(edge)
                        if len(removed) == k:
                            break

        to_remove = np.asarray(removed, dtype=np.int32)

    elif strategy in ("ord", "rev"):
        if ordering_dir is None or netname is None or trial_index is None:
            raise ValueError("strategy 'ord'/'rev' requires ordering_dir, netname, trial_index.")

        edges_ord, _meta = loadOrdEdges(netname, trial_index, ordering_dir)
        if strategy == "rev":
            edges_ord = edges_ord[::-1]

        present = set(zip(src.tolist(), tgt.tolist()))
        removed: list[tuple[int, int]] = []

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

    if to_remove.size == 0:
        return a, to_remove

    a_lil = a.tolil(copy=True)
    a_lil[to_remove[:, 0], to_remove[:, 1]] = 0
    a_csr = a_lil.tocsr()
    a_csr.eliminate_zeros()
    return a_csr, to_remove


# -----------------------------
# Neuron trimming (nodes)
# -----------------------------
def trimNeurons(
    graph,
    ni: int,
    n_remove_i: int = 0,
    n_remove_e: int = 0,
    strategy: str = "dout",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, csr_matrix]:
    """
    Remove neurons (rows & cols) according to a node-selection strategy.
    """
    rng = np.random.default_rng(seed)

    a = _edgesToCsr(_normalizeToEdges(graph))
    n = a.shape[0]
    assert n == a.shape[1], "trimNeurons expects a square adjacency."

    outdeg = np.asarray(a.sum(axis=1)).ravel()
    indeg = np.asarray(a.sum(axis=0)).ravel()
    totdeg = outdeg + indeg

    i_idx = np.arange(ni, dtype=np.int32)
    e_idx = np.arange(ni, n, dtype=np.int32)

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
            raise ValueError("Unknown strategy for trimNeurons.")

        return order[:how_many].astype(np.int32)

    rem_i = pick(i_idx, min(n_remove_i, i_idx.size))
    rem_e = pick(e_idx, min(n_remove_e, e_idx.size))
    removed = np.concatenate([rem_i, rem_e]).astype(np.int32)

    mask = np.ones(n, dtype=bool)
    mask[removed] = False
    kept_idx = np.flatnonzero(mask).astype(np.int32)

    a_kept = a[kept_idx][:, kept_idx].tocsr()
    return kept_idx, a_kept


# -----------------------------
# High-level trimming wrapper
# -----------------------------
def trimming(
    graph,
    ni: int,
    syn_k: int = 0,
    syn_strategy: str = "rand",
    node_strategy: Optional[str] = None,
    n_remove_i: int = 0,
    n_remove_e: int = 0,
    ordering_dir: Optional[str] = None,
    netname: Optional[str] = None,
    trial_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    High-level API to trim edges and/or neurons in one go.
    """
    a0 = _edgesToCsr(_normalizeToEdges(graph))
    removed_edges = np.empty((0, 2), dtype=np.int32)

    if syn_k and syn_k > 0:
        a1, removed_edges = trimSynapses(
            a0,
            k=syn_k,
            strategy=syn_strategy,
            ordering_dir=ordering_dir,
            netname=netname,
            trial_index=trial_index,
            seed=seed,
        )
    else:
        a1 = a0

    if node_strategy is not None and (n_remove_i > 0 or n_remove_e > 0):
        kept_idx, a2 = trimNeurons(
            a1,
            ni=ni,
            n_remove_i=n_remove_i,
            n_remove_e=n_remove_e,
            strategy=node_strategy,
            seed=seed,
        )
    else:
        n = a1.shape[0]
        kept_idx = np.arange(n, dtype=np.int32)
        a2 = a1

    return {"A": a2, "removed_edges": removed_edges, "kept_idx": kept_idx}
