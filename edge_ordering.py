# edge_ordering.py
from __future__ import annotations

import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse


def _normalizeToCsr(
    graph,
    shape: tuple[int, int] | None = None,
    dtype=np.int8,
) -> csr_matrix:
    """
    Normalize input into a CSR adjacency with internal orientation A[src, tgt].

    Parameters
    ----------
    graph : array-like | sparse matrix
        - If (m, 2) array: interpreted as canonical edges (src, tgt).
        - If sparse: converted to CSR (assumed already A[src, tgt]).
        - If dense: nonzeros become 1s (assumed already A[src, tgt]).
    shape : (n_rows, n_cols) or None
        Optional shape when graph are edges and max node < desired dimension.
    dtype : numpy dtype for data

    Returns
    -------
    csr_matrix
    """
    if issparse(graph):
        mtx = graph.tocsr(copy=False)
        if dtype is not None and mtx.dtype != dtype:
            mtx = mtx.asfptype() if np.issubdtype(dtype, np.floating) else mtx.astype(dtype, copy=False)
        return mtx

    arr = np.asarray(graph)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
        # edges (src, tgt)
        src = arr[:, 0].astype(np.int64, copy=False)
        tgt = arr[:, 1].astype(np.int64, copy=False)

        n_rows = int(max(src.max() + 1 if src.size else 0, shape[0] if shape else 0))
        n_cols = int(max(tgt.max() + 1 if tgt.size else 0, shape[1] if shape else 0))

        data = np.ones(src.size, dtype=dtype)
        mtx = coo_matrix((data, (src, tgt)), shape=(n_rows, n_cols)).tocsr()
        mtx.sum_duplicates()
        return mtx

    # dense
    dense = np.asarray(graph)
    mtx = csr_matrix((dense != 0).astype(dtype))
    return mtx


def _edgesFromCsr(mtx: csr_matrix) -> np.ndarray:
    """
    Return canonical edge list (src, tgt) from a CSR with internal orientation A[src, tgt].
    """
    src, tgt = mtx.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T




def buildPermutationFromOrder(edges_base: np.ndarray, ordered_edges: np.ndarray) -> np.ndarray:
    """
    Build a stable mapping from `edges_base` to the order defined by `ordered_edges`.

    Definitions
    -----------
    - Both inputs are arrays of shape (m, 2) in canonical (src, tgt) form.
    - We return `perm` such that: `edges_base[np.argsort(perm)] == ordered_edges`
      (up to duplicates; stable where duplicates exist).

    Implementation
    --------------
    We scan `ordered_edges` to build a mapping from edge -> queue of desired positions.
    Then, for each edge in `edges_base` (left-to-right), we pop the next desired position.
    Unseen edges (if any) are assigned positions after all seen edges, preserving stability.
    """
    edges_base_arr = np.asarray(edges_base, dtype=np.int64)
    ordered_edges_arr = np.asarray(ordered_edges, dtype=np.int64)

    from collections import defaultdict, deque

    pos_map: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for j, (src, tgt) in enumerate(ordered_edges_arr):
        pos_map[(int(src), int(tgt))].append(j)

    perm = np.empty(len(edges_base_arr), dtype=np.int64)

    unseen_offset = len(ordered_edges_arr)
    unseen_counter = 0

    for i, (src, tgt) in enumerate(edges_base_arr):
        key = (int(src), int(tgt))
        if key in pos_map and pos_map[key]:
            perm[i] = pos_map[key].popleft()
        else:
            perm[i] = unseen_offset + unseen_counter
            unseen_counter += 1

    return perm.astype(np.int32)


def saveCompressed(
    edges_base: np.ndarray,
    perm_maxmatch: np.ndarray,
    netname: str,
    itrial: int,
    sizes: tuple[int, int] | None = None,
    outdir: str | os.PathLike[str] | None = None,
) -> str:
    """
    Save ordering artifacts as a compressed .npz.

    Contents:
      - edges_base : (m,2) int32  canonical (src, tgt)
      - perm_maxmatch : (m,) int32 permutation so that edges_base[np.argsort(perm)] matches the target order
      - sizes : (2,) int32 optional (n_rows, n_cols)
      - meta : dict-like info

    Filename: {outdir}/{netname}_trial{itrial}_ordering.npz
    """
    if outdir is None:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)

    edges_base = np.asarray(edges_base, dtype=np.int32)
    perm_maxmatch = np.asarray(perm_maxmatch, dtype=np.int32)

    meta = {
        "netname": netname,
        "itrial": int(itrial),
        "orientation": "A[src, tgt]",
    }

    path = os.path.join(str(outdir), f"{netname}_trial{itrial}_ordering.npz")
    np.savez_compressed(
        path,
        edges_base=edges_base,
        perm_maxmatch=perm_maxmatch,
        sizes=np.array(sizes, dtype=np.int32) if sizes is not None else None,
        meta=np.array(meta, dtype=object),
    )
    return path


# Hopcroft–Karp bipartite matching
def _hkMaximumMatchingCsr(mtx: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Hopcroft–Karp on a bipartite graph represented by a CSR adjacency
    where rows are LEFT (U = sources) and cols are RIGHT (V = targets).

    Returns
    -------
    pair_u : (n_u,) int32  -1 if free else matched v index
    pair_v : (n_v,) int32  -1 if free else matched u index
    """
    mtx = mtx.tocsr(copy=False)
    n_u, n_v = mtx.shape
    indptr, indices = mtx.indptr, mtx.indices

    def neighbors_u(u: int):
        return indices[indptr[u] : indptr[u + 1]]

    inf = 1 << 30
    pair_u = np.full(n_u, -1, dtype=np.int32)
    pair_v = np.full(n_v, -1, dtype=np.int32)
    dist = np.empty(n_u, dtype=np.int32)

    from collections import deque

    def bfs() -> bool:
        q = deque()
        for u in range(n_u):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = inf

        reachable_free = False
        while q:
            u = q.popleft()
            for v in neighbors_u(u):
                matched_u = pair_v[v]
                if matched_u != -1 and dist[matched_u] == inf:
                    dist[matched_u] = dist[u] + 1
                    q.append(matched_u)
                if matched_u == -1:
                    reachable_free = True
        return reachable_free

    def dfs(u: int) -> bool:
        for v in neighbors_u(u):
            matched_u = pair_v[v]
            if matched_u == -1 or (dist[matched_u] == dist[u] + 1 and dfs(matched_u)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = (1 << 30)  # mark as visited / exhausted
        return False

    while bfs():
        for u in range(n_u):
            if pair_u[u] == -1:
                dfs(u)

    return pair_u, pair_v


def maxMatchFromSparse(
    mtx: csr_matrix,
    cols_are_sources: bool = False,
    one_based: bool = False,
) -> np.ndarray:
    """
    Compute a maximum matching and return matched pairs as edges (src, tgt).

    Parameters
    ----------
    mtx : csr_matrix
        Internal orientation by default: rows=sources, cols=targets.
    cols_are_sources : bool, default False
        If True, treat *columns* as sources (left part). We implement this by
        running HK on mtx.T, then mapping back to canonical (src, tgt).
        Use this if your input matrix has rows=targets, cols=sources.
    one_based : bool, default False
        If True, output edges are shifted by +1 (useful for 1-based external formats).

    Returns
    -------
    matched_edges : (k,2) int32 in canonical (src, tgt)
    """
    mtx = mtx.tocsr(copy=False)

    if cols_are_sources:
        # HK expects rows=LEFT=sources
        pair_v, pair_u = _hkMaximumMatchingCsr(mtx.T)
        matched_src = np.where(pair_v != -1)[0].astype(np.int32)
        matched_tgt = pair_v[matched_src]
    else:
        pair_u, pair_v = _hkMaximumMatchingCsr(mtx)
        matched_src = np.where(pair_u != -1)[0].astype(np.int32)
        matched_tgt = pair_u[matched_src]

    edges = np.vstack([matched_src, matched_tgt]).T
    if one_based and edges.size:
        edges = edges + 1
    return edges.astype(np.int32, copy=False)


def maxMatchDecomposition(
    graph,
    cols_are_sources: bool | None = None,
    max_layers: int | None = None,
    one_based: bool = False,
) -> list[np.ndarray]:
    """
    Decompose a digraph into successive maximum matchings:
      - Repeatedly compute a maximum matching,
      - Remove those edges,
      - Continue until no edges remain or `max_layers` reached.

    Inputs can be:
      - edges array-like of shape (m, 2) in canonical (src, tgt)
      - scipy.sparse matrix (any format) oriented as A[src, tgt]
      - dense ndarray adjacency (non-zero = edge)
    """
    mtx = _normalizeToCsr(graph)
    if cols_are_sources is None:
        cols_are_sources = False  # internal default

    layers: list[np.ndarray] = []

    # Work on a copy we can mutate
    a = mtx.tolil(copy=True)  # LIst of Lists, to preserve mtx.
    n_iter = 0

    while a.nnz > 0 and (max_layers is None or n_iter < max_layers):
        match = maxMatchFromSparse(a.tocsr(), cols_are_sources=cols_are_sources, one_based=False)
        if match.size == 0:
            break

        layers.append(match + (1 if one_based else 0))

        src = match[:, 0]
        tgt = match[:, 1]
        a[src, tgt] = 0

        a = a.tocsr()
        a.eliminate_zeros()
        a = a.tolil()

        n_iter += 1

    return layers
