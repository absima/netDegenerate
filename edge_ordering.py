
# edge_ordering.py
from __future__ import annotations

import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

# ==========================================================
# ORIENTATION CONTRACT (internal)
# ----------------------------------------------------------
# - We store edges as: A[src, tgt] = 1
#   rows   = sources (left side for matching by default)
#   cols   = targets (right side for matching by default)
# - Canonical edge tuples are (src, tgt).
# - If an external routine assumes rows=targets, cols=sources, use M.T at the boundary.
# ==========================================================


# -----------------------------
# Normalization helpers
# -----------------------------
def _NormalizeToCSR(graph,
                    shape: tuple[int, int] | None = None,
                    dtype=np.int8) -> csr_matrix:
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
        M = graph.tocsr(copy=False)
        if dtype is not None and M.dtype != dtype:
            M = M.asfptype() if np.issubdtype(dtype, np.floating) else M.astype(dtype, copy=False)
        return M

    arr = np.asarray(graph)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
        # edges (src, tgt)
        src = arr[:, 0].astype(np.int64, copy=False)
        tgt = arr[:, 1].astype(np.int64, copy=False)
        n_rows = int(max(src.max() + 1 if src.size else 0, shape[0] if shape else 0))
        n_cols = int(max(tgt.max() + 1 if tgt.size else 0, shape[1] if shape else 0))
        data = np.ones(src.size, dtype=dtype)
        M = coo_matrix((data, (src, tgt)), shape=(n_rows, n_cols)).tocsr()
        M.sum_duplicates()
        return M

    # dense
    dense = np.asarray(graph)
    M = csr_matrix((dense != 0).astype(dtype))
    return M


def _EdgesFromCSR(M: csr_matrix) -> np.ndarray:
    """
    Return canonical edge list (src, tgt) from a CSR with internal orientation A[src, tgt].
    """
    src, tgt = M.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.vstack([src.astype(np.int32), tgt.astype(np.int32)]).T


# -----------------------------
# Permutation utilities
# -----------------------------
def BuildPermutationFromOrder(edges_base: np.ndarray,
                              ordered_edges: np.ndarray) -> np.ndarray:
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
    eb = np.asarray(edges_base, dtype=np.int64)
    eo = np.asarray(ordered_edges, dtype=np.int64)

    # Map edge -> list of positions in desired order
    from collections import defaultdict, deque
    pos_map: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for j, (s, t) in enumerate(eo):
        pos_map[(int(s), int(t))].append(j)

    perm = np.empty(len(eb), dtype=np.int64)
    # Offset for edges not present in ordered_edges
    unseen_offset = len(eo)
    unseen_counter = 0

    for i, (s, t) in enumerate(eb):
        key = (int(s), int(t))
        if key in pos_map and pos_map[key]:
            perm[i] = pos_map[key].popleft()
        else:
            # place after seen edges, stably
            perm[i] = unseen_offset + unseen_counter
            unseen_counter += 1

    return perm.astype(np.int32)


def SaveCompressed(edges_base: np.ndarray,
                   perm_maxmatch: np.ndarray,
                   netname: str,
                   itrial: int,
                   sizes: tuple[int, int] | None = None,
                   outdir: str | os.PathLike[str] | None = None) -> str:
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
    np.savez_compressed(path,
                        edges_base=edges_base,
                        perm_maxmatch=perm_maxmatch,
                        sizes=np.array(sizes, dtype=np.int32) if sizes is not None else None,
                        meta=np.array(meta, dtype=object))
    return path


# -----------------------------
# Hopcroft–Karp bipartite matching
# -----------------------------
def _hk_maximum_matching_csr(M: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Hopcroft–Karp on a bipartite graph represented by a CSR adjacency
    where rows are LEFT (U = sources) and cols are RIGHT (V = targets).

    Returns
    -------
    pairU : (nU,) int32  -1 if free else matched v index
    pairV : (nV,) int32  -1 if free else matched u index
    """
    M = M.tocsr(copy=False)
    nU, nV = M.shape
    indptr, indices = M.indptr, M.indices

    # adjacency function U->list(V)
    def neighbors_u(u):
        return indices[indptr[u]:indptr[u+1]]

    INF = 1 << 30
    pairU = np.full(nU, -1, dtype=np.int32)
    pairV = np.full(nV, -1, dtype=np.int32)
    dist = np.empty(nU, dtype=np.int32)

    from collections import deque

    def bfs() -> bool:
        q = deque()
        for u in range(nU):
            if pairU[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        reachable_free = False
        while q:
            u = q.popleft()
            for v in neighbors_u(u):
                pu = pairV[v]
                if pu != -1 and dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
                if pu == -1:
                    reachable_free = True
        return reachable_free

    def dfs(u: int) -> bool:
        for v in neighbors_u(u):
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = (1 << 30)  # mark as visited / exhausted
        return False

    matching = 0
    while bfs():
        for u in range(nU):
            if pairU[u] == -1:
                if dfs(u):
                    matching += 1
    return pairU, pairV


def MaxMatchFromSparse(M: csr_matrix,
                       cols_are_sources: bool = False,
                       one_based: bool = False) -> np.ndarray:
    """
    Compute a maximum matching and return matched pairs as edges (src, tgt).

    Parameters
    ----------
    M : csr_matrix
        Internal orientation by default: rows=sources, cols=targets.
    cols_are_sources : bool, default False
        If True, treat *columns* as sources (left part). We implement this by
        running HK on M.T, then mapping back to canonical (src, tgt).
        Use this if your input matrix has rows=targets, cols=sources.
    one_based : bool, default False
        If True, output edges are shifted by +1 (useful for 1-based external formats).

    Returns
    -------
    matched_edges : (k,2) int32 in canonical (src, tgt)
    """
    M = M.tocsr(copy=False)
    if cols_are_sources:
        # HK expects rows=LEFT=sources
        pairV, pairU = _hk_maximum_matching_csr(M.T)
        # pairU is for RIGHT side of M.T => columns of M => targets
        # pairV is for LEFT side of M.T  => rows of M   => sources
        matched_src = np.where(pairV != -1)[0].astype(np.int32)
        matched_tgt = pairV[matched_src]
    else:
        pairU, pairV = _hk_maximum_matching_csr(M)
        matched_src = np.where(pairU != -1)[0].astype(np.int32)
        matched_tgt = pairU[matched_src]

    edges = np.vstack([matched_src, matched_tgt]).T
    if one_based and edges.size:
        edges = edges + 1
    return edges.astype(np.int32, copy=False)


def MaxMatchDecomposition(graph,
                          cols_are_sources: bool | None = None,
                          max_layers: int | None = None,
                          one_based: bool = False) -> list[np.ndarray]:
    """
    Decompose a digraph into successive maximum matchings:
      - Repeatedly compute a maximum matching,
      - Remove those edges,
      - Continue until no edges remain or `max_layers` reached.

    Inputs can be:
      - edges array-like of shape (m, 2) in canonical (src, tgt)
      - scipy.sparse matrix (any format) oriented as A[src, tgt]
      - dense ndarray adjacency (non-zero = edge)

    Parameters
    ----------
    graph : edges | sparse | dense
    cols_are_sources : bool or None
        If None, infer from internal contract (False). Set True if your input
        uses rows=targets, cols=sources (i.e., transposed vs. our internal).
    max_layers : int or None
        Limit the number of layers (useful for previews).
    one_based : bool, default False
        If True, each layer's edges are shifted by +1.

    Returns
    -------
    layers : list of arrays, each (k_i, 2) int32 with (src, tgt)
    """
    M = _NormalizeToCSR(graph)
    if cols_are_sources is None:
        cols_are_sources = False  # internal default

    layers: list[np.ndarray] = []
    # Work on a copy we can mutate
    A = M.tolil(copy=True)  # LIL for efficient deletions
    n_iter = 0
    while A.nnz > 0 and (max_layers is None or n_iter < max_layers):
        # Compute a matching on the current snapshot
        match = MaxMatchFromSparse(A.tocsr(), cols_are_sources=cols_are_sources, one_based=False)
        if match.size == 0:
            break
        layers.append(match + (1 if one_based else 0))

        # Remove matched edges
        src = match[:, 0]
        tgt = match[:, 1]
        A[src, tgt] = 0
        A = A.tocsr()
        A.eliminate_zeros()
        A = A.tolil()
        n_iter += 1

    return layers
