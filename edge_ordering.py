from __future__ import annotations

import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse


## convention:
# - Edge list: (src, tgt)
# - Matrix (dense OR sparse): M[tgt, src] = 1 means src -> tgt
#
# Conversions:
#   matrix -> edges: (src, tgt) = (col, row)
#   edges  -> matrix: put ones at (row=tgt, col=src)


def _normalizeToCsr(graph, shape=None, dtype=np.int8) -> csr_matrix:
    """
    Normalize input to CSR connectivity matrix.

    Args:
        graph: edges (m,2) as (src,tgt) OR matrix already stored as M[tgt,src]
        shape: optional (n_rows, n_cols) override when graph is edges
        dtype: dtype for returned CSR (None keeps current dtype for sparse input)

    Returns:
        CSR matrix stored as M[tgt,src]
    """
    if issparse(graph):
        mtx = graph.tocsr(copy=False)
        if dtype is not None and mtx.dtype != dtype:
            mtx = mtx.astype(dtype, copy=False)
        return mtx

    arr = np.asarray(graph)

    # Edge list (src, tgt)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind in "iu":
        src = arr[:, 0].astype(np.int64, copy=False)
        tgt = arr[:, 1].astype(np.int64, copy=False)

        if shape is None:
            n_rows = int(tgt.max() + 1) if tgt.size else 0
            n_cols = int(src.max() + 1) if src.size else 0
        else:
            n_rows = int(max((tgt.max() + 1) if tgt.size else 0, shape[0]))
            n_cols = int(max((src.max() + 1) if src.size else 0, shape[1]))

        data = np.ones(src.size, dtype=dtype)
        mtx = coo_matrix((data, (tgt, src)), shape=(n_rows, n_cols)).tocsr()
        mtx.sum_duplicates()
        return mtx

    # Dense matrix already follows Convention B (M[tgt,src])
    dense = arr
    return csr_matrix((dense != 0).astype(dtype, copy=False))


def saveCompressed(edges_base, perm_maxmatch, netname, itrial, sizes=None, outdir=None):
    """
    Save ordering artifacts as a compressed .npz.

    Args:
        edges_base: (m,2) edges (src,tgt)
        perm_maxmatch: (m,) permutation scores (smaller = earlier)
        netname: label for the network
        itrial: trial index
        sizes: optional shape tuple/list for the original matrix
        outdir: output directory (defaults to ".")

    Returns:
        path to saved .npz file
    """
    if outdir is None:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)

    edges_base = np.asarray(edges_base, dtype=np.int32)
    perm_maxmatch = np.asarray(perm_maxmatch, dtype=np.int32)

    meta = {
        "netname": netname,
        "itrial": int(itrial),
        "orientation_matrix": "M[tgt, src] means src->tgt",
        "orientation_edges": "(src, tgt)",
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


def _hkMaximumMatchingCsr(mtx: csr_matrix):
    """
    Hopcroft–Karp maximum matching on bipartite CSR adjacency.

    Args:
        mtx: CSR adjacency where rows = LEFT partition, cols = RIGHT partition

    Returns:
        (pair_u, pair_v) where:
            pair_u[u] = matched v or -1
            pair_v[v] = matched u or -1
    """
    mtx = mtx.tocsr(copy=False)
    n_u, n_v = mtx.shape
    indptr, indices = mtx.indptr, mtx.indices

    def neighbors_u(u):
        return indices[indptr[u] : indptr[u + 1]]

    inf = 1 << 30
    pair_u = np.full(n_u, -1, dtype=np.int32)
    pair_v = np.full(n_v, -1, dtype=np.int32)
    dist = np.empty(n_u, dtype=np.int32)

    from collections import deque

    def bfs():
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

    def dfs(u):
        for v in neighbors_u(u):
            matched_u = pair_v[v]
            if matched_u == -1 or (dist[matched_u] == dist[u] + 1 and dfs(matched_u)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = inf
        return False

    while bfs():
        for u in range(n_u):
            if pair_u[u] == -1:
                dfs(u)

    return pair_u, pair_v


def maxMatchFromSparse(mtx: csr_matrix, one_based=False, relabel=True, rng=None) -> np.ndarray:
    """
    Compute one maximum matching from a connectivity matrix.

    Args:
        mtx: CSR matrix stored as M[tgt,src]
        one_based: if True, return edges in 1-based indexing
        relabel: if True, random relabeling before HK to de-bias ties
        rng: seed or Generator

    Returns:
        edges: (k,2) int32 array of matched edges (src,tgt)
    """
    rng = np.random.default_rng(rng)
    mtx = mtx.tocsr(copy=False)

    # Convention B: rows=tgt, cols=src. HK expects rows=src, cols=tgt => transpose.
    b = mtx.T.tocsr(copy=False)  # rows=src, cols=tgt

    n_src, n_tgt = b.shape

    if relabel and (n_src > 1 or n_tgt > 1):
        perm_src = rng.permutation(n_src).astype(np.int32)
        perm_tgt = rng.permutation(n_tgt).astype(np.int32)

        pair_src, _ = _hkMaximumMatchingCsr(b[perm_src][:, perm_tgt])

        src_perm = np.where(pair_src != -1)[0].astype(np.int32)
        tgt_perm = pair_src[src_perm].astype(np.int32)

        # Map permuted indices back to original indices
        src = perm_src[src_perm].astype(np.int32, copy=False)
        tgt = perm_tgt[tgt_perm].astype(np.int32, copy=False)
    
    else:
        pair_src, _ = _hkMaximumMatchingCsr(b)
        src = np.where(pair_src != -1)[0].astype(np.int32)
        tgt = pair_src[src].astype(np.int32)

    edges = np.vstack([src, tgt]).T.astype(np.int32, copy=False)
    if one_based and edges.size:
        edges = edges + 1
    return edges


def maxMatchDecomposition(graph, max_layers=None, one_based=False, relabel=True, rng=None):
    """
    Decompose a graph into successive maximum matchings.

    Args:
        graph: edges (src,tgt) or matrix stored as M[tgt,src]
        max_layers: optional maximum number of layers to extract
        one_based: if True, output edges are 1-based
        relabel: if True, random relabeling before HK each layer
        rng: seed or Generator

    Returns:
        (edges_concat, layer_sizes)
        - edges_concat: (m,2) int32 edges (src,tgt) concatenated by layers
        - layer_sizes: (L,) int32 sizes for each layer
    """
    rng = np.random.default_rng(rng)

    mtx = _normalizeToCsr(graph)
    a = mtx.tolil(copy=True)

    edges_parts = []
    layer_sizes = []

    n_iter = 0
    prev_nnz = a.nnz

    while a.nnz > 0 and (max_layers is None or n_iter < max_layers):
        match = maxMatchFromSparse(a.tocsr(), one_based=False, relabel=relabel, rng=rng)
        if match.size == 0:
            break

        edges_parts.append(match + (1 if one_based else 0))
        layer_sizes.append(match.shape[0])

        # IMPORTANT: SciPy sparse fancy assignment with two index arrays is not reliably pairwise.
        # Use explicit paired deletions to guarantee correctness.
        for s, t in match:
            a[int(t), int(s)] = 0  # delete at (row=tgt, col=src)

        a = a.tocsr()
        a.eliminate_zeros()

        # Guard against runaway if deletion fails for any reason
        if a.nnz >= prev_nnz:
            raise RuntimeError(
                f"Residual did not shrink in maxMatchDecomposition: prev nnz={prev_nnz}, new nnz={a.nnz}. "
                "Edge removal failed."
            )
        prev_nnz = a.nnz

        a = a.tolil()
        n_iter += 1

    if edges_parts:
        edges_concat = np.vstack(edges_parts).astype(np.int32, copy=False)
    else:
        edges_concat = np.empty((0, 2), dtype=np.int32)

    return edges_concat, np.asarray(layer_sizes, dtype=np.int32)



