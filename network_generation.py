
# network_generation.py
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse


# ==========================================================
# ORIENTATION CONTRACT (internal, baseline for this project)
# ----------------------------------------------------------
# - We store edges as: A[src, tgt] = 1
#   rows   = sources
#   cols   = targets
# - Canonical edge tuples are (src, tgt).
# - If an external routine expects rows=targets/cols=sources, use M.T at the boundary.
# - When converting to/from edges:
#     * from matrix -> edges: (src, tgt) = M.nonzero() as (row, col)
#     * from edges  -> matrix: place ones at (row=src, col=tgt)
# ==========================================================


def _UniquePairsFromBlock(rows: np.ndarray,
                          cols: np.ndarray,
                          m: int,
                          forbid_self: bool = False,
                          rng: np.random.Generator | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample exactly m unique (src, tgt) pairs with src in `rows` and tgt in `cols`,
    uniformly without replacement.

    INTERNAL ORIENTATION: rows are sources, cols are targets.

    Parameters
    ----------
    rows : ndarray[int]
        Source indices for the row set of the Cartesian product.
    cols : ndarray[int]
        Target indices for the column set of the Cartesian product.
    m : int
        Number of unique pairs to sample.
    forbid_self : bool, default False
        If True and rows == cols (same set), enforce src != tgt (no diagonal).
    rng : np.random.Generator | int | None
        RNG or seed.

    Returns
    -------
    src_sel, tgt_sel : ndarray[int], ndarray[int]
        Selected source and target indices (length m).
    """
    rng = np.random.default_rng(rng)
    R, C = len(rows), len(cols)
    if m <= 0 or R == 0 or C == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    same_set = forbid_self and R == C and np.array_equal(rows, cols)
    if not same_set:
        n = R * C
        if m > n:
            raise ValueError(f"Requested {m} pairs but block capacity is {n}.")
        flat = rng.choice(n, size=m, replace=False)
        r_idx = flat // C
        c_idx = flat % C
        return rows[r_idx].astype(np.int32), cols[c_idx].astype(np.int32)

    # same set + forbid_self: exclude diagonal via batched rejection
    cap = R * C - R
    if m > cap:
        raise ValueError(f"Requested {m} pairs but capacity without diagonal is {cap}.")
    selected: set[tuple[int, int]] = set()
    batch = max(m, int(m * 1.25))
    while len(selected) < m:
        flat = rng.integers(0, R * C, size=batch)
        r_idx = flat // C
        c_idx = flat % C
        keep = r_idx != c_idx
        for ri, ci in zip(r_idx[keep], c_idx[keep]):
            selected.add((int(ri), int(ci)))
            if len(selected) == m:
                break
        batch = max(32, int(1.5 * (m - len(selected))))
    r_idx = np.fromiter((p[0] for p in selected), dtype=np.int32)
    c_idx = np.fromiter((p[1] for p in selected), dtype=np.int32)
    return rows[r_idx], cols[c_idx]


def BuildInterleavedIPermutation(NI: int, NE: int, N: int | None = None, step: int = 5) -> np.ndarray:
    """
    Construct a node permutation that places inhibitory (old indices 0..NI-1) at new
    positions 0, step, 2*step, ... as many as fit, and fills the rest with excitatory
    (old indices NI..NI+NE-1) in ascending order.

    Returns `perm` such that: new_index_of(old_i) = perm[old_i].

    This permutation can be applied symmetrically to rows and columns of adjacency
    since sources and targets refer to the same neuron set in square matrices.
    """
    if N is None:
        N = NI + NE
    if N != NI + NE:
        raise ValueError("If N is provided, it must equal NI + NE.")

    i_positions = np.arange(0, step * NI, step, dtype=np.int32)
    i_positions = i_positions[i_positions < N]

    new_to_old = np.empty(N, dtype=np.int32)
    used = np.zeros(N, dtype=bool)

    i_old = np.arange(NI, dtype=np.int32)
    n_i_place = min(len(i_positions), NI)
    new_to_old[i_positions[:n_i_place]] = i_old[:n_i_place]
    used[i_positions[:n_i_place]] = True

    i_remaining = i_old[n_i_place:]
    e_old = np.arange(NI, NI + NE, dtype=np.int32)
    fill_pool = np.concatenate([i_remaining, e_old], dtype=np.int32)

    free_slots = np.flatnonzero(~used)
    if free_slots.size != fill_pool.size:
        raise RuntimeError("Permutation construction mismatch.")
    new_to_old[free_slots] = fill_pool

    perm = np.empty(N, dtype=np.int32)
    perm[new_to_old] = np.arange(N, dtype=np.int32)
    return perm


def RelabelNeurons(A, perm: np.ndarray | None = None, return_sparse: bool = True):
    """
    Relabel (permute) neuron indices of a square sparse adjacency: A' = A[perm][:, perm].

    INTERNAL ORIENTATION: A[src, tgt].

    Parameters
    ----------
    A : scipy.sparse.spmatrix (square)
    perm : array-like[int] or None
        A permutation of np.arange(N). If None, a random permutation is used.
    return_sparse : bool, default True

    Returns
    -------
    A_perm : csr_matrix or ndarray
    """
    if not issparse(A):
        raise TypeError("A must be a scipy.sparse matrix.")
    N, M = A.shape
    if N != M:
        raise ValueError("A must be square (N x N).")

    if perm is None:
        rng = np.random.default_rng()
        perm = rng.permutation(N).astype(np.int32)
    else:
        perm = np.asarray(perm, dtype=np.int64)
        if perm.ndim != 1 or len(perm) != N:
            raise ValueError("perm must be a 1D array of length N.")
        if np.unique(perm).size != N or perm.min() != 0 or perm.max() != N - 1:
            raise ValueError("perm must be a permutation of 0..N-1.")
        perm = perm.astype(np.int32, copy=False)

    A_perm = A[perm][:, perm]
    if return_sparse:
        return A_perm.tocsr()
    return A_perm.toarray()


# ==========================
# Generators (Null Models)
# ==========================

def RandomizeBinaryMatrix(obmtx, return_sparse: bool = True, rng=None):
    """
    Randomize a binary block by shuffling the positions of ones while preserving the exact count.
    For square blocks, self-loops (diagonal) are forbidden.

    INTERNAL ORIENTATION: returns a matrix with ones at (src, tgt).

    Parameters
    ----------
    obmtx : array-like of {0,1}, shape (nrow, ncol)
    return_sparse : bool, default True
    rng : np.random.Generator | int | None

    Returns
    -------
    nbmtx : csr_matrix or ndarray
        Randomized binary block with the same number of ones as `obmtx`.
    """
    b = np.asarray(obmtx, dtype=np.int8)
    nrow, ncol = b.shape
    numE = int(b.sum())
    if numE == 0:
        return csr_matrix((nrow, ncol), dtype=np.int8) if return_sparse else np.zeros_like(b, dtype=np.int8)

    rows = np.arange(nrow, dtype=np.int32)  # sources
    cols = np.arange(ncol, dtype=np.int32)  # targets
    forbid_self = (nrow == ncol)
    r_sel, c_sel = _UniquePairsFromBlock(rows, cols, numE, forbid_self=forbid_self, rng=rng)

    if return_sparse:
        data = np.ones(len(r_sel), dtype=np.int8)
        A = csr_matrix((data, (r_sel, c_sel)), shape=(nrow, ncol))
        A.sum_duplicates()
        return A

    nbmtx = np.zeros((nrow, ncol), dtype=np.int8)
    nbmtx[r_sel, c_sel] = 1
    if forbid_self:
        np.fill_diagonal(nbmtx, 0)
    return nbmtx


def RandomBlockFromEmpirical(bmtx_emp, NI: int, strengths: dict | None = None,
                             rng=None, return_sparse: bool = True):
    """
    Preserve block-wise edge counts of an empirical I/E microcircuit (I→I, I→E, E→I, E→E),
    randomize connections uniformly within each block, and optionally assign per-block weights.

    INTERNAL ORIENTATION:
      - Input is interpreted as A[src, tgt].
      - Block (I→E) means sources in I, targets in E, etc.

    Parameters
    ----------
    bmtx_emp : (N x N) array-like or scipy.sparse.spmatrix
        Empirical adjacency (binary/weighted). Nonzero at A[src, tgt] means src→tgt.
    NI : int
        Number of inhibitory neurons (first NI indices are I; the rest are E).
    strengths : dict or None
        Optional weights per block: {"II","IE","EI","EE"} -> float. If None, edges weight=1.
    rng : np.random.Generator | int | None
    return_sparse : bool, default True

    Returns
    -------
    A : csr_matrix or ndarray
        Randomized adjacency with same edge counts per block.
    counts_emp : dict
        Empirical edge counts per block (diagonal excluded).
    counts_new : dict
        Edge counts per block in the generated A (should match counts_emp).
    """
    rng = np.random.default_rng(rng)

    if issparse(bmtx_emp):
        Aemp = bmtx_emp.tocsr(copy=False)
        N = Aemp.shape[0]
        # zero diagonal for counting
        Aemp = Aemp - csr_matrix((Aemp.diagonal(), (np.arange(N), np.arange(N))), shape=Aemp.shape)
        Aemp.eliminate_zeros()
        Aemp.data[:] = 1
        is_sparse = True
    else:
        Aemp = (np.asarray(bmtx_emp) != 0).astype(np.int8)
        N = Aemp.shape[0]
        np.fill_diagonal(Aemp, 0)
        is_sparse = False

    I = np.arange(NI, dtype=np.int32)
    E = np.arange(NI, N, dtype=np.int32)

    if is_sparse:
        def _count(A, r, c): return int(A[r][:, c].nnz)
        c_II = _count(Aemp, I, I); c_IE = _count(Aemp, I, E)
        c_EI = _count(Aemp, E, I); c_EE = _count(Aemp, E, E)
    else:
        c_II = int(Aemp[np.ix_(I, I)].sum()); c_IE = int(Aemp[np.ix_(I, E)].sum())
        c_EI = int(Aemp[np.ix_(E, I)].sum()); c_EE = int(Aemp[np.ix_(E, E)].sum())
    counts_emp = {"II": c_II, "IE": c_IE, "EI": c_EI, "EE": c_EE}

    r_II, c_IIr = _UniquePairsFromBlock(I, I, c_II, forbid_self=True,  rng=rng)
    r_IE, c_IEr = _UniquePairsFromBlock(I, E, c_IE, forbid_self=False, rng=rng)
    r_EI, c_EIr = _UniquePairsFromBlock(E, I, c_EI, forbid_self=False, rng=rng)
    r_EE, c_EEr = _UniquePairsFromBlock(E, E, c_EE, forbid_self=True,  rng=rng)

    rows = np.concatenate([r_II, r_IE, r_EI, r_EE], dtype=np.int32)
    cols = np.concatenate([c_IIr, c_IEr, c_EIr, c_EEr], dtype=np.int32)

    if strengths is None:
        data = np.ones(len(rows), dtype=np.float32)
    else:
        w_II = np.full(len(r_II), float(strengths.get("II", 1.0)), dtype=np.float32)
        w_IE = np.full(len(r_IE), float(strengths.get("IE", 1.0)), dtype=np.float32)
        w_EI = np.full(len(r_EI), float(strengths.get("EI", 1.0)), dtype=np.float32)
        w_EE = np.full(len(r_EE), float(strengths.get("EE", 1.0)), dtype=np.float32)
        data = np.concatenate([w_II, w_IE, w_EI, w_EE], dtype=np.float32)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    A.sum_duplicates()
    counts_new = {"II": len(r_II), "IE": len(r_IE), "EI": len(r_EI), "EE": len(r_EE)}

    if return_sparse:
        return A, counts_emp, counts_new
    return A.toarray(), counts_emp, counts_new


def GenerateDecayMatrix(size: int = 1000, total_ones: int = 100_000,
                        decay_rate: float = 4.0, power: float = 2.0,
                        shape: tuple[int, int] | None = None,
                        forbid_diagonal: bool = True,
                        rng=None, return_sparse: bool = True):
    """
    Generate a binary adjacency with density decaying from top-left:
    prob ~ exp(-decay_rate * distance**power). Useful as an unrelabeled null.

    INTERNAL ORIENTATION: returns A[src, tgt].

    Parameters
    ----------
    size : int, default 1000
    total_ones : int, default 100_000
    decay_rate : float, default 4.0
    power : float, default 2.0
    shape : (rows, cols) or None
        If provided, overrides `size` (can generate rectangular blocks).
    forbid_diagonal : bool, default True
        When square, excludes i==j (no self-loops).
    rng : np.random.Generator | int | None
    return_sparse : bool, default True

    Returns
    -------
    A : csr_matrix or ndarray
    """
    rng = np.random.default_rng(rng)

    if shape is None:
        N = int(size); n_rows = n_cols = N
    else:
        n_rows, n_cols = map(int, shape)

    # rows = sources, cols = targets
    x = np.linspace(0, 1, n_cols, dtype=np.float64)
    y = np.linspace(0, 1, n_rows, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)
    prob = np.exp(-decay_rate * (distance ** power))

    if forbid_diagonal and n_rows == n_cols:
        np.fill_diagonal(prob, 0.0)

    p = prob.ravel()
    s = p.sum()
    if s <= 0:
        p[:] = 1.0
        if forbid_diagonal and n_rows == n_cols:
            p.reshape(n_rows, n_cols)[np.diag_indices(n_rows)] = 0.0
        s = p.sum()
    p /= s

    capacity = n_rows * n_cols - (n_rows if (forbid_diagonal and n_rows == n_cols) else 0)
    k = int(min(total_ones, capacity))

    idx = rng.choice(p.size, size=k, replace=False, p=p)
    r = (idx // n_cols).astype(np.int32)
    c = (idx %  n_cols).astype(np.int32)

    if return_sparse:
        data = np.ones(k, dtype=np.int8)
        A = coo_matrix((data, (r, c)), shape=(n_rows, n_cols)).tocsr()
        A.sum_duplicates()
        return A

    A = np.zeros((n_rows, n_cols), dtype=np.int8)
    A[r, c] = 1
    return A


def SmallWorldDirected(N: int, nE: int, prand: float = 0.02,
                       forbid_self: bool = True, rng=None, return_sparse: bool = True):
    """
    Directed small-world–like digraph via ring-lattice + rewiring. Returns exactly nE edges.

    INTERNAL ORIENTATION: returns A[src, tgt].

    Construction (baseline):
      1) Start with k = ceil(nE / N) outgoing edges per node on a ring lattice,
         connecting each node i to its next k distinct successors: (i -> i+1, ..., i+k).
      2) Remove self-loops if requested.
      3) Deduplicate and adjust to exactly nE edges (sample/remove as needed).
      4) Rewire a fraction `prand` of edges uniformly to non-edges while avoiding duplicates
         and (optionally) self-loops.

    Parameters
    ----------
    N : int
        Number of nodes.
    nE : int
        Target number of directed edges.
    prand : float, default 0.02
        Fraction of edges to rewire to random non-edges.
    forbid_self : bool, default True
        Disallow self-loops.
    rng : np.random.Generator | int | None
    return_sparse : bool, default True

    Returns
    -------
    A : csr_matrix or ndarray
    """
    rng = np.random.default_rng(rng)

    k_max = max(0, N - 1) if forbid_self else N
    k = int(np.ceil(nE / N))
    k = int(min(max(0, k), k_max))

    if k == 0:
        rows = np.array([], dtype=np.int32); cols = np.array([], dtype=np.int32)
    else:
        rows = np.repeat(np.arange(N, dtype=np.int32), k)  # sources
        t = np.arange(1, k + 1, dtype=np.int32)
        cols = (np.arange(N, dtype=np.int32)[:, None] + t) % N  # targets are forward neighbors
        cols = cols.reshape(-1).astype(np.int32)
        if forbid_self:
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]

    if rows.size:
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        dedup = np.ones(rows.size, dtype=bool)
        dedup[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        rows, cols = rows[dedup], cols[dedup]

    E0 = rows.size
    if E0 > nE:
        take = rng.choice(E0, size=nE, replace=False)
        rows, cols = rows[take], cols[take]
    elif E0 < nE:
        need = nE - E0
        edge_set = set(zip(rows.tolist(), cols.tolist()))
        batch = max(need, int(1.25 * need))
        add_r, add_c = [], []
        while need > 0:
            r = rng.integers(0, N, size=batch, dtype=np.int64)
            c = rng.integers(0, N, size=batch, dtype=np.int64)
            if forbid_self:
                keep = r != c
                r, c = r[keep], c[keep]
            for ri, ci in zip(r, c):
                key = (int(ri), int(ci))
                if key not in edge_set:
                    edge_set.add(key)
                    add_r.append(key[0]); add_c.append(key[1])
                    need -= 1
                    if need == 0: break
            batch = max(32, int(1.5 * need))
        if add_r:
            rows = np.concatenate([rows, np.array(add_r, dtype=np.int32)])
            cols = np.concatenate([cols, np.array(add_c, dtype=np.int32)])

    m = rows.size
    if prand > 0 and m > 0:
        n_rewire = int(np.floor(prand * m))
        if n_rewire > 0:
            idx_remove = rng.choice(m, size=n_rewire, replace=False)
            keep_mask = np.ones(m, dtype=bool); keep_mask[idx_remove] = False
            rows_keep, cols_keep = rows[keep_mask], cols[keep_mask]
            edge_set = set(zip(rows_keep.tolist(), cols_keep.tolist()))

            new_r, new_c = [], []
            need = n_rewire; batch = max(need, int(1.25 * need))
            while need > 0:
                r = rng.integers(0, N, size=batch, dtype=np.int64)
                c = rng.integers(0, N, size=batch, dtype=np.int64)
                if forbid_self:
                    keep = r != c
                    r, c = r[keep], c[keep]
                for ri, ci in zip(r, c):
                    key = (int(ri), int(ci))
                    if key not in edge_set:
                        edge_set.add(key)
                        new_r.append(key[0]); new_c.append(key[1])
                        need -= 1
                        if need == 0: break
                batch = max(32, int(1.5 * need))

            rows = np.concatenate([rows_keep, np.array(new_r, dtype=np.int32)])
            cols = np.concatenate([cols_keep, np.array(new_c, dtype=np.int32)])

    data = np.ones(rows.size, dtype=np.int8)
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.sum_duplicates()
    if return_sparse:
        return A
    return A.toarray()


# =========
# Wrapper
# =========

def GenerateNet(cemtx,
                netname: str,
                NI: int | None = None,
                strengths: dict | None = None,
                step: int = 5,
                prand: float = 0.02,
                rng=None,
                return_sparse: bool = True):
    """
    Unified generator wrapper.

    INTERNAL ORIENTATION for all outputs: A[src, tgt] = 1.

    netname:
      'emp' : return input as CSR (or dense if return_sparse=False)
      'erb' : RandomBlockFromEmpirical (needs NI)
      'ero' : global ER (same nnz as cemtx; no self-loops)
      'sfo' : top-left–biased decay (same nnz)
      'sfr' : decay then relabel I interleaved (needs NI)
      'swr' : small-world then relabel I interleaved (needs NI)

    All branches return CSR if return_sparse=True, else dense ndarray.
    """
    rng = np.random.default_rng(rng)
    Aemp = cemtx.tocsr(copy=False) if issparse(cemtx) else csr_matrix(np.asarray(cemtx))
    N = Aemp.shape[0]
    nE = int(Aemp.nnz)

    if netname == 'emp':
        return Aemp if return_sparse else Aemp.toarray()

    if netname == 'erb':
        if NI is None:
            raise ValueError("NI is required for 'erb'.")
        A_block, _, _ = RandomBlockFromEmpirical(Aemp, NI=NI, strengths=strengths, rng=rng, return_sparse=True)
        return A_block if return_sparse else A_block.toarray()

    if netname == 'ero':
        rows = np.arange(N, dtype=np.int32)
        cols = np.arange(N, dtype=np.int32)
        r_sel, c_sel = _UniquePairsFromBlock(rows, cols, m=nE, forbid_self=True, rng=rng)
        data = np.ones(len(r_sel), dtype=np.int8)
        A = csr_matrix((data, (r_sel, c_sel)), shape=(N, N)); A.sum_duplicates()
        return A if return_sparse else A.toarray()

    if netname == 'sfo':
        return GenerateDecayMatrix(shape=(N, N), total_ones=nE,
                                   decay_rate=4.0, power=1.0,
                                   forbid_diagonal=True, rng=rng,
                                   return_sparse=return_sparse)

    if netname == 'sfr':
        if NI is None:
            raise ValueError("NI is required for 'sfr'.")
        NE = N - NI
        A = GenerateDecayMatrix(shape=(N, N), total_ones=nE,
                                decay_rate=4.0, power=1.0,
                                forbid_diagonal=True, rng=rng,
                                return_sparse=True)
        perm = BuildInterleavedIPermutation(NI=NI, NE=NE, N=N, step=step)
        return RelabelNeurons(A, perm=perm, return_sparse=return_sparse)

    if netname == 'swr':
        if NI is None:
            raise ValueError("NI is required for 'swr'.")
        NE = N - NI  # kept for symmetry/documentation
        A = SmallWorldDirected(N=N, nE=nE, prand=prand, forbid_self=True, rng=rng, return_sparse=True)
        perm = BuildInterleavedIPermutation(NI=NI, NE=NE, N=N, step=step)
        return RelabelNeurons(A, perm=perm, return_sparse=return_sparse)

    raise ValueError(f"Invalid netname: {netname}. Choose from 'emp','erb','ero','sfo','sfr','swr'.")
