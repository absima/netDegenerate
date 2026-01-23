from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import coo_matrix, csr_matrix, issparse, load_npz


# ==========================================================
# CONVENTION B (THIS FILE)
# ----------------------------------------------------------
# - Edge list: (src, tgt)
# - Matrix (dense or sparse): M[tgt, src] = 1 means src -> tgt
# - Therefore: CSR/COO indices are (row=tgt, col=src)
# ==========================================================


def _uniquePairsFromBlock(rows, cols, m, forbid_self=False, rng=None):
    """
    Sample exactly m unique (src, tgt) pairs from rows × cols (no replacement).

    Args:
        rows: 1D array of candidate src indices (values used as src ids)
        cols: 1D array of candidate tgt indices (values used as tgt ids)
        m: number of pairs to sample
        forbid_self: if True and rows==cols, exclude diagonal pairs
        rng: seed or Generator

    Returns:
        (src_sel, tgt_sel) int32 arrays of length m
    """
    rng = np.random.default_rng(rng)

    n_rows = len(rows)
    n_cols = len(cols)
    if m <= 0 or n_rows == 0 or n_cols == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    same_set = forbid_self and n_rows == n_cols and np.array_equal(rows, cols)

    if not same_set:
        capacity = n_rows * n_cols
        if m > capacity:
            raise ValueError(f"Requested {m} pairs but block capacity is {capacity}.")
        flat = rng.choice(capacity, size=m, replace=False)
        row_idx = flat // n_cols
        col_idx = flat % n_cols
        return rows[row_idx].astype(np.int32), cols[col_idx].astype(np.int32)

    # same_set + forbid_self => exclude diagonal
    capacity = n_rows * n_cols - n_rows
    if m > capacity:
        raise ValueError(f"Requested {m} pairs but capacity without diagonal is {capacity}.")

    selected = set()
    batch = max(m, int(m * 1.25))

    while len(selected) < m:
        flat = rng.integers(0, n_rows * n_cols, size=batch)
        row_idx = flat // n_cols
        col_idx = flat % n_cols
        keep = row_idx != col_idx

        for ri, ci in zip(row_idx[keep], col_idx[keep]):
            selected.add((int(ri), int(ci)))
            if len(selected) == m:
                break

        batch = max(32, int(1.5 * (m - len(selected))))

    row_idx = np.fromiter((p[0] for p in selected), dtype=np.int32)
    col_idx = np.fromiter((p[1] for p in selected), dtype=np.int32)
    return rows[row_idx], cols[col_idx]


def buildInterleavedIPermutation(NI, NE, N=None, step=5):
    """
    Build a permutation that interleaves inhibitory indices every `step` positions.

    Args:
        NI: number of inhibitory nodes (0..NI-1)
        NE: number of excitatory nodes (NI..NI+NE-1)
        N: total nodes (defaults to NI+NE)
        step: spacing for inhibitory placements

    Returns:
        perm: int32 permutation mapping old_index -> new_index
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


def relabelNeurons(cmtx, perm=None, return_sparse=True):
    """
    Relabel nodes of a square connectivity matrix by a permutation.

    Args:
        cmtx: square sparse matrix (CSR/COO/etc.)
        perm: permutation mapping old_index -> new_index (length N), or None for random
        return_sparse: CSR if True else dense

    Returns:
        Relabeled matrix (CSR or dense), still using Convention B storage (row=tgt, col=src)
    """
    if not issparse(cmtx):
        raise TypeError("cmtx must be a scipy.sparse matrix")

    N, M = cmtx.shape
    if N != M:
        raise ValueError("cmtx must be square")

    if perm is None:
        perm = np.random.permutation(N)
    else:
        perm = np.asarray(perm)
        if perm.ndim != 1 or perm.size != N:
            raise ValueError("perm must have length N")

    cmtx = cmtx.tocoo()
    new_rows = perm[cmtx.row]  # new tgt
    new_cols = perm[cmtx.col]  # new src

    a = coo_matrix((cmtx.data, (new_rows, new_cols)), shape=cmtx.shape)
    a.sum_duplicates()
    if a.data.size:
        a.data[:] = 1

    a = a.tocsr()
    return a if return_sparse else a.toarray()


import numpy as np
import scipy.sparse as sp


def buildPrototypeTemplate(N, NI, densities, rng=None, return_sparse=True):
    """
    Build a synthetic template with distinct I/E block densities.

    Convention B:
        Matrix stores M[tgt, src] = 1 for src -> tgt.

    Args:
        N: total number of nodes
        NI: number of inhibitory nodes (0..NI-1)
        densities: dict {"II","EI","IE","EE"} with:
            II: I -> I
            EI: I -> E  (rows=E, cols=I)
            IE: E -> I  (rows=I, cols=E)
            EE: E -> E
        rng: seed or np.random.Generator
        return_sparse: CSR if True else dense

    Returns:
        Binary connectivity matrix stored as M[tgt,src] (CSR or dense).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if not (0 < NI < N):
        raise ValueError(f"NI must satisfy 0 < NI < N, got NI={NI}, N={N}.")

    rng = np.random.default_rng(rng)
    NE = N - NI

    def _clip_density(x):
        return float(np.clip(float(x), 0.0, 1.0))

    d_ii = _clip_density(densities.get("II", 0.0))
    d_ei = _clip_density(densities.get("EI", 0.0))  # I -> E
    d_ie = _clip_density(densities.get("IE", 0.0))  # E -> I
    d_ee = _clip_density(densities.get("EE", 0.0))

    def rand_block(n_rows, n_cols, density):
        # Critical guard: avoid SciPy sampling with a=0
        if n_rows == 0 or n_cols == 0 or density <= 0.0:
            return sp.csr_matrix((n_rows, n_cols), dtype=np.int8)

        # For tiny blocks, it can be useful to avoid calling sparse.random with an
        # ultra-small expected nnz, but it's not strictly necessary.
        return sp.random(
            n_rows,
            n_cols,
            density=density,
            data_rvs=lambda k: np.ones(k, dtype=np.int8),
            format="csr",
            random_state=int(rng.integers(0, 2**32 - 1)),
        ).astype(np.int8)

    # Convention B blocks (rows=tgt, cols=src)
    block_ii = rand_block(NI, NI, d_ii)   # I -> I : rows I, cols I
    block_ei = rand_block(NE, NI, d_ei)   # I -> E : rows E, cols I
    block_ie = rand_block(NI, NE, d_ie)   # E -> I : rows I, cols E
    block_ee = rand_block(NE, NE, d_ee)   # E -> E : rows E, cols E

    # Assemble blocks safely
    a = sp.bmat([[block_ii, block_ie],
                [block_ei, block_ee]], format="csr", dtype=np.int8)

    a.sum_duplicates()
    if a.nnz:
        a.data[:] = 1
    a.setdiag(0)
    a.eliminate_zeros()

    return a if return_sparse else a.toarray()



def randomizeBinaryMatrix(ob_mtx, return_sparse=True, rng=None):
    """
    Randomize ones while preserving the count.

    Args:
        ob_mtx: binary matrix (dense or sparse accepted via np.asarray)
        return_sparse: CSR if True else dense
        rng: seed or Generator

    Returns:
        Randomized binary matrix with the same shape and number of ones (stored as M[tgt,src])
    """
    b = np.asarray(ob_mtx, dtype=np.int8)
    n_rows, n_cols = b.shape
    num_edges = int(b.sum())

    if num_edges == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.int8) if return_sparse else np.zeros_like(b, dtype=np.int8)

    # rows=tgt, cols=src; sampler returns (src,tgt)
    src_pool = np.arange(n_cols, dtype=np.int32)
    tgt_pool = np.arange(n_rows, dtype=np.int32)

    forbid_self = (n_rows == n_cols)
    src_sel, tgt_sel = _uniquePairsFromBlock(src_pool, tgt_pool, num_edges, forbid_self=forbid_self, rng=rng)

    if return_sparse:
        data = np.ones(len(src_sel), dtype=np.int8)
        a = csr_matrix((data, (tgt_sel, src_sel)), shape=(n_rows, n_cols))
        a.sum_duplicates()
        if a.nnz:
            a.data[:] = 1
        if forbid_self:
            a.setdiag(0)
            a.eliminate_zeros()
        return a

    nb = np.zeros((n_rows, n_cols), dtype=np.int8)
    nb[tgt_sel, src_sel] = 1
    if forbid_self:
        np.fill_diagonal(nb, 0)
    return nb


def randomBlockFromEmpirical(b_mtx_emp, NI, strengths=None, rng=None, return_sparse=True):
    """
    Preserve I/E block edge counts and randomize connections within each block.

    Args:
        b_mtx_emp: empirical matrix (sparse or dense), stored as M[tgt,src]
        NI: number of inhibitory nodes
        strengths: optional weights dict {"II","EI","IE","EE"}
        rng: seed or Generator
        return_sparse: CSR if True else dense

    Returns:
        (A, counts_emp, counts_new)
        - A: randomized blockwise matrix (stored as M[tgt,src])
        - counts_emp: dict of empirical block edge counts
        - counts_new: dict of realized block edge counts
    """
    rng = np.random.default_rng(rng)

    if issparse(b_mtx_emp):
        a_emp = b_mtx_emp.tocsr(copy=False)
        N = a_emp.shape[0]
        a_emp.setdiag(0)
        a_emp.eliminate_zeros()
        a_emp.sum_duplicates()
        if a_emp.nnz:
            a_emp.data[:] = 1
        is_sparse = True
    else:
        a_emp = (np.asarray(b_mtx_emp) != 0).astype(np.int8)
        N = a_emp.shape[0]
        np.fill_diagonal(a_emp, 0)
        is_sparse = False

    i_idx = np.arange(NI, dtype=np.int32)
    e_idx = np.arange(NI, N, dtype=np.int32)

    if is_sparse:
        def _count(a_mat, r, c):
            return int(a_mat[r][:, c].nnz)

        c_ii = _count(a_emp, i_idx, i_idx)
        c_ei = _count(a_emp, e_idx, i_idx)  # I->E
        c_ie = _count(a_emp, i_idx, e_idx)  # E->I
        c_ee = _count(a_emp, e_idx, e_idx)
    else:
        c_ii = int(a_emp[np.ix_(i_idx, i_idx)].sum())
        c_ei = int(a_emp[np.ix_(e_idx, i_idx)].sum())
        c_ie = int(a_emp[np.ix_(i_idx, e_idx)].sum())
        c_ee = int(a_emp[np.ix_(e_idx, e_idx)].sum())

    counts_emp = {"II": c_ii, "EI": c_ei, "IE": c_ie, "EE": c_ee}

    # Sample edges as (src,tgt), then store as (tgt,src)
    src_ii, tgt_ii = _uniquePairsFromBlock(i_idx, i_idx, c_ii, forbid_self=True, rng=rng)
    src_ei, tgt_ei = _uniquePairsFromBlock(i_idx, e_idx, c_ei, forbid_self=False, rng=rng)  # I->E
    src_ie, tgt_ie = _uniquePairsFromBlock(e_idx, i_idx, c_ie, forbid_self=False, rng=rng)  # E->I
    src_ee, tgt_ee = _uniquePairsFromBlock(e_idx, e_idx, c_ee, forbid_self=True, rng=rng)

    rows = np.concatenate([tgt_ii, tgt_ei, tgt_ie, tgt_ee], dtype=np.int32)  # tgt
    cols = np.concatenate([src_ii, src_ei, src_ie, src_ee], dtype=np.int32)  # src

    if strengths is None:
        data = np.ones(len(rows), dtype=np.float32)
    else:
        w_ii = np.full(len(tgt_ii), float(strengths.get("II", 1.0)), dtype=np.float32)
        w_ei = np.full(len(tgt_ei), float(strengths.get("EI", 1.0)), dtype=np.float32)
        w_ie = np.full(len(tgt_ie), float(strengths.get("IE", 1.0)), dtype=np.float32)
        w_ee = np.full(len(tgt_ee), float(strengths.get("EE", 1.0)), dtype=np.float32)
        data = np.concatenate([w_ii, w_ei, w_ie, w_ee], dtype=np.float32)

    a = csr_matrix((data, (rows, cols)), shape=(N, N))
    a.sum_duplicates()

    counts_new = {"II": len(tgt_ii), "EI": len(tgt_ei), "IE": len(tgt_ie), "EE": len(tgt_ee)}

    if return_sparse:
        return a, counts_emp, counts_new
    return a.toarray(), counts_emp, counts_new


def generateDecayMatrix(size=1000, total_ones=100_000, decay_rate=4.0, power=2.0,
                        shape=None, forbid_diagonal=True, rng=None, return_sparse=True):
    """
    Generate a binary matrix with probability decaying from the top-left.

    Args:
        size: square size if shape is None
        total_ones: number of ones to place
        decay_rate: decay strength
        power: distance exponent
        shape: (n_rows, n_cols) or None
        forbid_diagonal: if True and square, exclude diagonal
        rng: seed or Generator
        return_sparse: CSR if True else dense

    Returns:
        Binary matrix (CSR or dense) stored as M[tgt,src]
    """
    rng = np.random.default_rng(rng)

    if shape is None:
        N = int(size)
        n_rows = n_cols = N
    else:
        n_rows, n_cols = map(int, shape)

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
    tgt = (idx // n_cols).astype(np.int32)  # row
    src = (idx % n_cols).astype(np.int32)   # col

    if return_sparse:
        data = np.ones(k, dtype=np.int8)
        a = coo_matrix((data, (tgt, src)), shape=(n_rows, n_cols)).tocsr()
        a.sum_duplicates()
        if a.nnz:
            a.data[:] = 1
        return a

    a = np.zeros((n_rows, n_cols), dtype=np.int8)
    a[tgt, src] = 1
    return a


def smallWorldDirected(N, n_edges, p_rand=0.05, forbid_self=True, rng=None, return_sparse=True):
    """
    Directed small-world generator.

    Args:
        N: number of nodes
        n_edges: target number of edges
        p_rand: fraction rewired
        forbid_self: drop self-loops if True
        rng: seed or Generator
        return_sparse: CSR if True else dense

    Returns:
        Binary matrix (CSR or dense) stored as M[tgt,src]
    """
    rng = np.random.default_rng(rng)

    k = int(np.ceil(n_edges / N))
    links = []

    for src in range(N):
        for j in range(1, k + 1):
            tgt = (src + j) % N
            if forbid_self and src == tgt:
                continue
            links.append((src, tgt))

    links = np.asarray(links, dtype=np.int32)

    if p_rand == 0:
        if links.shape[0] > n_edges:
            links = links[rng.choice(links.shape[0], size=n_edges, replace=False)]

        src = links[:, 0]
        tgt = links[:, 1]

        if return_sparse:
            a = csr_matrix((np.ones(len(src), dtype=np.int8), (tgt, src)), shape=(N, N))
            a.sum_duplicates()
            if a.nnz:
                a.data[:] = 1
            if forbid_self:
                a.setdiag(0)
                a.eliminate_zeros()
            return a

        mtx = np.zeros((N, N), dtype=np.int8)
        mtx[tgt, src] = 1
        if forbid_self:
            np.fill_diagonal(mtx, 0)
        return mtx

    # rewiring
    links = links[rng.permutation(len(links))]
    n_rewire = int(p_rand * len(links))
    keep_links = links[n_rewire:]

    all_possible = [(s, t) for s in range(N) for t in range(N) if not (forbid_self and s == t)]
    existing = set(map(tuple, keep_links.tolist()))
    candidates = [e for e in all_possible if e not in existing]
    candidates = np.asarray(candidates, dtype=np.int32)

    if candidates.size == 0:
        # no candidates available; fall back to keep_links truncated
        final_links = keep_links[:n_edges]
    else:
        candidates = candidates[rng.permutation(len(candidates))]
        n_rewire = min(n_rewire, len(candidates))
        new_links = candidates[:n_rewire]
        final_links = np.vstack((keep_links, new_links))
        final_links = final_links[rng.permutation(len(final_links))][:n_edges]

    src = final_links[:, 0]
    tgt = final_links[:, 1]

    if return_sparse:
        a = csr_matrix((np.ones(len(src), dtype=np.int8), (tgt, src)), shape=(N, N))
        a.sum_duplicates()
        if a.nnz:
            a.data[:] = 1
        if forbid_self:
            a.setdiag(0)
            a.eliminate_zeros()
        return a

    mtx = np.zeros((N, N), dtype=np.int8)
    mtx[tgt, src] = 1
    if forbid_self:
        np.fill_diagonal(mtx, 0)
    return mtx


def generateNet(
    ce_mtx,
    netname,
    N,
    NI,
    perm=None,
    strengths=None,
    step=5,
    p_rand=0.02,
    rng=None,
    return_sparse=True,
    real_data=None,
):
    """
    Generate a network null model.

    Args:
        ce_mtx: prototype template adjacency (used if real_data is None and ce_mtx is not None)
        netname: one of {"emp","erb","ero","sfo","sfr","swr"}
        N: total number of nodes
        NI: inhibitory count (0..NI-1)
        perm: permutation (used for swr), or None for random
        strengths: optional weights dict (erb)
        step: inhibitory interleaving step (sfr)
        p_rand: rewiring fraction (swr)
        rng: seed or Generator
        return_sparse: CSR if True else dense
        real_data: path to empirical CSR saved via scipy.sparse.save_npz

    Returns:
        Generated matrix (CSR or dense) stored as M[tgt,src]
    """
    rng = np.random.default_rng(rng)

    if N is None or int(N) <= 0:
        raise ValueError(f"N must be a positive integer, got {N}.")
    N = int(N)

    if NI is None:
        raise ValueError("NI is required.")
    NI = int(NI)
    if not (0 < NI < N):
        raise ValueError(f"NI must satisfy 0 < NI < N, got NI={NI}, N={N}.")
    NE = N - NI

    # ---- load / template / prototype (all must end up as NxN) ----
    if real_data is not None:
        a_emp = load_npz(str(real_data)).tocsr(copy=False)
    elif ce_mtx is not None:
        a_emp = ce_mtx.tocsr(copy=False) if issparse(ce_mtx) else csr_matrix(np.asarray(ce_mtx))
    else:
        a_emp = buildPrototypeTemplate(
            N=N,
            NI=NI,
            densities={"II": 0.10, "EI": 0.15, "IE": 0.15, "EE": 0.05},
            rng=rng,
            return_sparse=True,
        )

    if a_emp.shape != (N, N):
        raise ValueError(f"Template shape mismatch: expected {(N, N)}, got {a_emp.shape}.")

    # enforce binary, no self-loops
    a_emp = a_emp.tocsr(copy=False)
    a_emp.sum_duplicates()
    if a_emp.nnz:
        a_emp.data[:] = 1
    a_emp.setdiag(0)
    a_emp.eliminate_zeros()

    n_edges = int(a_emp.nnz)

    if netname == "emp":
        out = a_emp

    elif netname == "erb":
        out, _, _ = randomBlockFromEmpirical(
            a_emp, NI=NI, strengths=strengths, rng=rng, return_sparse=True
        )

    elif netname == "ero":
        nodes = np.arange(N, dtype=np.int32)
        src_sel, tgt_sel = _uniquePairsFromBlock(nodes, nodes, m=n_edges, forbid_self=True, rng=rng)
        out = csr_matrix((np.ones(len(src_sel), dtype=np.int8), (tgt_sel, src_sel)), shape=(N, N))
        out.sum_duplicates()
        if out.nnz:
            out.data[:] = 1

    elif netname == "sfo":
        out = generateDecayMatrix(
            shape=(N, N),
            total_ones=n_edges,
            decay_rate=4.0,
            power=1.0,
            forbid_diagonal=True,
            rng=rng,
            return_sparse=True,
        )

    elif netname == "sfr":
        a = generateDecayMatrix(
            shape=(N, N),
            total_ones=n_edges,
            decay_rate=4.0,
            power=1.0,
            forbid_diagonal=True,
            rng=rng,
            return_sparse=True,
        )
        perm_ie = buildInterleavedIPermutation(NI=NI, NE=NE, N=N, step=step)
        out = relabelNeurons(a, perm=perm_ie, return_sparse=True)

    elif netname == "swr":
        out = smallWorldDirected(
            N=N, n_edges=n_edges, p_rand=p_rand, forbid_self=True, rng=rng, return_sparse=True
        )
        if perm is None:
            perm = rng.permutation(N)
        else:
            perm = np.asarray(perm, dtype=np.int64)
            if perm.ndim != 1 or perm.size != N:
                raise ValueError("perm must have length N")
        out = relabelNeurons(out, perm=perm, return_sparse=True)

    else:
        raise ValueError(f"Invalid netname: {netname}. Choose from 'emp','erb','ero','sfo','sfr','swr'.")

    # return out if return_sparse else out.toarray()
    out = out.tocsr(copy=False)
    out.sum_duplicates()
    if out.nnz:
        out.data[:] = 1
    out.setdiag(0)
    out.eliminate_zeros()

    return out if return_sparse else out.toarray()
    
