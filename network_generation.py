from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

# during generation,
# - We store edges as: A[src, tgt] = 1
#   rows   = sources
#   cols   = targets
# - Canonical edge tuples are (src, tgt).
# - When converting to/from edges:
#     * from matrix -> edges: (src, tgt) = M.nonzero() as (row, col)
#     * from edges  -> matrix: place ones at (row=src, col=tgt)


def _uniquePairsFromBlock(
    rows: np.ndarray,
    cols: np.ndarray,
    m: int,
    forbid_self: bool = False,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample exactly m unique (src, tgt) pairs with src in `rows` and tgt in `cols`,
    uniformly without replacement.

    INTERNAL ORIENTATION: rows are sources, cols are targets.
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

    # same set + forbid_self: exclude diagonal via batched rejection
    capacity = n_rows * n_cols - n_rows
    if m > capacity:
        raise ValueError(f"Requested {m} pairs but capacity without diagonal is {capacity}.")

    selected: set[tuple[int, int]] = set()
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


def buildInterleavedIPermutation(ni: int, ne: int, n: int | None = None, step: int = 5) -> np.ndarray:
    """
    Construct a node permutation that places inhibitory (old indices 0..ni-1) at new
    positions 0, step, 2*step, ... as many as fit, and fills the rest with excitatory
    (old indices ni..ni+ne-1) in ascending order.

    Returns `perm` such that: new_index_of(old_i) = perm[old_i].

    This permutation can be applied symmetrically to rows and columns of adjacency
    since sources and targets refer to the same neuron set in square matrices.
    """
    if n is None:
        n = ni + ne
    if n != ni + ne:
        raise ValueError("If n is provided, it must equal ni + ne.")

    i_positions = np.arange(0, step * ni, step, dtype=np.int32)
    i_positions = i_positions[i_positions < n]

    new_to_old = np.empty(n, dtype=np.int32)
    used = np.zeros(n, dtype=bool)

    i_old = np.arange(ni, dtype=np.int32)
    n_i_place = min(len(i_positions), ni)

    new_to_old[i_positions[:n_i_place]] = i_old[:n_i_place]
    used[i_positions[:n_i_place]] = True

    i_remaining = i_old[n_i_place:]
    e_old = np.arange(ni, ni + ne, dtype=np.int32)
    fill_pool = np.concatenate([i_remaining, e_old], dtype=np.int32)

    free_slots = np.flatnonzero(~used)
    if free_slots.size != fill_pool.size:
        raise RuntimeError("Permutation construction mismatch.")

    new_to_old[free_slots] = fill_pool

    perm = np.empty(n, dtype=np.int32)
    perm[new_to_old] = np.arange(n, dtype=np.int32)
    return perm


def relabelNeurons(a, perm: np.ndarray | None = None, return_sparse: bool = True):
    """
    Relabel (permute) neuron indices of a square sparse adjacency: A' = A[perm][:, perm].

    INTERNAL ORIENTATION: A[src, tgt].
    """
    if not issparse(a):
        raise TypeError("a must be a scipy.sparse matrix.")
    n, m = a.shape
    if n != m:
        raise ValueError("a must be square (n x n).")

    if perm is None:
        rng = np.random.default_rng()
        perm = rng.permutation(n).astype(np.int32)
    else:
        perm = np.asarray(perm, dtype=np.int64)
        if perm.ndim != 1 or len(perm) != n:
            raise ValueError("perm must be a 1D array of length n.")
        if np.unique(perm).size != n or perm.min() != 0 or perm.max() != n - 1:
            raise ValueError("perm must be a permutation of 0..n-1.")
        perm = perm.astype(np.int32, copy=False)

    a_perm = a[perm][:, perm]
    if return_sparse:
        return a_perm.tocsr()
    return a_perm.toarray()


# ==========================
# Generators (Null Models)
# ==========================

def randomizeBinaryMatrix(ob_mtx, return_sparse: bool = True, rng=None):
    """
    Randomize a binary block by shuffling the positions of ones while preserving the exact count.
    For square blocks, self-loops (diagonal) are forbidden.

    INTERNAL ORIENTATION: returns a matrix with ones at (src, tgt).
    """
    b = np.asarray(ob_mtx, dtype=np.int8)
    n_rows, n_cols = b.shape
    num_edges = int(b.sum())

    if num_edges == 0:
        return csr_matrix((n_rows, n_cols), dtype=np.int8) if return_sparse else np.zeros_like(b, dtype=np.int8)

    rows = np.arange(n_rows, dtype=np.int32)  # sources
    cols = np.arange(n_cols, dtype=np.int32)  # targets
    forbid_self = (n_rows == n_cols)

    r_sel, c_sel = _uniquePairsFromBlock(rows, cols, num_edges, forbid_self=forbid_self, rng=rng)

    if return_sparse:
        data = np.ones(len(r_sel), dtype=np.int8)
        a = csr_matrix((data, (r_sel, c_sel)), shape=(n_rows, n_cols))
        a.sum_duplicates()
        return a

    nb_mtx = np.zeros((n_rows, n_cols), dtype=np.int8)
    nb_mtx[r_sel, c_sel] = 1
    if forbid_self:
        np.fill_diagonal(nb_mtx, 0)
    return nb_mtx


def randomBlockFromEmpirical(b_mtx_emp, ni: int, strengths: dict | None = None, rng=None, return_sparse: bool = True):
    """
    Preserve block-wise edge counts of an empirical I/E microcircuit (I→I, I→E, E→I, E→E),
    randomize connections uniformly within each block, and optionally assign per-block weights.

    INTERNAL ORIENTATION:
      - Input is interpreted as A[src, tgt].
      - Block (I→E) means sources in I, targets in E, etc.
    """
    rng = np.random.default_rng(rng)

    if issparse(b_mtx_emp):
        a_emp = b_mtx_emp.tocsr(copy=False)
        n = a_emp.shape[0]
        a_emp = a_emp - csr_matrix((a_emp.diagonal(), (np.arange(n), np.arange(n))), shape=a_emp.shape)
        a_emp.eliminate_zeros()
        a_emp.data[:] = 1
        is_sparse = True
    else:
        a_emp = (np.asarray(b_mtx_emp) != 0).astype(np.int8)
        n = a_emp.shape[0]
        np.fill_diagonal(a_emp, 0)
        is_sparse = False

    i_idx = np.arange(ni, dtype=np.int32)
    e_idx = np.arange(ni, n, dtype=np.int32)

    if is_sparse:
        def _count(a_mat, r, c) -> int:
            return int(a_mat[r][:, c].nnz)

        c_ii = _count(a_emp, i_idx, i_idx)
        c_ie = _count(a_emp, i_idx, e_idx)
        c_ei = _count(a_emp, e_idx, i_idx)
        c_ee = _count(a_emp, e_idx, e_idx)
    else:
        c_ii = int(a_emp[np.ix_(i_idx, i_idx)].sum())
        c_ie = int(a_emp[np.ix_(i_idx, e_idx)].sum())
        c_ei = int(a_emp[np.ix_(e_idx, i_idx)].sum())
        c_ee = int(a_emp[np.ix_(e_idx, e_idx)].sum())

    counts_emp = {"II": c_ii, "IE": c_ie, "EI": c_ei, "EE": c_ee}

    r_ii, c_ii_r = _uniquePairsFromBlock(i_idx, i_idx, c_ii, forbid_self=True, rng=rng)
    r_ie, c_ie_r = _uniquePairsFromBlock(i_idx, e_idx, c_ie, forbid_self=False, rng=rng)
    r_ei, c_ei_r = _uniquePairsFromBlock(e_idx, i_idx, c_ei, forbid_self=False, rng=rng)
    r_ee, c_ee_r = _uniquePairsFromBlock(e_idx, e_idx, c_ee, forbid_self=True, rng=rng)

    rows = np.concatenate([r_ii, r_ie, r_ei, r_ee], dtype=np.int32)
    cols = np.concatenate([c_ii_r, c_ie_r, c_ei_r, c_ee_r], dtype=np.int32)

    if strengths is None:
        data = np.ones(len(rows), dtype=np.float32)
    else:
        w_ii = np.full(len(r_ii), float(strengths.get("II", 1.0)), dtype=np.float32)
        w_ie = np.full(len(r_ie), float(strengths.get("IE", 1.0)), dtype=np.float32)
        w_ei = np.full(len(r_ei), float(strengths.get("EI", 1.0)), dtype=np.float32)
        w_ee = np.full(len(r_ee), float(strengths.get("EE", 1.0)), dtype=np.float32)
        data = np.concatenate([w_ii, w_ie, w_ei, w_ee], dtype=np.float32)

    a = csr_matrix((data, (rows, cols)), shape=(n, n))
    a.sum_duplicates()

    counts_new = {"II": len(r_ii), "IE": len(r_ie), "EI": len(r_ei), "EE": len(r_ee)}

    if return_sparse:
        return a, counts_emp, counts_new
    return a.toarray(), counts_emp, counts_new


def generateDecayMatrix(
    size: int = 1000,
    total_ones: int = 100_000,
    decay_rate: float = 4.0,
    power: float = 2.0,
    shape: tuple[int, int] | None = None,
    forbid_diagonal: bool = True,
    rng=None,
    return_sparse: bool = True,
):
    """
    Generate a binary adjacency with density decaying from top-left:
    prob ~ exp(-decay_rate * distance**power). Useful as an unrelabeled null.

    one way of generating network with hetrogenous deg_dist
    """
    rng = np.random.default_rng(rng)

    if shape is None:
        n = int(size)
        n_rows = n_cols = n
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
    r = (idx // n_cols).astype(np.int32)
    c = (idx % n_cols).astype(np.int32)

    if return_sparse:
        data = np.ones(k, dtype=np.int8)
        a = coo_matrix((data, (r, c)), shape=(n_rows, n_cols)).tocsr()
        a.sum_duplicates()
        return a

    a = np.zeros((n_rows, n_cols), dtype=np.int8)
    a[r, c] = 1
    return a


def smallWorldDirected(
    n: int,
    n_edges: int,
    p_rand: float = 0.02,
    forbid_self: bool = True,
    rng=None,
    return_sparse: bool = True,
):
    """
    Directed small-world–like digraph via ring-lattice + rewiring.
    Returns exactly n_edges edges.

    INTERNAL ORIENTATION: returns A[src, tgt].
    """
    rng = np.random.default_rng(rng)

    # --- determine baseline out-degree per node ---
    k_max = max(0, n - 1) if forbid_self else n
    k = int(np.ceil(n_edges / n))
    k = int(min(max(0, k), k_max))

    # --- build initial ring-lattice edges ---
    if k == 0:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
    else:
        rows = np.repeat(np.arange(n, dtype=np.int32), k)  # sources
        t = np.arange(1, k + 1, dtype=np.int32)
        cols = (np.arange(n, dtype=np.int32)[:, None] + t) % n  # forward neighbors
        cols = cols.reshape(-1).astype(np.int32)

        # remove self-loops if disallowed
        if forbid_self:
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]

    # --- deduplicate (defensive; ring lattice should already be unique) ---
    if rows.size:
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        dedup = np.ones(rows.size, dtype=bool)
        dedup[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        rows, cols = rows[dedup], cols[dedup]

    # --- adjust edge count to exactly n_edges ---
    e0 = rows.size
    if e0 > n_edges:
        # too many edges → randomly downsample
        take = rng.choice(e0, size=n_edges, replace=False)
        rows, cols = rows[take], cols[take]

    elif e0 < n_edges:
        # too few edges → randomly add non-edges
        need = n_edges - e0
        edge_set = set(zip(rows.tolist(), cols.tolist()))
        batch = max(need, int(1.25 * need))
        add_r, add_c = [], []

        while need > 0:
            r = rng.integers(0, n, size=batch, dtype=np.int64)
            c = rng.integers(0, n, size=batch, dtype=np.int64)

            if forbid_self:
                keep = r != c
                r, c = r[keep], c[keep]

            for ri, ci in zip(r, c):
                key = (int(ri), int(ci))
                if key not in edge_set:
                    edge_set.add(key)
                    add_r.append(key[0])
                    add_c.append(key[1])
                    need -= 1
                    if need == 0:
                        break

            batch = max(32, int(1.5 * need))

        if add_r:
            rows = np.concatenate([rows, np.array(add_r, dtype=np.int32)])
            cols = np.concatenate([cols, np.array(add_c, dtype=np.int32)])

    # rewire a fraction of edges (ones) to random non-edges (zeros)
    m = rows.size
    if p_rand > 0 and m > 0:
        n_rewire = int(np.floor(p_rand * m))
        if n_rewire > 0:
            idx_remove = rng.choice(m, size=n_rewire, replace=False)
            keep_mask = np.ones(m, dtype=bool)
            keep_mask[idx_remove] = False

            rows_keep, cols_keep = rows[keep_mask], cols[keep_mask]
            edge_set = set(zip(rows_keep.tolist(), cols_keep.tolist()))

            new_r, new_c = [], []
            need = n_rewire
            batch = max(need, int(1.25 * need))

            while need > 0:
                r = rng.integers(0, n, size=batch, dtype=np.int64)
                c = rng.integers(0, n, size=batch, dtype=np.int64)

                if forbid_self:
                    keep = r != c
                    r, c = r[keep], c[keep]

                for ri, ci in zip(r, c):
                    key = (int(ri), int(ci))
                    if key not in edge_set:
                        edge_set.add(key)
                        new_r.append(key[0])
                        new_c.append(key[1])
                        need -= 1
                        if need == 0:
                            break

                batch = max(32, int(1.5 * need))

            rows = np.concatenate([rows_keep, np.array(new_r, dtype=np.int32)])
            cols = np.concatenate([cols_keep, np.array(new_c, dtype=np.int32)])

    data = np.ones(rows.size, dtype=np.int8)
    a = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.data[:] = 1  
    return a if return_sparse else a.toarray()



# =========
# Wrapper
# =========

def generateNet(
    ce_mtx,
    netname: str,
    ni: int | None = None,
    strengths: dict | None = None,
    step: int = 5,
    p_rand: float = 0.02,
    rng=None,
    return_sparse: bool = True,
):
    """
    Unified generator wrapper.

    INTERNAL ORIENTATION for all outputs: A[src, tgt] = 1.

    netname:
      'emp' : return input as CSR (or dense if return_sparse=False)
      'erb' : randomBlockFromEmpirical (needs ni)
      'ero' : global ER (same nnz as ce_mtx; no self-loops)
      'sfo' : top-left–biased decay (same nnz)
      'sfr' : decay then relabel I interleaved (needs ni)
      'swr' : small-world then relabel I interleaved (needs ni)
    """
    rng = np.random.default_rng(rng)

    a_emp = ce_mtx.tocsr(copy=False) if issparse(ce_mtx) else csr_matrix(np.asarray(ce_mtx))
    n = a_emp.shape[0]
    n_edges = int(a_emp.nnz)

    if netname == "emp":
        return a_emp if return_sparse else a_emp.toarray()

    if netname == "erb":
        if ni is None:
            raise ValueError("ni is required for 'erb'.")
        a_block, _, _ = randomBlockFromEmpirical(a_emp, ni=ni, strengths=strengths, rng=rng, return_sparse=True)
        return a_block if return_sparse else a_block.toarray()

    if netname == "ero":
        rows = np.arange(n, dtype=np.int32)
        cols = np.arange(n, dtype=np.int32)
        r_sel, c_sel = _uniquePairsFromBlock(rows, cols, m=n_edges, forbid_self=True, rng=rng)
        data = np.ones(len(r_sel), dtype=np.int8)
        a = csr_matrix((data, (r_sel, c_sel)), shape=(n, n))
        a.sum_duplicates()
        return a if return_sparse else a.toarray()

    if netname == "sfo":
        return generateDecayMatrix(
            shape=(n, n),
            total_ones=n_edges,
            decay_rate=4.0,
            power=1.0,
            forbid_diagonal=True,
            rng=rng,
            return_sparse=return_sparse,
        )

    if netname == "sfr":
        if ni is None:
            raise ValueError("ni is required for 'sfr'.")
        ne = n - ni
        a = generateDecayMatrix(
            shape=(n, n),
            total_ones=n_edges,
            decay_rate=4.0,
            power=1.0,
            forbid_diagonal=True,
            rng=rng,
            return_sparse=True,
        )
        perm = buildInterleavedIPermutation(ni=ni, ne=ne, n=n, step=step)
        return relabelNeurons(a, perm=perm, return_sparse=return_sparse)

    if netname == "swr":
        if ni is None:
            raise ValueError("ni is required for 'swr'.")
        ne = n - ni  # kept for symmetry/documentation
        a = smallWorldDirected(n=n, n_edges=n_edges, p_rand=p_rand, forbid_self=True, rng=rng, return_sparse=True)
        perm = buildInterleavedIPermutation(ni=ni, ne=ne, n=n, step=step)
        return relabelNeurons(a, perm=perm, return_sparse=return_sparse)

    raise ValueError("Invalid netname: {0}. Choose from 'emp','erb','ero','sfo','sfr','swr'.".format(netname))
