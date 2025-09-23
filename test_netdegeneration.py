
# test_netdegeneration.py
# PyTest suite for:
# - network_generation.py
# - edge_ordering.py
# - trimming.py
#
# Internal convention: A[src, tgt]

import os
from pathlib import Path
import numpy as np
import importlib
import scipy.sparse as sp
import sys

# Ensure the modules under test are importable
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import network_generation as ng
import edge_ordering as eo
import network_degeneration as nd

# Reload to ensure latest code if running repeatedly in notebooks
importlib.reload(ng)
importlib.reload(eo)
importlib.reload(nd)


def edges_from_csr(A: sp.csr_matrix):
    r, c = A.nonzero()
    return np.vstack([r, c]).T.astype(np.int32)


def assert_edges_equal_unordered(e1, e2):
    s1 = set(map(tuple, np.asarray(e1, dtype=np.int64)))
    s2 = set(map(tuple, np.asarray(e2, dtype=np.int64)))
    assert s1 == s2, f"Edge sets differ. Δ={len(s1 ^ s2)}"


def build_template(N: int, density: float = 0.06, rng=None):
    rng = np.random.default_rng(rng)
    target_edges = max(1, int(density * N * (N - 1)))
    rows = np.arange(N, dtype=np.int32)
    cols = np.arange(N, dtype=np.int32)
    r_sel, c_sel = ng._UniquePairsFromBlock(rows, cols, m=target_edges, forbid_self=True, rng=rng)
    template = sp.csr_matrix((np.ones_like(r_sel, dtype=np.int8), (r_sel, c_sel)), shape=(N, N))
    return template


def test_orientation_roundtrip():
    N = 50
    rng = np.random.default_rng(0)
    template = build_template(N, density=0.08, rng=rng)
    A = ng.GenerateNet(cemtx=template, netname='ero', rng=rng, return_sparse=True)

    edges = edges_from_csr(A)
    A_rt = nd._EdgesToCSR(edges, shape=A.shape)
    assert (A != 0).nnz == (A_rt != 0).nnz
    assert_edges_equal_unordered(edges, edges_from_csr(A_rt))


def test_maxmatch_and_save(tmp_path: Path):
    N, NI = 120, 30
    rng = np.random.default_rng(42)
    template = build_template(N, density=0.06, rng=rng)
    A = ng.GenerateNet(cemtx=template, netname='swr', NI=NI, step=4, prand=0.05, rng=rng, return_sparse=True)

    layers = eo.MaxMatchDecomposition(A, cols_are_sources=False, max_layers=5)
    assert isinstance(layers, list)
    total_matched = sum(len(L) for L in layers)
    assert total_matched >= 0

    edges = edges_from_csr(A)
    ordered = np.vstack(layers) if layers else np.empty((0,2), dtype=np.int32)
    perm = eo.BuildPermutationFromOrder(edges, ordered)

    save_path = eo.SaveCompressed(edges, perm, netname="pytest_smoke", itrial=1, sizes=A.shape, outdir=str(tmp_path))
    assert Path(save_path).exists()

    # Verify sorting by perm follows 'ordered' prefix where present
    idx = np.argsort(perm)
    sorted_edges = edges[idx]
    present = set(map(tuple, edges))
    ordered_present = np.array([e for e in ordered if tuple(e) in present], dtype=np.int32)
    k2 = min(len(ordered_present), len(sorted_edges))
    if k2 > 0:
        assert_edges_equal_unordered(sorted_edges[:k2], ordered_present[:k2])


def test_trim_synapses(tmp_path: Path):
    N, NI = 100, 25
    rng = np.random.default_rng(7)
    template = build_template(N, density=0.08, rng=rng)
    A = ng.GenerateNet(cemtx=template, netname='swr', NI=NI, step=4, prand=0.03, rng=rng, return_sparse=True)
    edges = edges_from_csr(A)

    # Save ordering to use 'ord'/'rev'
    layers = eo.MaxMatchDecomposition(A, cols_are_sources=False, max_layers=3)
    ordered = np.vstack(layers) if layers else np.empty((0,2), dtype=np.int32)
    perm = eo.BuildPermutationFromOrder(edges, ordered)
    eo.SaveCompressed(edges, perm, netname="pytest_trim", itrial=2, sizes=A.shape, outdir=str(tmp_path))

    k = min(40, A.nnz)

    A_rand, rem_rand = nd.TrimSynapses(A, k=k, strategy="rand", seed=0)
    A_out,  rem_out  = nd.TrimSynapses(A, k=k, strategy="out")
    A_in,   rem_in   = nd.TrimSynapses(A, k=k, strategy="in")
    A_ord,  rem_ord  = nd.TrimSynapses(A, k=k, strategy="ord",
                                       ordering_dir=str(tmp_path), netname="pytest_trim", trial_index=2)
    A_rev,  rem_rev  = nd.TrimSynapses(A, k=k, strategy="rev",
                                       ordering_dir=str(tmp_path), netname="pytest_trim", trial_index=2)

    for Apruned, removed in [(A_rand, rem_rand), (A_out, rem_out), (A_in, rem_in), (A_ord, rem_ord), (A_rev, rem_rev)]:
        assert isinstance(Apruned, sp.csr_matrix)
        assert isinstance(removed, np.ndarray)
        assert Apruned.nnz == A.nnz - len(removed)


def test_trim_neurons_and_weights():
    N, NI = 90, 20
    rng = np.random.default_rng(11)
    template = build_template(N, density=0.07, rng=rng)
    A = ng.GenerateNet(cemtx=template, netname='swr', NI=NI, step=3, prand=0.04, rng=rng, return_sparse=True)

    kept_idx, A_kept = nd.TrimNeurons(A, NI=NI, n_remove_I=4, n_remove_E=6, strategy="dout", seed=0)
    assert len(kept_idx) == N - 10
    assert A_kept.shape == (N - 10, N - 10)

    W = nd.WeightedFromAdjacency(A, NI=NI, weights=nd.Lweight(2.0, 3.0, 5.0, 7.0), return_sparse=True)
    assert sp.issparse(W) and W.shape == A.shape
    assert (W != 0).nnz == A.nnz
