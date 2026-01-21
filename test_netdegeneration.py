# PyTest suite for:
# - network_generation.py
# - edge_ordering.py
# - trimming.py
#
# local convention: A[src, tgt]

import sys
import importlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Ensure the modules under test are importable
base = Path(__file__).parent
sys.path.insert(0, str(base))

import network_generation as ng
import edge_ordering as eo
import network_degeneration as nd

# Reload to ensure latest code if running repeatedly
importlib.reload(ng)
importlib.reload(eo)
importlib.reload(nd)


def edgesFromCsr(a: sp.csr_matrix) -> np.ndarray:
    rows, cols = a.nonzero()
    return np.vstack([rows, cols]).T.astype(np.int32)


def assertEdgesEqualUnordered(e1, e2):
    s1 = set(map(tuple, np.asarray(e1, dtype=np.int64)))
    s2 = set(map(tuple, np.asarray(e2, dtype=np.int64)))
    assert s1 == s2, f"Edge sets differ. Δ={len(s1 ^ s2)}"


def buildTemplate(n: int, density: float = 0.06, rng=None) -> sp.csr_matrix:
    rng = np.random.default_rng(rng)
    target_edges = max(1, int(density * n * (n - 1)))

    rows = np.arange(n, dtype=np.int32)
    cols = np.arange(n, dtype=np.int32)

    r_sel, c_sel = ng._uniquePairsFromBlock(
        rows, cols, m=target_edges, forbid_self=True, rng=rng
    )

    data = np.ones_like(r_sel, dtype=np.int8)
    return sp.csr_matrix((data, (r_sel, c_sel)), shape=(n, n))


# Tests
def testOrientationRoundtrip():
    n = 50
    rng = np.random.default_rng(0)

    template = buildTemplate(n, density=0.08, rng=rng)
    a = ng.generateNet(ce_mtx=template, netname="ero", rng=rng, return_sparse=True)

    edges = edgesFromCsr(a)
    a_rt = nd._edgesToCsr(edges, shape=a.shape)

    assert a.nnz == a_rt.nnz
    assertEdgesEqualUnordered(edges, edgesFromCsr(a_rt))


def testMaxMatchAndSave(tmp_path: Path):
    n, ni = 120, 30
    rng = np.random.default_rng(42)

    template = buildTemplate(n, density=0.06, rng=rng)
    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        ni=ni,
        step=4,
        p_rand=0.05,
        rng=rng,
        return_sparse=True,
    )

    layers = eo.maxMatchDecomposition(a, cols_are_sources=False, max_layers=5)
    assert isinstance(layers, list)

    edges = edgesFromCsr(a)
    ordered = np.vstack(layers) if layers else np.empty((0, 2), dtype=np.int32)

    perm = eo.buildPermutationFromOrder(edges, ordered)
    save_path = eo.saveCompressed(
        edges,
        perm,
        netname="pytest_smoke",
        itrial=1,
        sizes=a.shape,
        outdir=str(tmp_path),
    )

    assert Path(save_path).exists()

    idx = np.argsort(perm)
    sorted_edges = edges[idx]

    present = set(map(tuple, edges))
    ordered_present = np.array(
        [e for e in ordered if tuple(e) in present], dtype=np.int32
    )

    k2 = min(len(sorted_edges), len(ordered_present))
    if k2 > 0:
        assertEdgesEqualUnordered(sorted_edges[:k2], ordered_present[:k2])


def testTrimSynapses(tmp_path: Path):
    n, ni = 100, 25
    rng = np.random.default_rng(7)

    template = buildTemplate(n, density=0.08, rng=rng)
    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        ni=ni,
        step=4,
        p_rand=0.03,
        rng=rng,
        return_sparse=True,
    )

    edges = edgesFromCsr(a)

    layers = eo.maxMatchDecomposition(a, cols_are_sources=False, max_layers=3)
    ordered = np.vstack(layers) if layers else np.empty((0, 2), dtype=np.int32)
    perm = eo.buildPermutationFromOrder(edges, ordered)

    eo.saveCompressed(
        edges,
        perm,
        netname="pytest_trim",
        itrial=2,
        sizes=a.shape,
        outdir=str(tmp_path),
    )

    k = min(40, a.nnz)

    results = [
        nd.trimSynapses(a, k=k, strategy="rand", seed=0),
        nd.trimSynapses(a, k=k, strategy="out"),
        nd.trimSynapses(a, k=k, strategy="in"),
        nd.trimSynapses(
            a,
            k=k,
            strategy="ord",
            ordering_dir=str(tmp_path),
            netname="pytest_trim",
            trial_index=2,
        ),
        nd.trimSynapses(
            a,
            k=k,
            strategy="rev",
            ordering_dir=str(tmp_path),
            netname="pytest_trim",
            trial_index=2,
        ),
    ]

    for a_pruned, removed in results:
        assert isinstance(a_pruned, sp.csr_matrix)
        assert a_pruned.nnz == a.nnz - len(removed)


def testTrimNeuronsAndWeights():
    n, ni = 90, 20
    rng = np.random.default_rng(11)

    template = buildTemplate(n, density=0.07, rng=rng)
    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        ni=ni,
        step=3,
        p_rand=0.04,
        rng=rng,
        return_sparse=True,
    )

    kept_idx, a_kept = nd.trimNeurons(
        a,
        ni=ni,
        n_remove_i=4,
        n_remove_e=6,
        strategy="dout",
        seed=0,
    )

    assert len(kept_idx) == n - 10
    assert a_kept.shape == (n - 10, n - 10)

    w = nd.weightedFromAdjacency(
        a,
        ni=ni,
        weights=nd.lweight(2.0, 3.0, 5.0, 7.0),
        return_sparse=True,
    )

    assert sp.issparse(w)
    assert w.nnz == a.nnz
