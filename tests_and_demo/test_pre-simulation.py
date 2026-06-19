# for pytest

import sys
import importlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

base = Path(__file__).parent
sys.path.insert(0, str(base))

import network_generation as ng
import edge_ordering as eo
import simulation_pipeline as nd

importlib.reload(ng)
importlib.reload(eo)
importlib.reload(nd)


def edgesFromCsr(a: sp.csr_matrix) -> np.ndarray:
    tgt, src = a.nonzero()  # matrix stores (tgt,src)
    return np.vstack([src, tgt]).T.astype(np.int32)


def edgesToCsr(edges: np.ndarray, shape) -> sp.csr_matrix:
    """Build CSR matrix stored as M[tgt,src] from (src,tgt) edges."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return sp.csr_matrix(shape, dtype=np.int8)
    src = edges[:, 0]
    tgt = edges[:, 1]
    data = np.ones(len(edges), dtype=np.int8)
    return sp.csr_matrix((data, (tgt, src)), shape=shape)


def assertEdgesEqualUnordered(e1, e2):
    s1 = set(map(tuple, np.asarray(e1, dtype=np.int64)))
    s2 = set(map(tuple, np.asarray(e2, dtype=np.int64)))
    assert s1 == s2, f"Edge sets differ. Δ={len(s1 ^ s2)}"


def buildTemplate(n: int, density: float = 0.06, rng=None) -> sp.csr_matrix:
    rng = np.random.default_rng(rng)
    target_edges = max(1, int(density * n * (n - 1)))

    rows = np.arange(n, dtype=np.int32)
    cols = np.arange(n, dtype=np.int32)

    # returns (src, tgt)
    r_sel, c_sel = ng._uniquePairsFromBlock(rows, cols, m=target_edges, forbid_self=True, rng=rng)

    data = np.ones_like(r_sel, dtype=np.int8)

    # matrix stores at (tgt, src)
    return sp.csr_matrix((data, (c_sel, r_sel)), shape=(n, n))


def setNdGlobalsForTest(n: int, ni: int):
    """
    simulation_pipeline.py uses module globals N0/NI/NE/Inrn/Enrn.
    Patch them for small, fast tests.
    """
    nd.NI = int(ni)
    nd.NE = int(n - ni)
    nd.N0 = int(n)
    nd.Inrn = np.arange(nd.NI)
    nd.Enrn = np.arange(nd.NI, nd.N0)


def testOrientationRoundtrip():
    n = 50
    ni = 10
    rng = np.random.default_rng(0)

    template = buildTemplate(n, density=0.08, rng=rng)
    a = ng.generateNet(
        ce_mtx=template,
        netname="ero",
        N=n,
        NI=ni,
        rng=rng,
        return_sparse=True,
    )

    edges = edgesFromCsr(a)
    a_rt = edgesToCsr(edges, shape=a.shape)

    assert a.nnz == a_rt.nnz
    assertEdgesEqualUnordered(edges, edgesFromCsr(a_rt))


def testMaxMatchAndSave(tmp_path: Path):
    n, ni = 120, 30
    rng = np.random.default_rng(42)

    template = buildTemplate(n, density=0.06, rng=rng)
    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        N=n,
        NI=ni,
        step=4,
        p_rand=0.05,
        rng=rng,
        return_sparse=True,
    )
    
    
    ordered, layer_sizes = eo.maxMatchDecomposition(a, max_layers=None)
    assert isinstance(ordered, np.ndarray)
    assert isinstance(layer_sizes, np.ndarray)

    ordered = np.asarray(ordered, dtype=np.int64)
    assert ordered.ndim == 2 and ordered.shape[1] == 2

    # No duplicate edges within the ordered list
    ordered_set = set(map(tuple, ordered))
    assert len(ordered_set) == len(ordered)

    # Ordered edges should be drawn from the graph's edge set (subset is allowed with max_layers cap)
    edges = edgesFromCsr(a).astype(np.int64)
    edge_set = set(map(tuple, edges))
    assert ordered_set.issubset(edge_set), "Ordered edges contain pairs not present in the input graph."

    # Save only what Stage II expects
    out_path = tmp_path / "pytest_ordered_edges.npz"
    np.savez_compressed(out_path, ordedges=ordered, layer_sizes=layer_sizes)
    assert out_path.exists()

    re = np.load(out_path)
    ord2 = re["ordedges"]
    sizes2 = re["layer_sizes"]

    assert np.array_equal(ord2, ordered)
    assert np.array_equal(sizes2, layer_sizes)


def testTrimSynapses():
    # Patch globals so trimSynapses uses our intended N
    n, ni = 100, 25
    setNdGlobalsForTest(n, ni)

    rng = np.random.default_rng(7)
    template = buildTemplate(n, density=0.08, rng=rng)

    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        N=n,
        NI=ni,
        step=4,
        p_rand=0.03,
        rng=rng,
        return_sparse=True,
    )

    edges = edgesFromCsr(a).astype(np.int64)

    # Old API: trimSynapses((edges, idxprun, istage))
    istage = 1
    del_frac = getattr(nd, "del_frac", 0.1)
    cut = int(istage * del_frac * len(edges))
    expected_nnz = len(edges) - cut

    for idxprun in range(5):
        a_pruned = nd.trimSynapses((edges, idxprun, istage))
        assert sp.issparse(a_pruned)
        assert a_pruned.nnz == expected_nnz


def testTrimNeuronsAndWeights():
    # Patch globals so trimNeurons uses our intended N, NI, etc.
    n, ni = 90, 20
    setNdGlobalsForTest(n, ni)

    rng = np.random.default_rng(11)
    template = buildTemplate(n, density=0.07, rng=rng)

    a = ng.generateNet(
        ce_mtx=template,
        netname="swr",
        N=n,
        NI=ni,
        step=3,
        p_rand=0.04,
        rng=rng,
        return_sparse=True,
    )

    edges = edgesFromCsr(a).astype(np.int64)

    # trimNeurons((edges, idxprun, istage)) -> matrix
    istage = 1
    del_frac = getattr(nd, "del_frac", 0.1)
    nidel = int(del_frac * ni * istage)
    nedel = int(del_frac * (n - ni) * istage)
    expected_n = n - (nidel + nedel)

    a_kept = nd.trimNeurons((edges, 4, istage))  # idxprun=4 -> dout (old mapping)
    assert sp.issparse(a_kept)
    assert a_kept.shape == (expected_n, expected_n)

    weights = [2.0, 3.0, 5.0, 7.0]

    w = nd.weightedFromAdjacency(a.tocoo(), ni, weights)
    assert sp.issparse(w)
    assert w.nnz == a.nnz


