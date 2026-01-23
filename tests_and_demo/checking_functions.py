# viz_demo.py
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path

from network_generation import smallWorldDirected, generateNet
from edge_ordering import maxMatchDecomposition
import network_degeneration as nd


# Matrix convention everywhere in this file:
# M[tgt, src] = 1 means src -> tgt (columns are sources)


def _edgesFromCsr(a: sp.csr_matrix) -> np.ndarray:
    tgt, src = a.nonzero()
    return np.vstack([src, tgt]).T.astype(np.int64)


def _edgeSet(a: sp.csr_matrix) -> set[tuple[int, int]]:
    tgt, src = a.nonzero()
    return set(zip(src.tolist(), tgt.tolist()))


def _setNdGlobalsForDemo(N: int, NI: int) -> None:
    """
    network_degeneration (old trimming code) uses module globals:
      N, NI, NE, inrn, enrn, del_frac

    Patch them so our small demos behave correctly.
    """
    nd.N = int(N)
    nd.NI = int(NI)
    nd.NE = int(N - NI)
    nd.inrn = np.arange(nd.NI)
    nd.enrn = np.arange(nd.NI, nd.N)


def demoSmallWorldDirected():
    N = 50
    n_edges = 200
    rng_seed = 123

    a = smallWorldDirected(
        N=N,
        n_edges=n_edges,
        p_rand=0.1,
        forbid_self=True,
        rng=rng_seed,
        return_sparse=True,
    ).tocsr(copy=False)

    print("shape:", a.shape)
    print("nnz:", a.nnz)

    assert a.nnz == n_edges, f"Expected nnz={n_edges}, got {a.nnz}"
    assert np.all(a.data == 1), "Adjacency is not binary!"
    assert a.diagonal().sum() == 0, "Found self-loops despite forbid_self=True"

    tgt, src = a.nonzero()  # rows=tgt, cols=src
    assert src.shape == tgt.shape
    print("First 10 edges (src->tgt):")
    print(np.vstack([src[:10], tgt[:10]]).T)

    return a


def demoRewiringEffect():
    N = 80
    n_edges = 320
    seed = 0

    a0 = smallWorldDirected(
        N=N, n_edges=n_edges, p_rand=0.0, forbid_self=True, rng=seed, return_sparse=True
    ).tocsr(copy=False)
    a1 = smallWorldDirected(
        N=N, n_edges=n_edges, p_rand=0.2, forbid_self=True, rng=seed, return_sparse=True
    ).tocsr(copy=False)

    s0 = _edgeSet(a0)
    s1 = _edgeSet(a1)

    changed = len(s0 ^ s1)
    overlap = len(s0 & s1)

    print("nnz a0:", a0.nnz, "nnz a1:", a1.nnz)
    print("overlap edges:", overlap)
    print("changed edges (sym diff):", changed)

    assert a0.nnz == n_edges and a1.nnz == n_edges
    assert np.all(a0.data == 1) and np.all(a1.data == 1)

    return a0, a1


def demoGenerateNet(return_networks: bool = True):
    """
    Uses generateNet() prototype behavior (real_data=None).
    """
    N = 100
    NI = 15
    rng = np.random.default_rng(1)

    networks = {}
    for netname in ["emp", "ero", "erb", "sfo", "swr", "sfr"]:
        a = generateNet(
            ce_mtx=None,
            netname=netname,
            N=N,
            NI=NI,
            perm=None,
            strengths=None,
            step=4,
            p_rand=0.1,
            rng=rng,
            return_sparse=True,
            real_data=None,  # force prototype path
        ).tocsr(copy=False)

        a.sum_duplicates()
        if a.nnz:
            a.data[:] = 1

        print(netname, "nnz:", a.nnz, "diag_sum:", a.diagonal().sum())
        assert a.shape[0] == a.shape[1]
        networks[netname] = a

    if return_networks:
        return networks
    return None


def plotAdjacencyHeatmaps(adj_dict, n_rows=2, n_cols=3, max_nodes=None, figsize=(12, 7), binary=True):
    names = list(adj_dict.keys())
    mats = list(adj_dict.values())

    plt.close("all")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(mats) :]:
        ax.axis("off")

    for i, (name, adj) in enumerate(zip(names, mats)):
        if i >= n_rows * n_cols:
            break

        ax = axes[i]
        mat = adj.tocsr(copy=False) if sp.issparse(adj) else np.asarray(adj)
        N = mat.shape[0]

        if max_nodes is not None and N > max_nodes:
            idx = np.linspace(0, N - 1, max_nodes, dtype=int)
            view = mat[idx][:, idx].toarray() if sp.issparse(mat) else mat[np.ix_(idx, idx)]
        else:
            view = mat.toarray() if sp.issparse(mat) else mat

        if binary:
            ax.imshow(view, interpolation="nearest", aspect="auto", vmin=0, vmax=1)
        else:
            ax.imshow(view, interpolation="nearest", aspect="auto")

        ax.set_title(name)
        ax.set_xlabel("src (column)")
        ax.set_ylabel("tgt (row)")

    plt.tight_layout()
    return fig, axes


def _isMatching(edges: np.ndarray) -> bool:
    if edges.size == 0:
        return True
    src = edges[:, 0]
    tgt = edges[:, 1]
    return (len(np.unique(src)) == len(src)) and (len(np.unique(tgt)) == len(tgt))


# def demoMaxMatchDecomposition(a: sp.csr_matrix):
#     # For a quick demo you can cap layers; for correctness checks use max_layers=None
#     edges_concat, layer_sizes = maxMatchDecomposition(a, max_layers=5)
#
#     start = 0
#     for i, k in enumerate(layer_sizes):
#         layer = edges_concat[start : start + k]
#         start += k
#         print(f"Layer {i}: {len(layer)} edges, matching={_isMatching(layer)}")
#         assert _isMatching(layer)
#
#     return edges_concat, layer_sizes

def demoMaxMatchDecomposition(
    a: sp.csr_matrix,
    *,
    exhaustive: bool = False,
    max_layers_demo: int = 5,
):
    """
    Demonstrate maxMatchDecomposition.

    Modes
    -----
    exhaustive=False (default):
        Use a small number of layers for readability (demo mode).

    exhaustive=True:
        Fully decompose the graph until all edges are exhausted
        (correctness / pipeline mode).
    """
    max_layers = None if exhaustive else max_layers_demo

    edges_concat, layer_sizes = maxMatchDecomposition(
        a,
        max_layers=max_layers,
    )

    start = 0
    for i, k in enumerate(layer_sizes):
        layer = edges_concat[start : start + k]
        start += k
        print(
            f"Layer {i}: {len(layer)} edges, matching={_isMatching(layer)}"
        )
        assert _isMatching(layer)

    # Strong correctness check in exhaustive mode
    if exhaustive:
        tgt, src = a.nonzero()
        edges_in = set(zip(src.tolist(), tgt.tolist()))
        edges_out = set(map(tuple, edges_concat))

        assert edges_in == edges_out, (
            f"Edge mismatch after exhaustive decomposition: "
            f"Δ={len(edges_in ^ edges_out)}"
        )

    return edges_concat, layer_sizes



def demoTrimSynapses():
    """
    Demonstrate old trimming API:
      nd.trimSynapses((edges, idxprun, istage)) -> sparse adjacency (COO/CSR)
    """
    N = 70
    NI = 14
    _setNdGlobalsForDemo(N, NI)

    a = sp.random(N, N, density=0.05, format="csr", random_state=0)
    a.setdiag(0)
    a.eliminate_zeros()
    if a.nnz:
        a.data[:] = 1

    edges = _edgesFromCsr(a)

    idxprun = 2  # "rnd"
    istage = 1   # removes int(del_frac * len(edges))

    a2 = nd.trimSynapses((edges, idxprun, istage)).tocsr(copy=False)

    print("before nnz:", a.nnz, "after nnz:", a2.nnz)
    cut = int(nd.del_frac * istage * len(edges))
    assert a2.nnz == len(edges) - cut

    return a, a2


def demoTrimNeurons():
    """
    Demonstrate old neuron trimming API:
      nd.trimNeurons((edges, idxprun, istage)) -> sparse adjacency (COO/CSR)
    """
    N = 90
    NI = 20
    _setNdGlobalsForDemo(N, NI)

    rng = np.random.default_rng(11)
    template = sp.random(N, N, density=0.07, format="csr", random_state=11)
    template.setdiag(0)
    template.eliminate_zeros()
    if template.nnz:
        template.data[:] = 1

    a = generateNet(
        ce_mtx=template,
        netname="swr",
        N=N,
        NI=NI,
        perm=None,
        strengths=None,
        step=3,
        p_rand=0.04,
        rng=rng,
        return_sparse=True,
        real_data=None,
    ).tocsr(copy=False)

    edges = _edgesFromCsr(a)

    idxprun = 4  # "dout" per your old trimNeurons mapping
    istage = 1

    a2 = nd.trimNeurons((edges, idxprun, istage)).tocsr(copy=False)

    nidel = int(nd.del_frac * NI * istage)
    nedel = int(nd.del_frac * (N - NI) * istage)
    expected_n = N - (nidel + nedel)

    print("before shape:", a.shape, "after shape:", a2.shape)
    assert a2.shape == (expected_n, expected_n)

    return a, a2


def demoTrimmingLoadOnly():
    """
    Demo the load-only trimming(params) wrapper:
      params = (cp_index, inet, idtyp, idxprun, istage)

    This requires Stage-I artifact:
      {nd.netfold}/{netname}_relabeled_ordered_restored_{cp_index}.npz
    """
    cp_index = 0
    inet = 3     # "swr" in your netnames
    idtyp = 0    # 0=synapse trimming, 1=neuron trimming
    idxprun = 2  # "rnd"
    istage = 1

    netname = nd.netnames[inet]
    ename = Path(nd.netfold) / f"{netname}_relabeled_ordered_restored_{cp_index}.npz"

    if not ename.exists():
        print(f"[SKIP] Missing ordered-edge file: {ename}")
        print("       Run Stage I (generation + ordering + save) first.")
        return None

    a = nd.trimming((cp_index, inet, idtyp, idxprun, istage)).tocsr(copy=False)
    print("Loaded+trimmed adjacency:", a.shape, "nnz:", a.nnz)
    assert a.shape[0] == a.shape[1]
    return a


if __name__ == "__main__":
    print("\n1\n--------\n")
    a_sw = demoSmallWorldDirected()

    print("\n2\n--------\n")
    a0, a1 = demoRewiringEffect()

    print("\n3\n--------\n")
    nets = demoGenerateNet(return_networks=True)

    print("\nPlotting generateNet outputs (2x3)...\n")
    plotAdjacencyHeatmaps(nets, n_rows=2, n_cols=3, max_nodes=120, binary=True)
    plt.show()

    print("\n4\n--------\n")
    a_example = sp.random(80, 80, density=0.03, format="csr", random_state=0)
    a_example.setdiag(0)
    a_example.eliminate_zeros()
    if a_example.nnz:
        a_example.data[:] = 1
    demoMaxMatchDecomposition(a_example)

    print("\n5\n--------\n")
    demoTrimSynapses()

    print("\n6\n--------\n")
    demoTrimNeurons()

    print("\n7\n--------\n")
    demoTrimmingLoadOnly()
