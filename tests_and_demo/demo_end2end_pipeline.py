# cascade_pipeline_demo.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

import parameters as P
import network_generation as ng
import edge_ordering as eo
import network_degeneration as nd

# CHANGE THIS import to wherever your simulateAndStore() lives:
from simulating import simulateAndStore  # <-- rename module if needed


def _edges_from_csr(a: sp.csr_matrix) -> np.ndarray:
    """Return (src,tgt) edges from CSR stored as M[tgt,src]."""
    tgt, src = a.nonzero()
    if src.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack([src, tgt]).T.astype(np.int64, copy=False)


def run_cascade(data_dir: str | Path, paramss: tuple[int, int, int, int, int, int, int]) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- unpack paramss (as you defined) ----
    cp_index, ilb, inet, idtyp, idxprun, istage, iscale = paramss
    netname = P.netnames[inet]

    # ---- Route all I/O folders to data_dir (so we don’t touch /parent_dir etc.) ----
    # parameters.py
    P.netfold = str(data_dir)
    P.qntfold = str(data_dir)
    if not hasattr(P, "curfold"):
        P.curfold = str(data_dir)
    else:
        P.curfold = str(data_dir)

    # network_degeneration imported names via "from parameters import *", so patch module globals too
    nd.netfold = str(data_dir)
    nd.qntfold = str(data_dir)
    nd.curfold = str(data_dir)  # harmless even if unused

    # ---- (3) generate untrimmed network and save sparse adjacency ----
    rng = np.random.default_rng(cp_index)

    a = ng.generateNet(
        ce_mtx=None,
        netname=netname,
        N=P.N,          # IMPORTANT: explicit N
        NI=P.NI,        # IMPORTANT: explicit NI
        perm=None,
        strengths=None,
        step=4,
        p_rand=0.05,
        rng=rng,
        return_sparse=True,
        real_data=None,
    ).tocsr(copy=False)

    # enforce binary, no self-loops (belt+braces)
    a.sum_duplicates()
    if a.nnz:
        a.data[:] = 1
    a.setdiag(0)
    a.eliminate_zeros()

    adj_path = data_dir / f"{netname}_adj_cp{cp_index}.npz"
    sp.save_npz(adj_path, a)
    print(f"[3] saved adjacency: {adj_path}  (shape={a.shape}, nnz={a.nnz})")

    # ---- (4) order edges and save orderedEdges artifact expected by trimming() ----
    # Exhaustive decomposition: you wanted full partitioning of ALL edges.
    ordered, layer_sizes = eo.maxMatchDecomposition(a, max_layers=None)

    ordered = np.asarray(ordered, dtype=np.int64)
    if ordered.ndim != 2 or ordered.shape[1] != 2:
        raise ValueError(f"maxMatchDecomposition returned unexpected shape: {ordered.shape}")

    ord_path = data_dir / f"{netname}_relabeled_ordered_restored_{cp_index}.npz"
    np.savez_compressed(
        ord_path,
        ordedges=ordered.astype(np.int64, copy=False),
        layer_sizes=np.asarray(layer_sizes, dtype=np.int32),
    )
    print(f"[4] saved ordered edges: {ord_path}  (ordered.shape={ordered.shape}, layers={len(layer_sizes)})")

    # Strong consistency check: ordered edges should equal the original edge set exactly.
    edges_in = _edges_from_csr(a)
    s_in = set(map(tuple, edges_in))
    s_ord = set(map(tuple, ordered))
    if s_in != s_ord:
        diff = len(s_in ^ s_ord)
        raise RuntimeError(f"Ordering mismatch: edge sets differ (Δ={diff}).")

    if len(ordered) != len(edges_in):
        raise RuntimeError(
            f"Ordering length mismatch: ordered={len(ordered)} vs edges_in={len(edges_in)} "
            "(this should match when max_layers=None)."
        )

    # ---- (5) run simulation+analysis+store (expects ordered edge file exists) ----
    print(f"[5] running simulateAndStore(paramss={paramss}) ...")
    simulateAndStore(paramss, Jbg=P.j_bg, prate=P.p_rate)
    print("[5] done.")


if __name__ == "__main__":
    # (cp_index, ilb, inet, idtyp, idxprun, istage, iscale)
    paramss = (0, 0, 0, 1, 2, 5, 0)

    data_dir = Path("data")
    run_cascade(data_dir, paramss)
