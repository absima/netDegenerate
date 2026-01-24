# post simulation step -- creating the multiD array from the output files ... 
# merged_array.shape = (n_real, n_kind, n_net, n_dtyp, n_prun, n_stage, n_scale, n_vars, n_pop) # npop = 3 as in all, I, E
from __future__ import annotations

from pathlib import Path
import numpy as np


def organize_quantities(qnt: dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert one saved qnt_*.npz payload into a (35, 3) matrix (excluding dgsh).

    Expected input shapes:
        radius: ()
        esw: (6, 3)
        mneff: ()
        mnsdeff: (2, 4)
        rvc: (3, 3)
        sync: (3,)
        ccff: (28,)
        synI: (2, 3)
    """
    esw = qnt["esw"]                  # (6,3)
    eff = float(qnt["mneff"])         # scalar
    radi = float(np.reshape(qnt["radius"], (1,))[0])  # scalar

    mnsdeff = np.asarray(qnt["mnsdeff"])
    assert mnsdeff.shape == (2, 4)
    mn_block = mnsdeff[0]             # (4,)
    sd_block = mnsdeff[1]             # (4,)

    syn_i = np.asarray(qnt["synI"])
    assert syn_i.shape == (2, 3)
    syn_i_sum = syn_i.sum(axis=0, keepdims=True)      # (1,3)
    syn_i = np.vstack((syn_i_sum, syn_i))             # (3,3)

    rvc = np.asarray(qnt["rvc"])
    assert rvc.shape == (3, 3)

    sync = np.asarray(qnt["sync"])
    assert sync.shape == (3,)

    ccff = np.asarray(qnt["ccff"])
    assert ccff.shape == (28,)

    ccc = ccff[:16].reshape(4, 4)
    ccs = ccc[:, [0, 2, 3]]           # (4,3)
    cc_ei = ccc[:, 1]                 # (4,)
    ffs = ccff[16:].reshape(4, 3)     # (4,3)

    blocks_mean = np.repeat(mn_block[:, None], 3, axis=1)  # (4,3)
    blocks_std  = np.repeat(sd_block[:, None], 3, axis=1)  # (4,3)
    cc_ei_mat   = np.repeat(cc_ei[:, None], 3, axis=1)     # (4,3)

    combo = np.vstack([
        esw,                                  # 6
        np.repeat(eff, 3).reshape(1, 3),      # 1
        np.repeat(radi, 3).reshape(1, 3),     # 1
        blocks_mean,                          # 4
        blocks_std,                           # 4
        syn_i,                                # 3
        rvc,                                  # 3
        sync.reshape(1, 3),                   # 1
        ffs,                                  # 4
        ccs,                                  # 4
        cc_ei_mat,                            # 4
    ])

    assert combo.shape == (35, 3), f"Expected (35,3), got {combo.shape}"
    return combo



def load_npz_safely(path: Path) -> dict[str, np.ndarray] | None:
    """
    Return dict-like npz content, or None if missing/corrupt.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None



def merge_outputs(
    qfold: str | Path,
    out_path: str | Path = "merged_multidim_output_array.npy",
    *,
    n_real: int = 10,
    n_kind: int = 2,     # landau, brunel
    n_net: int = 6,
    n_dtyp: int = 2,
    n_prun: int = 5,
    n_stage: int = 10,
    n_scale: int = 10,
    n_vars: int = 42,
    n_pop: int = 3,
) -> np.ndarray:
    """
    Build merged array:
        qarr[trial, kind, net, dtyp, prun, stage, scale, var, pop]

    Notes:
      - stage==0 uses (idtyp2=0, idxprun2=0) to reflect "no pruning"
      - dgsh is iscale-invariant, so we reuse from iscale=0
    """
    qfold = Path(qfold)
    out_path = Path(out_path)

    # Final output tensor
    qarr = np.full(
        (n_real, n_kind, n_net, n_dtyp, n_prun, n_stage, n_scale, n_vars, n_pop),
        np.nan,
        dtype=np.float32,
    )

    kind_names = ["landau", "brunel"]

    for itrial in range(n_real):
        print(f"[trial {itrial}]")

        for idkind, _dkind in enumerate(kind_names):
            for inet in range(n_net):
                for idtyp in range(n_dtyp):
                    for idxprun in range(n_prun):
                        for istage in range(n_stage):
                            for iscale in range(n_scale):

                                # Stage 0: no pruning => force canonical (idtyp=0, idxprun=0)
                                if istage == 0:
                                    idtyp2 = 0
                                    idxprun2 = 0
                                else:
                                    idtyp2 = idtyp
                                    idxprun2 = idxprun

                                fname = qfold / (
                                    f"qnt_{itrial}_{idkind}_{inet}_{idtyp2}_{idxprun2}_{istage}_{iscale}.npz"
                                )

                                qnt = load_npz_safely(fname)
                                if qnt is None:
                                    # keep NaN initials, don't crash
                                    continue

                                # because dgsh is invariant with weight scales, ...
                                if iscale == 0:
                                    dgsh = qnt.get("dgsh", None)
                                    if dgsh is None:
                                        continue
                                else:
                                    dgsh = qarr[itrial, idkind, inet, idtyp2, idxprun2, istage, 0, :7, :]
                                    if np.isnan(dgsh).all():
                                        fname0 = qfold / (
                                            f"qnt_{itrial}_{idkind}_{inet}_{idtyp2}_{idxprun2}_{istage}_0.npz"
                                        )
                                        q0 = load_npz_safely(fname0)
                                        if q0 is None or ("dgsh" not in q0):
                                            continue
                                        dgsh = q0["dgsh"]

                                combo = organize_quantities(qnt)  # (35,3) expected
                                comb = np.vstack((dgsh, combo))   # (42,3) expected

                                # Defensive: only write if shape matches the target slot
                                if comb.shape != (n_vars, n_pop):
                                    continue

                                qarr[itrial, idkind, inet, idtyp, idxprun, istage, iscale] = comb

    np.save(out_path, qarr)
    print(f"[saved] {out_path}  shape={qarr.shape} dtype={qarr.dtype}")
    return qarr


if __name__ == "__main__":
    merge_outputs(qfold="data/qntfold", out_path="merged_multidim_output_array.npy")





# import numpy as np
# import scipy.io
# import os
#
# def organizing(qnt):
#     # dgsh = qnt['dgsh']
#     esw  = qnt['esw']
#     eff  = qnt['mneff']
#     radi = np.reshape(qnt['radius'], (1,))
#     mnBlock, sdBlock = qnt['mnsdeff']
#
#     synI = qnt['synI']
#     a = np.sum(synI, 0)
#     synI = np.vstack((a, synI))
#
#     rvc  = qnt['rvc']
#     sync = qnt['sync']
#
#     ccff = qnt['ccff']
#     ccc = ccff[:16].reshape(4,4)
#
#     ccs = ccc[:,[0,2,3]]
#     ffs = ccff[16:].reshape(4,3)
#     cc_ei = ccc[:,1]
#
#     # x = np.vstack((
#     x = [
#         ## dgsh, #7
#         esw, #6
#         np.repeat(eff, 3).reshape(1,3),#1
#         np.repeat(radi, 3).reshape(1,3),#1
#         np.repeat(mnBlock[:,None], 3, 1),#4 mean blocks
#         np.repeat(sdBlock[:,None], 3, 1),#4 sigma blocks
#         synI,#2 inh, exc # 3....
#         rvc,#3 mn, sd, cv
#         sync.reshape(1,3),#1
#         ffs,#4 bins
#         ccs,#4 bins
#         np.repeat(cc_ei[:,None], 3, 1)#4 bins
#         ]
#     return np.vstack(x)
#
#
#
#
#
#
# """ v1 -- all in one go"""
# netnames = [
#     'emp',
#     'erb',
#     'ero',
#     'swr',
#     'sfo',
#     'sfr',
# ]
#
#
# nReal = 10 ###
# nBgd = 3
# nKind = 2
# nNet = 6
# nDtyp = 2
# nPrun = 5
# nStage = 10
# nScale = 10
#
# nVars = 42
# nPop = 3
#
#
# qfold = 'data/qntfold'
#
#
# itr = 0
# qarr = np.full((nReal, nKind, nNet, nDtyp, nPrun, nStage, nScale, nVars, nPop), np.nan)
# for itrial in range(nReal):
#     for idkind, dkind in enumerate(['landau', 'brunel']):
#         for iname in range(nNet):
#             if itr == itrial:
#                 print(itrial)
#                 itr +=1
#             for idtyp in range(nDtyp):
#                 for idxprun in range(nPrun):
#                     for istage in range(nStage):
#                         for iscale in range(nScale):
#
#                             if istage==0:
#                                 idxprun2 = 0
#                                 idtyp2 = 0
#                             else:
#                                 idxprun2 = idxprun
#                                 idtyp2 = idtyp
#
#                             params = [itrial, idkind, iname, idtyp2, idxprun2, istage, iscale]
#                             qstrng = tuple([qfold]+list(params))
#                             fname = '%s/qnt_%d_%d_%d_%d_%d_%d_%d.npz'%qstrng
#
#                             qnt = np.load(fname)
#
#                             if iscale:
#                                 # dgsh is iscale invariant
#                                 dgsh = qarr[itrial, idkind, iname, idtyp2, idxprun2, istage, 0, :7,:]
#                             else:
#                                 dgsh = qnt['dgsh']
#                             combo = organizing(qnt)
#                             comb = np.vstack((dgsh, combo))
#
#                             qarr[itrial, idkind, iname, idtyp, idxprun, istage, iscale] = comb
# np.save(f'merged_multidim_output_array.npy', qarr)