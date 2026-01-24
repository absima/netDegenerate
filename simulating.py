import os
import time
import numpy as np
import nest

from network_degeneration import *
from analysis import *
from parameters import *


def simulateAndStore(paramss, Jbg=None, prate=None):
    """
    Run one NEST simulation instance and store structural + activity measures.

    Args:
        paramss: (cp_index, ilb, inet, idtyp, idxprun, istage, iscale)
            cp_index: realization index
            ilb: weight-scheme index
            inet: network index
            idtype: degeneration mode index, as in synapseTrim or neuronTrim
            idxprun: pruning strategy index
            istage: index of degeneration index
            iscale: index of weight scalers
        Jbg: optional background synaptic weight override
        prate: optional poisson generator rate override

    Returns:
        None
    """
    if prate is None:
        prate = p_rate
    if Jbg is None:
        Jbg = J_bg

    cp_index, ilb, inet, idtyp, idxprun, istage, iscale = paramss
    netname = netnames[inet]

    qstrng1 = tuple([qntfold] + list(paramss))
    # qstrng2 = tuple([curfold] + list(paramss))  # requires curfold defined somewhere
    fname1 = "%s/qnt_%d_%d_%d_%d_%d_%d_%d.npz" % qstrng1
    # fname2 = "%s/mnSynI_I_E_%d_%d_%d_%d_%d_%d_%d.npy" % qstrng2
    os.makedirs(f"{qntfold}", exist_ok=True)
    # os.makedirs(f"{curfold}", exist_ok=True)
    
    
    # Weight scheme
    scale = wscales[iscale]
    if np.isnan(scale):
        return

    ww = wk0 * (1 - ilb) + wg0 * ilb
    weight = scale * ww
    blocks = ['II', 'EI', 'IE', 'EE']
    weights = {blocks[i]:weight[i] for i in range(4)}
    
    # Trimming / degeneration
    tparam = (cp_index, inet, idtyp, idxprun, istage)
    bmtx = trimming(tparam)          # expected binary adjacency (sparse), Convention B: M[tgt, src]
    N = bmtx.shape[0]

    new_NI = NI - idtyp * int(del_frac * istage * NI)
    new_NE = N - new_NI

    # Structural features that are weight-scale invariant. 
    if iscale == 0:
        dgg = meanDegree(bmtx.copy(), new_NI, new_NE)
        shd = contributionToPairwiseSharing(bmtx.copy(), new_NI, new_NE)
        dgsh = np.vstack((dgg, shd))
    else:
        dgsh = None

    # allocating weights to adjacency
    wmtx = weightedFromAdjacency(bmtx.copy(), new_NI, weights)

    eigvals = np.linalg.eigvals(wmtx.toarray())
    spectral_radius = np.array(max(abs(eigvals)))

    esw = meanEffectiveLinkWeight(wmtx.copy(), new_NI, new_NE)
    eff, mnsdeff = meanBlockWeights(bmtx, new_NI, weight)

    # -----------------------
    # NEST simulation
    # -----------------------
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(print_time=False)

    nrnall = nest.Create(
        "iaf_psc_alpha",
        N,
        params={
            "C_m": 250.0,
            "tau_m": 10.0,
            "E_L": -70.0,
            "V_reset": -70.0,
            "V_th": -55.0,
            "t_ref": 2.0,
            "tau_syn_ex": 2.0,
            "tau_syn_in": 2.0,
            "I_e": 0.0,
        },
    )

    # Connect recurrent network using sparse edges (NOT all_to_all with matrix)
    # Convention B: wmtx[tgt, src] -> rows=tgt, cols=src
    w_coo = wmtx.tocoo()
    pre_ids = np.asarray(nrnall)[w_coo.col]
    post_ids = np.asarray(nrnall)[w_coo.row]

    nest.Connect(
        pre_ids,
        post_ids,
        conn_spec="one_to_one",
        syn_spec={"weight": w_coo.data, "delay": [delay]*len(pre_ids)},
    )
    # nest.Connect(nrnall, nrnall, 'all_to_all', syn_spec={'weight': wmtx.toarray(), 'delay': delay})
    
    # Background Poisson input
    bg = nest.Create("poisson_generator", params={"rate": prate})
    nest.Connect(bg, nrnall, syn_spec={"weight": Jbg, "delay": delay})

    # Recordings
    mm = nest.Create(
        "multimeter",
        1,
        {
            "start": ms_recstart,
            "stop": ms_simtime,
            "record_from": ["V_m", "I_syn_in"],
        },
    )
    nest.Connect(mm, nrnall)

    spkD = nest.Create("spike_recorder", params={"start": ms_recstart, "stop": ms_simtime})
    nest.Connect(nrnall, spkD)

    # Simulate
    nest.Simulate(ms_simtime)

    # -----------------------
    # Collect outputs
    # -----------------------
    spk_send = nest.GetStatus(spkD)[0]["events"]["senders"]
    spk_time = nest.GetStatus(spkD)[0]["events"]["times"]

    if len(spk_send) == 0:
        rvc = np.nan
        mncf = np.nan
        mnI = np.nan
        asynch = np.nan
        mnIarray = np.nan

        np.savez(
            fname1,
            dgsh=dgsh,
            radius=spectral_radius,
            esw=esw,
            mneff=eff,
            mnsdeff=mnsdeff,
            rvc=rvc,
            sync=asynch,
            ccff=mncf,
            synI=mnI,
        )
        np.save(fname2, mnIarray)
        return

    data = np.column_stack((spk_send, spk_time))
    data[:, 1] = data[:, 1] - ms_recstart
    data[:, 0] = data[:, 0] - 1  # index from 0

    cparams = (cp_index, inet, idtyp, idxprun, istage, iscale)

    rvc = dynPart(data, new_NI, new_NE)
    mncf = meanCorr(data, cparams)

    idx = np.searchsorted(data[:, 1], ms_startI)
    sdata = data[idx:].copy()
    sdata[:, 1] -= ms_startI

    mnI, mnIarray, inhI, excI = meanSynCurrent(cparams, sdata, bmtx.toarray(), weight, new_NI)

    vmvm = nest.GetStatus(mm)[0]["events"]["V_m"]
    asynch = asynchrony(vmvm, new_NI, new_NE)

    np.savez(
        fname1,
        dgsh=dgsh,
        radius=spectral_radius,
        esw=esw,
        mneff=eff,
        mnsdeff=mnsdeff,
        rvc=rvc,
        sync=asynch,
        ccff=mncf,
        synI=mnI,
    )

    # np.save(fname2, mnIarray)  # if desired to be stored. 
    print(paramss)
    return



