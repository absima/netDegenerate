"""
Degenerate, simulate, analyze, and store one network instance.

This module is the current no-NEST simulation pipeline. It owns the
simulation parameter grid, degeneration helpers, structural/dynamic summary
metrics, the current-based alpha-synapse simulator, and qnt_*.npz storage.

Parameter convention:
    paramss = (ilb, cp_index, inet, idtyp, idxprun, istage, iscale)

Connectivity convention:
    Edge list: (src, tgt)
    Matrix: W[tgt, src] means src -> tgt
"""

import itertools
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix


# ==========================================================
# Folders
# ==========================================================
parfold = "/home/sima/projects/netset2"
netfold = parfold
qntfold = "qntfold"
curfold = "curfold"


# ==========================================================
# Network and experiment parameters
# ==========================================================
NI, NE = [680, 2931]
N0 = NI + NE
Inrn = np.arange(NI)
Enrn = np.arange(NI, N0)

netnames = ["emp", "erb", "ero", "swr", "sfo", "sfr"]
USE_DEMO_CONNECTIVITY = False

nDegen = 2
nIndex = 10
nPrun = 5
nStage = 10
del_frac = 0.1


# ==========================================================
# Neuron and synapse parameters
# ==========================================================
T_REF = 2.0
TAU_M = 10.0
C_M = 250.0
E_L = -70.0
V_RESET = -70.0
V_TH = -55.0

J_unit = 0.013
Je = 0.1
J_exc = Je / J_unit
J_bg = J_exc
p_rate = 8000.0

tau_syn = 2.0
delay = 1.5
g_default = 5.0

# ilb = 0 -> wg, ilb = 1 -> wk
wk = np.array([-3.0, -3.0, 1.5, 1.0])
wg5 = np.array([-5.0, -5.0, 1.0, 1.0])
wg10 = np.array([-10.0, -10.0, 1.0, 1.0])

# scalez = np.logspace(np.log10(0.1), np.log10(15.0), 10)
# 

ilb = 1

if ilb == 0:# standard weighting scheme
    scalez = np.array(
        [[ 2.36104176,  2.10117813,  8.36205398, 15.,          4.07917836,  8.22985433],
         [ 2.03397758,  1.85208968,  7.36691033, 15.,          3.29720195,  6.69365201],
         [ 1.74322534,  1.61947669,  6.51595704, 13.85699257,  2.70402919,  5.71200085],
         [ 1.48003433,  1.41982171,  5.80477279,  9.40490957,  2.20268297,  5.01522424],
         [ 1.26029526,  1.22338808,  5.06526478,  7.13113479,  1.74848692,  4.29025274],
         [ 1.08799983,  1.04531311,  4.66471615,  6.10926899,  1.35900594,  3.73250451],
         [ 0.88563154,  0.85364357,  4.15977286,  4.95400435,  1.06504039,  3.32877267],
         [ 0.71024377,  0.69222111,  3.79694545,  4.41396789,  0.76799608,  2.92281343],
         [ 0.52363655,  0.53185231,  3.40850918,  3.86667291,  0.54615268,  2.48892374],
         [ 0.32774549,  0.32708174,  3.05701863,  3.41669593,  0.30455865,  2.15034207]]
    )
if ilb ==1: # strong inhibition scheme
    # scalez = np.array(
    #     [[0.105, 0.1  , 0.825, 1.618, 0.1  , 0.554]]
    # )
    scalez = np.array([[0.63281464, 0.5698673 , 1.92829821, 1.98056818, 1.38507591, 2.28388294],
       [0.51784524, 0.47380991, 1.383332  , 1.49430901, 0.96598979, 1.69858603],
       [0.41453699, 0.38935399, 1.12917462, 1.1317218 , 0.69012464, 1.21465548],
       [0.33603787, 0.31972571, 0.87096327, 0.88608737, 0.48513365, 0.92575807],
       [0.27360966, 0.26099724, 0.668916  , 0.67649131, 0.35033414, 0.68145037],
       [0.20852191, 0.20666759, 0.50507298, 0.50913976, 0.25003219, 0.49809397],
       [0.16127609, 0.15757984, 0.37492536, 0.38252939, 0.18029944, 0.36735692],
       [0.11645525, 0.1110061 , 0.25500683, 0.26349705, 0.11772736, 0.25066496],
       [0.07174878, 0.07450245, 0.16855403, 0.16924101, 0.07246907, 0.15395085],
       [0.03959269, 0.03731672, 0.08268426, 0.08302634, 0.03421887, 0.07404255]])

nScale = len(scalez)

# ==========================================================
# Simulation timing
# ==========================================================
s_simtime = 11.0
s_recstart = 1.0
s_nettime = s_simtime - s_recstart

dt = 0.1  # ms
ms_simtime = s_simtime * 1000.0
ms_recstart = s_recstart * 1000.0
ms_nettime = int(s_nettime * 1000.0)
ms_duration = ms_nettime + 1

CURRENT_WINDOW_MS = 2000.0
FF_BIN_MS = 5.0
CORR_BIN_LIST_MS = [1.0, 5.0, 10.0]
STORE_ALL_VOLTAGES = False


list_pruning_types = [range(1), range(nPrun)]
list_pruning_stages = [range(1), range(1, nStage)]

list_paramList = [
    list(
        itertools.product(
            [ilb],                  # ilb
            range(nIndex),             # cp_index
            range(len(netnames)),      # inet
            range(nDegen),             # idtyp
            list_pruning_types[ips],   # idxprun
            list_pruning_stages[ips],  # istage
            range(nScale),             # iscale
        )
    )
    for ips in range(2)
]


# ==========================================================
# Structural helpers
# ==========================================================
def weightedFromAdjacency(cmtx, how_many_Ineurons, weight, orderIE=None):
    """
    Convert binary adjacency to weighted adjacency.

    Args:
        cmtx: sparse binary adjacency stored as M[tgt, src].
        how_many_Ineurons: number of inhibitory neurons at the start of orderIE.
        weight: four block weights [II, I->E, E->I, EE].
        orderIE: optional node order with inhibitory neurons first.

    Returns:
        COO weighted adjacency stored as W[tgt, src].
    """
    if orderIE is None:
        orderIE = np.arange(cmtx.shape[0])

    nrnI = orderIE[:how_many_Ineurons]
    nrnE = orderIE[how_many_Ineurons:]

    j_i2i, j_i2e, j_e2i, j_e2e = weight

    cmtx = cmtx.toarray().astype(np.float32)
    cmtx[np.ix_(nrnI, nrnI)] *= j_i2i
    cmtx[np.ix_(nrnE, nrnI)] *= j_i2e
    cmtx[np.ix_(nrnI, nrnE)] *= j_e2i
    cmtx[np.ix_(nrnE, nrnE)] *= j_e2e

    return coo_matrix(cmtx)


def groupEdgesPerNodeNoEIsort(edgez, Ninh, inout="out", perms=None):
    """
    Order edges by source or target after a random I/E-preserving permutation.

    Args:
        edgez: (m,2) edge list as (src, tgt).
        Ninh: number of inhibitory neurons in the current graph.
        inout: "out" to group by source, "in" to group by target.
        perms: optional (iperm, eperm) permutations.

    Returns:
        Permutation indices that reorder edgez.
    """
    if perms is None:
        iperm = np.random.permutation(Ninh)
        eperm = np.random.permutation(np.arange(Ninh, N0))
    else:
        iperm, eperm = perms

    ieperm = np.concatenate((iperm, eperm))
    iepermInv = ieperm.argsort()

    idx = 0 if inout == "out" else 1
    hedgez = edgez[:, idx]
    positions = iepermInv[hedgez]
    perm = np.argsort(positions)
    return perm


def trimSynapses(trim_params):
    """
    Prune a fraction of synapses according to an edge-ordering strategy.

    Args:
        trim_params: (edges, idxprun, istage), where edges are (src,tgt).

    Returns:
        COO binary adjacency of surviving synapses stored as M[tgt, src].
    """
    edges, idxprun, istage = trim_params

    if idxprun in [0, 1]:
        outin = ["out", "in"][idxprun]
        synperm = groupEdgesPerNodeNoEIsort(edges, Ninh=0, inout=outin)
        edges = edges[synperm]
    elif idxprun == 2:
        prm = np.random.permutation(len(edges))
        edges = edges[prm]
    elif idxprun == 3:
        pass
    elif idxprun == 4:
        edges = edges[::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}.")

    ecutt = int(istage * del_frac * len(edges))
    edges = edges[ecutt:]
    return coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 1], edges[:, 0])),
        shape=(N0, N0),
    )


def trimNeurons(trim_params):
    """
    Delete inhibitory and excitatory neurons according to a node strategy.

    Args:
        trim_params: (edges, idxprun, istage), where edges are (src,tgt).

    Returns:
        COO binary adjacency after neuron deletion and reindexing.
    """
    edges, idxprun, istage = trim_params
    odeg = np.bincount(edges[:, 0], minlength=N0)
    ideg = np.bincount(edges[:, 1], minlength=N0)
    degg = odeg + ideg

    prrm = np.arange(N0)
    np.random.shuffle(prrm)
    odd = np.column_stack((odeg, degg, prrm, np.arange(N0)))

    if idxprun == 0:
        sort = odd[:, 3][odd[:, 0].argsort()]
    elif idxprun == 1:
        sort = odd[:, 3][odd[:, 1].argsort()]
    elif idxprun == 2:
        sort = odd[:, 3][odd[:, 2].argsort()]
    elif idxprun == 3:
        sort = odd[:, 3][odd[:, 1].argsort()][::-1]
    elif idxprun == 4:
        sort = odd[:, 3][odd[:, 0].argsort()][::-1]
    else:
        raise ValueError(f"Invalid pruning index: {idxprun}.")

    sort = sort.astype(int)
    isort = sort[np.isin(sort, Inrn)]
    esort = sort[np.isin(sort, Enrn)]

    nidel = int(del_frac * NI * istage)
    nedel = int(del_frac * NE * istage)
    remaining = np.concatenate((isort[nidel:], esort[nedel:])).astype(int)

    xx = np.zeros((N0, N0), dtype=np.int8)
    xx[edges[:, 1], edges[:, 0]] = 1
    xx = xx[np.ix_(remaining, remaining)]
    return coo_matrix(xx)


def trimming(params):
    """
    Load ordered edges and apply synapse or neuron degeneration.

    Args:
        params: (cp_index, inet, idtyp, idxprun, istage).

    Returns:
        COO binary adjacency stored as M[tgt, src].
    """
    cp_index, inet, idtyp, idxprun, istage = params

    if USE_DEMO_CONNECTIVITY:
        rng = np.random.default_rng(1000 + 100 * cp_index + inet)
        cmtx = rng.binomial(1, 0.1, (N0, N0)).astype(np.int8)
        np.fill_diagonal(cmtx, 0)
        edgez = np.column_stack(np.nonzero(cmtx.T)).astype(int)
    else:
        netname = netnames[inet]
        ename = f"{netfold}/{netname}_relabeled_ordered_restored_{cp_index}.npz"
        edgez = np.load(ename)["ordedges"]

    if istage == 0:
        return coo_matrix(
            (np.ones(len(edgez), dtype=np.int8), (edgez[:, 1], edgez[:, 0])),
            shape=(N0, N0),
        )

    trim_params = (edgez, idxprun, istage)
    return trimNeurons(trim_params) if idtyp else trimSynapses(trim_params)


# ==========================================================
# Dynamic summaries
# ==========================================================
def firingRate(spike_data, newNI, newNE):
    """
    Compute mean and standard deviation of firing rates.

    Args:
        spike_data: (n_spikes,2) array with [neuron_id, spike_time_ms].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (2,3) array for mean/std rates across all/I/E populations.
    """
    network_size = newNI + newNE
    node_ids = spike_data[:, 0].astype(int)
    firing_rates = np.bincount(node_ids, minlength=network_size) / s_nettime

    mean_rates = [
        np.mean(firing_rates),
        np.mean(firing_rates[:newNI]),
        np.mean(firing_rates[newNI:]),
    ]
    sd_rates = [
        np.std(firing_rates),
        np.std(firing_rates[:newNI]),
        np.std(firing_rates[newNI:]),
    ]
    return np.array([mean_rates, sd_rates])


def cvISI(spike_data, newNI, newNE):
    """
    Compute coefficient of variation of inter-spike intervals.

    Args:
        spike_data: (n_spikes,2) array with [neuron_id, spike_time_ms].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (3,) array with CV-ISI for all/I/E populations.
    """
    node_ids = spike_data[:, 0].astype(int)
    spike_times = spike_data[:, 1]

    unique_nodes = np.unique(node_ids)
    cv_values = np.full(len(unique_nodes), np.nan, dtype=float)

    for idx, node in enumerate(unique_nodes):
        node_spikes = spike_times[node_ids == node]
        if len(node_spikes) > 1:
            isis = np.diff(node_spikes)
            mean_isi = np.mean(isis)
            if mean_isi != 0:
                cv_values[idx] = np.std(isis) / mean_isi

    return np.array(
        [
            np.nanmean(cv_values),
            np.nanmean(cv_values[:newNI]),
            np.nanmean(cv_values[newNI:]),
        ]
    )


def dynPart(spike_data, newNI, newNE):
    """
    Combine firing-rate and CV-ISI dynamic summaries.

    Args:
        spike_data: (n_spikes,2) array with [neuron_id, spike_time_ms].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (3,3) array containing rate mean, rate std, and CV-ISI.
    """
    rr = firingRate(spike_data, newNI, newNE)
    cv = cvISI(spike_data, newNI, newNE)
    return np.vstack((rr, cv))


def asynchrony_from_voltage_stats(sum_v, sum_v2, n_steps, newNI, newNE):
    """
    Estimate population asynchrony from accumulated voltage statistics.

    Args:
        sum_v: per-neuron sum of membrane voltage samples.
        sum_v2: per-neuron sum of squared membrane voltage samples.
        n_steps: number of voltage samples accumulated.
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        List with asynchrony for all/I/E populations.
    """
    if n_steps == 0:
        return [np.nan, np.nan, np.nan]

    mean_v = sum_v / n_steps
    var_v = np.maximum(sum_v2 / n_steps - mean_v**2, 0.0)

    def block_async(mu, var):
        denom = np.mean(var) + 1e-12
        nom = np.var(mu)
        return np.sqrt(nom / denom)

    all_a = block_async(mean_v, var_v)
    inh_a = block_async(mean_v[:newNI], var_v[:newNI])
    exc_a = block_async(mean_v[newNI:], var_v[newNI:])
    return [all_a, inh_a, exc_a]


def meanDegree(data, newNI, newNE):
    """
    Compute in/out/total degree summaries for a binary adjacency.

    Args:
        data: COO adjacency stored as M[tgt, src].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (6,3) array of mean and std degree summaries for all/I/E.
    """
    in_degrees = np.bincount(data.row, minlength=data.shape[0])
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    sum_degrees = in_degrees + out_degrees

    def pack(vals):
        return [np.mean(vals), np.mean(vals[:newNI]), np.mean(vals[newNI:])]

    def pack_std(vals):
        return [np.std(vals), np.std(vals[:newNI]), np.std(vals[newNI:])]

    return np.array(
        [
            pack(sum_degrees),
            pack(in_degrees),
            pack(out_degrees),
            pack_std(sum_degrees),
            pack_std(in_degrees),
            pack_std(out_degrees),
        ]
    )


def meanEffectiveLinkWeight(data, newNI, newNE):
    """
    Compute incoming/outgoing/total weighted-link summaries.

    Args:
        data: weighted sparse adjacency stored as W[tgt, src].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (6,3) array of mean and std effective-link weights for all/I/E.
    """
    arr = data.toarray()
    incoming = np.sum(arr, 1)
    outgoing = np.sum(arr, 0)
    total = incoming + outgoing

    def pack(vals):
        return [np.mean(vals), np.mean(vals[:newNI]), np.mean(vals[newNI:])]

    def pack_std(vals):
        return [np.std(vals), np.std(vals[:newNI]), np.std(vals[newNI:])]

    return np.array(
        [
            pack(total),
            pack(incoming),
            pack(outgoing),
            pack_std(total),
            pack_std(incoming),
            pack_std(outgoing),
        ]
    )


def meanBlockWeights(data, newNI, weight):
    """
    Compute block-wise mean/std weights and effective balance.

    Args:
        data: binary sparse adjacency stored as M[tgt, src].
        newNI: inhibitory neuron count after degeneration.
        weight: four block weights [II, I->E, E->I, EE].

    Returns:
        eff: scalar effective balance summary.
        mnsd: (2,4) array with block means and standard deviations.
    """
    wii, wie, wei, wee = weight
    arr = data.toarray()

    iiblock = arr[:newNI, :newNI] * wii
    ieblock = arr[newNI:, :newNI] * wie
    eiblock = arr[:newNI, newNI:] * wei
    eeblock = arr[newNI:, newNI:] * wee

    means = np.array(
        [
            np.abs(np.mean(iiblock)),
            np.abs(np.mean(ieblock)),
            np.abs(np.mean(eiblock)),
            np.abs(np.mean(eeblock)),
        ]
    )
    stds = np.array(
        [np.std(iiblock), np.std(ieblock), np.std(eiblock), np.std(eeblock)]
    )
    eff = means[0] + means[3] - means[1] - means[2]
    return eff, np.vstack((means, stds))


def contributionToPairwiseSharing(data, newNI, newNE):
    """
    Compute pairwise-sharing contribution from out-degree counts.

    Args:
        data: COO adjacency stored as M[tgt, src].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.

    Returns:
        (3,) array for all/I/E pairwise-sharing contribution.
    """
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    sharing_pairs = out_degrees * (out_degrees - 1) / 2.0
    newNN = newNI + newNE
    return np.array(
        [
            np.mean(sharing_pairs) / newNN,
            np.mean(sharing_pairs[:newNI]) / newNN,
            np.mean(sharing_pairs[newNI:]) / newNN,
        ]
    )


def fanoFactor(spike_data, newNI, newNE, ff_binsize_ms):
    """
    Compute Fano factor of binned spike counts.

    Args:
        spike_data: (n_spikes,2) array with [neuron_id, spike_time_ms].
        newNI: inhibitory neuron count after degeneration.
        newNE: excitatory neuron count after degeneration.
        ff_binsize_ms: bin width in milliseconds.

    Returns:
        List with Fano factor for all/I/E populations.
    """
    node_ids, spike_times = spike_data.T

    def meanff(spktime):
        if len(spktime) == 0:
            return np.nan
        last_spike_time = float(np.ceil(spktime[-1]))
        bins = np.arange(-0.05, last_spike_time + ff_binsize_ms + 0.05, ff_binsize_ms)
        psth, _ = np.histogram(spktime, bins)
        mean_psth = np.mean(psth) + 1e-12
        return np.var(psth) / mean_psth

    ff_all = meanff(spike_times)
    ff_inh = meanff(spike_times[node_ids < newNI])
    ff_exc = meanff(spike_times[node_ids >= newNI])
    return [ff_all, ff_inh, ff_exc]


def rebin(spike_train, bin_size):
    """
    Aggregate a spike-train matrix into wider time bins.

    Args:
        spike_train: (n_neurons,n_time) binary/count spike matrix.
        bin_size: number of original bins per output bin.

    Returns:
        Rebinned spike-count matrix.
    """
    n_bins = spike_train.shape[1] // bin_size
    return spike_train[:, : n_bins * bin_size].reshape(
        spike_train.shape[0], n_bins, bin_size
    ).sum(axis=2)


def spike_train_matrix(spike_list, n_neurons):
    """
    Convert spike list data into a dense spike-train matrix.

    Args:
        spike_list: (n_spikes,2) array with [neuron_id, spike_time_ms].
        n_neurons: number of neurons in the current graph.

    Returns:
        (n_neurons,ms_duration) uint8 spike matrix.
    """
    neuron_ids = spike_list[:, 0].astype(int)
    spike_times = spike_list[:, 1]
    t_idx = np.clip(np.floor(spike_times).astype(int), 0, ms_duration - 1)
    spkm = np.zeros((n_neurons, ms_duration), dtype=np.uint8)
    spkm[neuron_ids, t_idx] = 1
    return spkm


def meanCorr(spkk, params):
    """
    Compute pairwise spike-count correlations and Fano factors.

    Args:
        spkk: (n_spikes,2) array with [neuron_id, spike_time_ms].
        params: (cp_index, inet, idtyp, idxprun, istage, iscale).

    Returns:
        Flat list of correlations and Fano factors across configured bins.
    """
    cp_index, inet, idtyp, idxprun, istage, iweight = params
    newNI = NI - idtyp * int(del_frac * istage * NI)
    newNE = NE - idtyp * int(del_frac * istage * NE)
    N = newNI + newNE

    spike_matrix = spike_train_matrix(spkk, N)
    mncs = []
    ffs = []
    for bin_ms in CORR_BIN_LIST_MS:
        spktrain = rebin(spike_matrix, int(bin_ms))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            c = np.corrcoef(spktrain)
        mncs.extend(
            [
                np.nanmean(c[np.triu_indices(N, 1)]),
                np.nanmean(c[:newNI, :newNI][np.triu_indices(newNI, 1)]),
                np.nanmean(c[:newNI, newNI:]),
                np.nanmean(c[newNI:, newNI:][np.triu_indices(newNE, 1)]),
            ]
        )
        ffs.extend(fanoFactor(spkk, newNI, newNE, bin_ms))
    return mncs + ffs


# ==========================================================
# Current-based alpha-synapse simulation
# ==========================================================
def simulate_weighted_network(
    W,
    *,
    n_inh,
    dt=dt,
    t_sim=ms_simtime,
    burn_in=ms_recstart,
    tau_syn=tau_syn,
    delay=delay,
    c_ext=1,
    nu_ext=p_rate,
    j_ext=J_bg,
    seed=42,
    ff_bin_ms=FF_BIN_MS,
    store_all_voltages=STORE_ALL_VOLTAGES,
):
    """
    Simulate a weighted recurrent network with alpha synapses.

    Args:
        W: dense weighted adjacency stored as W[tgt, src].
        n_inh: number of inhibitory neurons at the start of W.
        dt: simulation step in milliseconds.
        t_sim: total simulation duration in milliseconds.
        burn_in: recording burn-in duration in milliseconds.
        tau_syn: synaptic time constant in milliseconds.
        delay: synaptic delay in milliseconds.
        c_ext: external input multiplicity.
        nu_ext: external Poisson rate in Hz.
        j_ext: external synaptic weight.
        seed: random seed for initial voltages and external input.
        ff_bin_ms: Fano-factor bin size retained for compatibility.
        store_all_voltages: if True, store voltage samples during recording.

    Returns:
        Dict with spike_data, asynchrony, and mean synaptic-current summaries.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=np.float32)
    N = W.shape[0]

    steps = int(round(t_sim / dt))
    burn_step = int(round(burn_in / dt))
    ref_steps = int(round(T_REF / dt))
    delay_steps = int(round(delay / dt))

    decay_syn = np.exp(-dt / tau_syn)
    decay_m = np.exp(-dt / TAU_M)
    alpha_scale = np.e / tau_syn
    lam_ext = c_ext * nu_ext * dt / 1000.0

    W_inh = W[:, :n_inh]
    W_exc = W[:, n_inh:]

    V = rng.uniform(V_RESET, V_TH, size=N).astype(np.float32)
    refr = np.zeros(N, dtype=np.int16)

    x_ex = np.zeros(N, dtype=np.float32)
    x_in = np.zeros(N, dtype=np.float32)
    I_ex = np.zeros(N, dtype=np.float32)
    I_in = np.zeros(N, dtype=np.float32)

    buf_ex = np.zeros((delay_steps + 1, N), dtype=np.float32)
    buf_in = np.zeros((delay_steps + 1, N), dtype=np.float32)

    spike_counts = np.zeros(N, dtype=np.int32)
    pop_spike_count = np.zeros(max(steps - burn_step, 0), dtype=np.int32)
    spike_pairs = []

    if store_all_voltages:
        V_all = np.zeros((steps - burn_step, N), dtype=np.float32)
    else:
        V_all = None

    sum_v = np.zeros(N, dtype=np.float64)
    sum_v2 = np.zeros(N, dtype=np.float64)
    v_count = 0

    current_start_ms = max(burn_in, t_sim - CURRENT_WINDOW_MS)
    current_start_step = int(round(current_start_ms / dt))
    sum_I_ex = np.zeros(3, dtype=np.float64)
    sum_I_in = np.zeros(3, dtype=np.float64)
    sum_I_syn = np.zeros(3, dtype=np.float64)
    current_count = 0

    for k in range(steps):
        buf_idx = k % (delay_steps + 1)
        x_ex += buf_ex[buf_idx]
        x_in += buf_in[buf_idx]
        buf_ex[buf_idx].fill(0.0)
        buf_in[buf_idx].fill(0.0)

        x_ex += (j_ext * alpha_scale) * rng.poisson(lam_ext, size=N)

        x_ex_old = x_ex
        x_in_old = x_in
        I_ex_old = I_ex
        I_in_old = I_in

        x_ex = decay_syn * x_ex_old
        x_in = decay_syn * x_in_old
        I_ex = decay_syn * (I_ex_old + dt * x_ex_old)
        I_in = decay_syn * (I_in_old + dt * x_in_old)
        I_syn = I_ex + I_in

        active = refr == 0
        V[active] = (
            E_L
            + (V[active] - E_L) * decay_m
            + (TAU_M / C_M) * (1.0 - decay_m) * I_syn[active]
        )
        V[~active] = V_RESET
        refr[~active] -= 1

        if k >= burn_step:
            local_k = k - burn_step
            pop_spike_count[local_k] = 0
            sum_v += V
            sum_v2 += V * V
            v_count += 1
            if store_all_voltages:
                V_all[local_k] = V

        if k >= current_start_step:
            sum_I_ex += [I_ex.mean(), I_ex[:n_inh].mean(), I_ex[n_inh:].mean()]
            sum_I_in += [I_in.mean(), I_in[:n_inh].mean(), I_in[n_inh:].mean()]
            sum_I_syn += [I_syn.mean(), I_syn[:n_inh].mean(), I_syn[n_inh:].mean()]
            current_count += 1

        spikers = np.where(V >= V_TH)[0]
        if spikers.size == 0:
            continue

        V[spikers] = V_RESET
        refr[spikers] = ref_steps

        future_idx = (k + delay_steps) % (delay_steps + 1)
        spk_inh = spikers[spikers < n_inh]
        spk_exc = spikers[spikers >= n_inh]

        if spk_inh.size > 0:
            buf_in[future_idx] += alpha_scale * W_inh[:, spk_inh].sum(axis=1)
        if spk_exc.size > 0:
            buf_ex[future_idx] += alpha_scale * W_exc[:, spk_exc - n_inh].sum(axis=1)

        if k >= burn_step:
            local_k = k - burn_step
            spike_counts[spikers] += 1
            pop_spike_count[local_k] = spikers.size
            t_rel = k * dt - burn_in
            spike_pairs.extend((idx, t_rel) for idx in spikers.tolist())

    spike_data = np.array(spike_pairs, dtype=float) if spike_pairs else np.empty((0, 2))
    # ff_bins = compute_population_ff(pop_spike_count, N, dt, ff_bin_ms)
    async_vals = asynchrony_from_voltage_stats(sum_v, sum_v2, v_count, n_inh, N - n_inh)

    mean_I_ex = sum_I_ex / max(current_count, 1)
    mean_I_in = sum_I_in / max(current_count, 1)
    mean_I_syn = sum_I_syn / max(current_count, 1)

    return {
        "spike_data": spike_data,
        # "spike_counts": spike_counts,
        # "pop_spike_count": pop_spike_count,
        # "V_all": V_all,
        "asynchrony": np.array(async_vals),
        # "ff_bin_ms": ff_bin_ms,
        # "ff_population": ff_bins,
        "mean_I_ex": mean_I_ex,
        "mean_I_in": mean_I_in,
        "mean_I_syn": mean_I_syn,
        # "lam_ext": lam_ext,
        # "delay_steps": delay_steps,
    }


# def compute_population_ff(pop_spike_count, n_neurons, dt, ff_bin_ms):
#     bin_steps = int(round(ff_bin_ms / dt))
#     if len(pop_spike_count) == 0 or bin_steps <= 0:
#         return np.nan
#     bins = np.add.reduceat(pop_spike_count, np.arange(0, len(pop_spike_count), bin_steps))
#     mean_bins = np.mean(bins) + 1e-12
#     return np.var(bins) / mean_bins


# ==========================================================
# Main pipeline replacement
# ==========================================================
def simulateAndStore(paramss, Jbg=None, prate=None):
    """
    Run degeneration, simulation, analysis, and qnt_*.npz storage.

    Args:
        paramss: (ilb, cp_index, inet, idtyp, idxprun, istage, iscale).
        Jbg: optional external synaptic weight override.
        prate: optional external Poisson rate override.

    Returns:
        None.
    """
    if prate is None:
        prate = p_rate
    if Jbg is None:
        Jbg = J_bg

    ilb = paramss[0]
    params = paramss[1:]
    qstrng1 = tuple([qntfold] + list(paramss))
    fname1 = "%s/qnt_%d_%d_%d_%d_%d_%d_%d.npz" % qstrng1
    os.makedirs(qntfold, exist_ok=True)

    cp_index, inet, idtyp, idxprun, istage, iscale = params
    
    ##########################
    ww = wg10 if ilb else wk #
    ##########################
    
    scale = scalez[iscale, inet]
    weight = scale * ww

    tparam = paramss[1:-1]
    bmtx = trimming(tparam)
    N = bmtx.shape[0]
    newNI = NI - idtyp * int(del_frac * istage * NI)
    newNE = N - newNI

    if iscale == 0:
        dgg = meanDegree(bmtx.copy(), newNI, newNE)
        shd = contributionToPairwiseSharing(bmtx.copy(), newNI, newNE)
        dgsh = np.vstack((dgg, shd))
    else:
        dgsh = None

    wmtx = weightedFromAdjacency(bmtx.copy(), newNI, weight)
    W = wmtx.toarray().astype(np.float32)

    eigvals = np.linalg.eigvals(W)
    spR = np.array(max(abs(eigvals)))
    esw = meanEffectiveLinkWeight(wmtx.copy(), newNI, newNE)
    eff, mnsdeff = meanBlockWeights(bmtx, newNI, weight)

    sim = simulate_weighted_network(
        W,
        n_inh=newNI,
        dt=dt,
        t_sim=ms_simtime,
        burn_in=ms_recstart,
        tau_syn=tau_syn,
        delay=delay,
        c_ext=1,
        nu_ext=prate,
        j_ext=Jbg,
        ff_bin_ms=FF_BIN_MS,
        store_all_voltages=STORE_ALL_VOLTAGES,
        seed=cp_index + 1000 * inet + 10000 * istage + 7 * iscale,
    )

    data = sim["spike_data"]
    if len(data) == 0:
        np.savez(
            fname1,
            dgsh=dgsh,
            radius=spR,
            esw=esw,
            mneff=eff,
            mnsdeff=mnsdeff,
            rvc=np.nan,
            sync=np.nan,
            ccff=np.nan,
            synI=np.nan,
        )
        return None

    rvc = dynPart(data, newNI, newNE)
    mncf = meanCorr(data, params)

    synI = np.column_stack((sim["mean_I_syn"], sim["mean_I_in"], sim["mean_I_ex"]))
    asynch = sim["asynchrony"]

    np.savez(
        fname1,
        dgsh=dgsh,
        radius=spR,
        esw=esw,
        mneff=eff,
        mnsdeff=mnsdeff,
        rvc=rvc,
        sync=asynch,
        ccff=mncf,
        synI=synI,
        # ff_pop=np.array([sim["ff_population"]]),
    )

    print(ilb, cp_index, inet, idtyp, idxprun, istage, iscale)
    # return {
    #     "params": paramss,
    #     "spike_data": data,
    #     "asynchrony": asynch,
    #     "synI": synI,
    #     # "ff_pop": sim["ff_population"],
    # }



