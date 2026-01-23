import numpy as np
import warnings

from parameters import *
# Uses: N, NI, NE, del_frac, delay, tau_syn, t_grid, ms_duration, s_nettime


def alphaKernel(t_diff, tau):
    """
    Unit-peak alpha kernel matching NEST psc_alpha.

    Args:
        t_diff: array of time differences (t - t_spike_effective)
        tau: synaptic time constant

    Returns:
        Alpha-kernel values (same shape as t_diff)
    """
    alpha = np.zeros_like(t_diff)
    mask = t_diff >= 0
    alpha[mask] = (t_diff[mask] / tau) * np.exp(1.0 - t_diff[mask] / tau)
    return alpha


def computeMeanSynapticCurrentsIE(params, raster_data, W):
    """
    Compute mean inhibitory and excitatory synaptic currents (population mean traces).

    Notes:
        Preserves original indexing assumptions and does not reinterpret W orientation
        to avoid changing outputs.

    Args:
        params: (cp_index, inet, idtyp, idxprun, istage, iweight)
        raster_data: (n_spikes, 2) array [pre_id, t_spike]
        W: connectivity/weight matrix used exactly as in original code

    Returns:
        Array of shape (T-1, 2):
            [:, 0] mean inhibitory current over neurons
            [:, 1] mean excitatory current over neurons
    """
    cp_index, inet, idtyp, idxprun, istage, iweight = params
    netname = netnames[inet]  # kept for parity/debugging if needed

    N = W.shape[0]
    new_NI = NI - idtyp * int(del_frac * istage * NI)
    new_NE = N - new_NI

    n_time = len(t_grid)

    I_syn_exc = np.zeros((N, n_time))
    I_syn_inh = np.zeros((N, n_time))

    pre_ids = raster_data[:, 0].astype(int)
    t_spikes = raster_data[:, 1]

    for pre_id, t_spike in zip(pre_ids, t_spikes):
        is_inhibitory = pre_id < new_NI

        post_ids = np.nonzero(W[pre_id])[0]
        weights = W[pre_id, post_ids]

        t_eff = t_spike + delay
        idx_start = np.searchsorted(t_grid, t_eff)
        if idx_start >= n_time:
            continue

        t_segment = t_grid[idx_start:] - t_eff
        alpha_vals = alphaKernel(t_segment, tau_syn)

        for post_id, w in zip(post_ids, weights):
            contrib = w * alpha_vals
            if is_inhibitory:
                I_syn_inh[post_id, idx_start:] += contrib
            else:
                I_syn_exc[post_id, idx_start:] += contrib

    inhI = I_syn_inh[:, 1:].T  # (T-1, N)
    excI = I_syn_exc[:, 1:].T  # (T-1, N)

    mean_inh = inhI.mean(axis=1)
    mean_exc = excI.mean(axis=1)

    I_mean_stacked = np.stack([mean_inh, mean_exc], axis=1)
    return I_mean_stacked


def meanSynCurrent(params, raster_data, A, weight, new_NI):
    """
    Compute mean inhibitory and excitatory synaptic currents over all neurons at each time step.

    Args:
        params: (cp_index, inet, idtyp, idxprun, istage, iweight)
        raster_data: (n_spikes, 2) array [pre_id, t_spike]
        A: adjacency used exactly as in original code
        weight: (W_II, W_IE, W_EI, W_EE)
        new_NI: inhibitory count for this network instance

    Returns:
        mnI: (2, 3) aggregated means (kept identical to original)
        mnIarray: stacked [mean_inh, mean_exc]
        inhI: inhibitory currents (T-1, N)
        excI: excitatory currents (T-1, N)
    """
    N = A.shape[0]
    n_time = len(t_grid)
    W_II, W_IE, W_EI, W_EE = weight

    I_syn_exc = np.zeros((N, n_time))
    I_syn_inh = np.zeros((N, n_time))

    pre_ids = raster_data[:, 0].astype(int)
    t_spikes = raster_data[:, 1]
    t_eff_all = t_spikes + delay

    for pre_id, t_eff in zip(pre_ids, t_eff_all):
        idx_start = np.searchsorted(t_grid, t_eff)
        if idx_start >= n_time:
            continue

        t_segment = t_grid[idx_start:] - t_eff
        alpha_vals = alphaKernel(t_segment, tau_syn)

        is_inh = pre_id < new_NI

        if is_inh:
            posts_II = np.where(A[:new_NI, pre_id])[0]
            posts_IE = np.where(A[new_NI:, pre_id])[0]

            I_syn_inh[posts_II, idx_start:] += W_II * alpha_vals
            I_syn_inh[posts_IE + new_NI, idx_start:] += W_IE * alpha_vals
        else:
            posts_EI = np.where(A[:new_NI, pre_id])[0]
            posts_EE = np.where(A[new_NI:, pre_id])[0]

            I_syn_exc[posts_EI, idx_start:] += W_EI * alpha_vals
            I_syn_exc[posts_EE + new_NI, idx_start:] += W_EE * alpha_vals

    inhI = I_syn_inh[:, 1:].T
    excI = I_syn_exc[:, 1:].T

    # Preserve original behavior (even if it looks unusual):
    mean_inh = inhI.mean(axis=0)
    mean_exc = excI.mean(axis=0)

    mnIarray = np.stack([mean_inh, mean_exc], axis=1)

    mnI = np.stack(
        (
            np.mean(mnIarray, 0),
            np.mean(mnIarray[:new_NI], 0),
            np.mean(mnIarray[new_NI:], 0),
        ),
        axis=1,
    )

    return mnI, mnIarray, inhI, excI


def fanoFactor(data, new_NI, new_NE, ff_binsize):
    """
    Compute Fano factor of binned spike counts for all/I/E populations.

    Args:
        data: (n_spikes, 2) [node_id, spike_time]
        new_NI: inhibitory count
        new_NE: excitatory count
        ff_binsize: bin size (same units as spike_time)

    Returns:
        [ff_all, ff_inh, ff_exc]
    """
    def meanff(spktime):
        if spktime.size == 0:
            return 0.0
        last_spike_time = int(np.ceil(spktime[-1]))
        bins = np.arange(-0.05, last_spike_time + 0.05, ff_binsize)
        psth, _ = np.histogram(spktime, bins)
        mean_psth = np.mean(psth) + 1e-12
        return float((mean_psth != 0) * np.var(psth) / mean_psth)

    node_ids, spike_times = data.T
    node_ids = node_ids.astype(int)

    ff_all = meanff(spike_times)
    ff_inh = meanff(spike_times[node_ids < new_NI])
    ff_exc = meanff(spike_times[node_ids >= new_NI])

    return [ff_all, ff_inh, ff_exc]


def rebin(spike_train, bin_size):
    """
    Rebin a spike train matrix along the time axis.

    Args:
        spike_train: (N, T) binary/count matrix
        bin_size: number of time bins to aggregate

    Returns:
        (N, T//bin_size) rebinned count matrix
    """
    n_bins = spike_train.shape[1] // bin_size
    trimmed = spike_train[:, : n_bins * bin_size]
    return trimmed.reshape(spike_train.shape[0], n_bins, bin_size).sum(axis=2)


def spikeTrainMatrix(spike_list, n_neurons):
    """
    Convert spike list to binary spike train matrix.

    Args:
        spike_list: (n_spikes, 2) [neuron_id, spike_time_ms] (time must be int-like)
        n_neurons: number of neurons

    Returns:
        (n_neurons, ms_duration) binary matrix
    """
    neuron_ids, spike_times = spike_list.T
    neuron_ids = neuron_ids.astype(int)
    spike_times = spike_times.astype(int)

    spkm = np.zeros((n_neurons, ms_duration), dtype=np.uint8)
    spkm[neuron_ids, spike_times] = 1
    return spkm


def meanCorr(spkk, params):
    """
    Compute mean pairwise correlations over several bin sizes + Fano factors.

    Args:
        spkk: (n_spikes, 2) [node_id, spike_time_ms]
        params: (cp_index, inet, idtyp, idxprun, istage, iweight)

    Returns:
        Flat list: correlations + fano factors (same ordering as original code)
    """
    cp_index, inet, idtyp, idxprun, istage, iweight = params

    new_NI = NI - idtyp * int(del_frac * istage * NI)
    new_NE = NE - idtyp * int(del_frac * istage * NE)
    N = new_NI + new_NE

    spike_matrix = spikeTrainMatrix(spkk, N)

    all_bin_list = [1, 10, 50, 100]
    mncs = []
    ffs = []

    for nbin in all_bin_list:
        spktrain = rebin(spike_matrix, nbin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            c = np.corrcoef(spktrain)

        mncaa = np.nanmean(c[np.triu_indices(N, 1)])
        mncei = np.nanmean(c[:new_NI, new_NI:])
        mncii = np.nanmean(c[:new_NI, :new_NI][np.triu_indices(new_NI, 1)])
        mncee = np.nanmean(c[new_NI:, new_NI:][np.triu_indices(new_NE, 1)])

        mncs += [mncaa, mncii, mncei, mncee]

        f = fanoFactor(spkk, new_NI, new_NE, nbin)
        ffs += f

    return mncs + ffs


def firingRate(data, new_NI, new_NE):
    """
    Compute firing rates (mean and std) for all/I/E populations.

    Args:
        data: (n_spikes, 2) [node_id, spike_time]
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        (2, 3) array:
            row 0: mean rates [all, I, E]
            row 1: std rates  [all, I, E]
    """
    network_size = new_NI + new_NE
    node_ids, spike_times = data.T
    node_ids = node_ids.astype(int)

    firing_rates = np.bincount(node_ids, minlength=network_size).astype(float)
    firing_rates = firing_rates / s_nettime

    mean_rates = [
        np.mean(firing_rates),
        np.mean(firing_rates[:new_NI]),
        np.mean(firing_rates[new_NI:]),
    ]
    sd_rates = [
        np.std(firing_rates),
        np.std(firing_rates[:new_NI]),
        np.std(firing_rates[new_NI:]),
    ]

    mnrr = np.array([mean_rates, sd_rates])
    return mnrr


def asynchrony(vmm, new_NI, new_NE):
    """
    Compute asynchrony metric from membrane traces.

    Args:
        vmm: flattened Vm samples (assumed shape compatible with (T*N,))
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        [asynch_all, asynch_inh, asynch_exc]
    """
    N = new_NI + new_NE
    vms = np.reshape(vmm, (len(vmm) // N, N)).T

    nom_all = np.var(np.mean(vms, 1))
    denom_all = np.mean(np.var(vms, 1))
    asynch_all = np.sqrt(nom_all / denom_all)

    nom_inh = np.var(np.mean(vms[:new_NI], 1))
    denom_inh = np.mean(np.var(vms[:new_NI], 1))
    asynch_inh = np.sqrt(nom_inh / denom_inh)

    nom_exc = np.var(np.mean(vms[new_NI:], 1))
    denom_exc = np.mean(np.var(vms[new_NI:], 1))
    asynch_exc = np.sqrt(nom_exc / denom_exc)

    return [asynch_all, asynch_inh, asynch_exc]


def cvIsi(data, new_NI, new_NE):
    """
    Compute mean CV of ISI for all/I/E.

    Args:
        data: (n_spikes, 2) [node_id, spike_time]
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        (3,) array [cv_all, cv_inh, cv_exc]
    """
    node_ids, spike_times = data.T
    node_ids = node_ids.astype(int)

    unique_nodes = np.unique(node_ids)
    cv_values = np.zeros(len(unique_nodes))

    for idx, node in enumerate(unique_nodes):
        node_spikes = spike_times[node_ids == node]
        if len(node_spikes) > 1:
            isis = np.diff(node_spikes)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_values[idx] = std_isi / mean_isi if mean_isi != 0 else 0.0
        else:
            cv_values[idx] = np.nan

    mean_cv_all = np.nanmean(cv_values)
    mean_cv_inh = np.nanmean(cv_values[:new_NI])
    mean_cv_exc = np.nanmean(cv_values[new_NI:])

    mncv = np.array([mean_cv_all, mean_cv_inh, mean_cv_exc])
    return mncv


def dynPart(data, new_NI, new_NE):
    """
    Aggregate dynamic summary metrics.

    Args:
        data: (n_spikes, 2) [node_id, spike_time]
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        Row-stacked array combining firing rate stats and CV-ISI.
    """
    rr = firingRate(data, new_NI, new_NE)
    cv = cvIsi(data, new_NI, new_NE)
    return np.row_stack((rr, cv))


def meanDegree(data, new_NI, new_NE):
    """
    Compute mean and std of in/out/total degree for all/I/E.

    Args:
        data: sparse COO-like with .row/.col and shape (N,N); assumed unweighted
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        (6, 3) array:
            rows: mean_total, mean_in, mean_out, std_total, std_in, std_out
            cols: [all, I, E]
    """
    in_degrees = np.bincount(data.row, minlength=data.shape[0])
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    sum_degrees = in_degrees + out_degrees

    mean_in_degrees = np.mean(in_degrees)
    mean_out_degrees = np.mean(out_degrees)
    mean_sum_degrees = np.mean(sum_degrees)

    mean_in_degrees_I = np.mean(in_degrees[:new_NI])
    mean_out_degrees_I = np.mean(out_degrees[:new_NI])
    mean_sum_degrees_I = np.mean(sum_degrees[:new_NI])

    mean_in_degrees_E = np.mean(in_degrees[new_NI:])
    mean_out_degrees_E = np.mean(out_degrees[new_NI:])
    mean_sum_degrees_E = np.mean(sum_degrees[new_NI:])

    std_in_degrees = np.std(in_degrees)
    std_out_degrees = np.std(out_degrees)
    std_sum_degrees = np.std(sum_degrees)

    std_in_degrees_I = np.std(in_degrees[:new_NI])
    std_out_degrees_I = np.std(out_degrees[:new_NI])
    std_sum_degrees_I = np.std(sum_degrees[:new_NI])

    std_in_degrees_E = np.std(in_degrees[new_NI:])
    std_out_degrees_E = np.std(out_degrees[new_NI:])
    std_sum_degrees_E = np.std(sum_degrees[new_NI:])

    mean_degree = [mean_sum_degrees, mean_sum_degrees_I, mean_sum_degrees_E]
    mean_in_degree = [mean_in_degrees, mean_in_degrees_I, mean_in_degrees_E]
    mean_out_degree = [mean_out_degrees, mean_out_degrees_I, mean_out_degrees_E]

    std_degree = [std_sum_degrees, std_sum_degrees_I, std_sum_degrees_E]
    std_in_degree = [std_in_degrees, std_in_degrees_I, std_in_degrees_E]
    std_out_degree = [std_out_degrees, std_out_degrees_I, std_out_degrees_E]

    mndg = np.array(
        [mean_degree, mean_in_degree, mean_out_degree, std_degree, std_in_degree, std_out_degree]
    )
    return mndg


def meanEffectiveLinkWeight(data, new_NI, new_NE):
    """
    Compute mean and std of incoming/outgoing/total weights for all/I/E.

    Args:
        data: weighted sparse matrix
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        (6, 3) array analogous to meanDegree().
    """
    dense = data.toarray()
    incoming = np.sum(dense, 1)
    outgoing = np.sum(dense, 0)
    projecting = incoming + outgoing

    mean_in_weight = np.mean(incoming)
    mean_out_weight = np.mean(outgoing)
    mean_sum_weight = np.mean(projecting)

    mean_in_weight_I = np.mean(incoming[:new_NI])
    mean_out_weight_I = np.mean(outgoing[:new_NI])
    mean_sum_weight_I = np.mean(projecting[:new_NI])

    mean_in_weight_E = np.mean(incoming[new_NI:])
    mean_out_weight_E = np.mean(outgoing[new_NI:])
    mean_sum_weight_E = np.mean(projecting[new_NI:])

    std_in_weight = np.std(incoming)
    std_out_weight = np.std(outgoing)
    std_sum_weight = np.std(projecting)

    std_in_weight_I = np.std(incoming[:new_NI])
    std_out_weight_I = np.std(outgoing[:new_NI])
    std_sum_weight_I = np.std(projecting[:new_NI])

    std_in_weight_E = np.std(incoming[new_NI:])
    std_out_weight_E = np.std(outgoing[new_NI:])
    std_sum_weight_E = np.std(projecting[new_NI:])

    mean_weight = [mean_sum_weight, mean_sum_weight_I, mean_sum_weight_E]
    mean_in_weight = [mean_in_weight, mean_in_weight_I, mean_in_weight_E]
    mean_out_weight = [mean_out_weight, mean_out_weight_I, mean_out_weight_E]

    std_weight = [std_sum_weight, std_sum_weight_I, std_sum_weight_E]
    std_in_weight = [std_in_weight, std_in_weight_I, std_in_weight_E]
    std_out_weight = [std_out_weight, std_out_weight_I, std_out_weight_E]

    mnesw = np.array(
        [mean_weight, mean_in_weight, mean_out_weight, std_weight, std_in_weight, std_out_weight]
    )
    return mnesw


def meanBlockWeights(data, new_NI, weight):
    """
    Compute block-wise mean/std weights and an effective balance metric.

    Args:
        data: weighted matrix (sparse or dense)
        new_NI: inhibitory count
        weight: (wii, wie, wei, wee)

    Returns:
        eff: scalar
        mnsdblocks: (2,4) array [means; stds]
    """
    wii, wie, wei, wee = weight
    dense = data.toarray() if hasattr(data, "toarray") else np.asarray(data)

    iiblock = dense[:new_NI, :new_NI] * wii
    ieblock = dense[new_NI:, :new_NI] * wie
    eiblock = dense[:new_NI, new_NI:] * wei
    eeblock = dense[new_NI:, new_NI:] * wee

    mniis = np.abs(np.mean(iiblock))
    mnies = np.abs(np.mean(ieblock))
    mneis = np.abs(np.mean(eiblock))
    mnees = np.abs(np.mean(eeblock))

    sdiis = np.std(iiblock)
    sdies = np.std(ieblock)
    sdeis = np.std(eiblock)
    sdees = np.std(eeblock)

    eff = mniis + mnees - mneis - mnies

    mnsdblocks = np.array(
        [
            [mniis, mnies, mneis, mnees],
            [sdiis, sdies, sdeis, sdees],
        ]
    )
    return eff, mnsdblocks


def contributionToPairwiseSharing(data, new_NI, new_NE):
    """
    Compute mean contribution to pairwise sharing based on out-degree.

    Args:
        data: sparse COO-like with .col
        new_NI: inhibitory count
        new_NE: excitatory count

    Returns:
        (3,) array [mean_shared, mean_shared_I, mean_shared_E]
    """
    N = new_NI + new_NE
    out_degrees = np.bincount(data.col, minlength=data.shape[1])

    sharing_pairs = out_degrees * (out_degrees - 1) / 2.0
    shared_by_I = sharing_pairs[:new_NI]
    shared_by_E = sharing_pairs[new_NI:]

    mean_shared = np.mean(sharing_pairs) / N
    mean_shared_I = np.mean(shared_by_I) / N
    mean_shared_E = np.mean(shared_by_E) / N

    mnsh = np.array([mean_shared, mean_shared_I, mean_shared_E])
    return mnsh
