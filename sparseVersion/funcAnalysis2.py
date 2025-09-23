import numpy as np
import scipy.io

from funcDegeneration import *
from parameters import *

def analyzeCurrent(excI, inhI, prms):
    inet, idtyp, cp_index, idxprun, istage, iweight = prms
    netname = all_network_types[inet]
    newNI = NI-idtyp*int(del_frac*istage*NI)
    excIptime = np.mean(excI, 1)
    inhIptime = np.mean(inhI, 1)

    i_excIptime = np.mean(excIptime[:newNI])  
    e_excIptime = np.mean(excIptime[newNI:])
    b_excI = np.mean(excIptime) 
    i_inhIptime = np.mean(inhIptime[:newNI])  
    e_inhIptime = np.mean(inhIptime[newNI:])
    b_inhI = np.mean(inhIptime) 

    quant = np.array([
        i_excIptime, 
        e_excIptime, 
        b_excI,
        i_inhIptime, 
        e_inhIptime, 
        b_inhI,
    ])
    synItime = np.vstack((
        excIptime,
        inhIptime))
    synIpop = np.vstack((
        np.mean(excI[:newNI], 0),
        np.mean(excI[newNI:], 0),
        np.mean(excI, 0),
        np.mean(inhI[:newNI], 0),
        np.mean(inhI[newNI:], 0),
        np.mean(inhI, 0)
        ))


    idict = {
        'mn_across_time': synItime,
        'mn_across_pop': synIpop
    }
    qstrng = tuple([qntfold]+list(prms))
    
    scipy.io.savemat('%s/currdict_%d_%d_%d_%d_%d_%d.mat'%qstrng, idict)
    # filename = f'{outdir}/mn_ei_Isyn__{netname}_{idtyp}_{cp_index}_{idxprun}_{istage}_{iweight}.mat'
    # scipy.io.savemat(filename, ddict)
    
    return None
    
    
def fanoFactor(data, newNI, newNE, ff_binsize):
    def meanff(spktime):
        '''
        - spktime is a spike time array
        output: the fano-factor 
        '''
        last_spike_time = int(np.ceil(spktime[-1]))
        bins = np.arange(-0.05, last_spike_time + 0.05, ff_binsize)
        psth, _ = np.histogram(spktime, bins)
        mean_psth = np.mean(psth)+ 1e-12
        return int(mean_psth != 0) * np.var(psth) / mean_psth
        
    # # if ff_binsize is None:
    # ff_binsize = 50

    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    ff_all = meanff(spike_times)
    ff_inh = meanff(spike_times[node_ids<newNI])
    ff_exc = meanff(spike_times[node_ids>=newNI])
    
    mnff = [ff_all, ff_inh, ff_exc]
    
    return mnff




def rebin(spike_train, bin_size):
    n_bins = spike_train.shape[1] // bin_size
    return spike_train[:, :n_bins * bin_size].reshape(spike_train.shape[0], n_bins, bin_size).sum(axis=2)


def spike_train_matrix(spike_list, n_neurons):
    neuron_ids, spike_times = spike_list.T.astype(int)
    spkm = np.zeros((n_neurons, duration_ms), dtype=np.uint8)
    spkm[neuron_ids, spike_times] = 1
    
    return spkm


def meanCorr(spkk, params):
    inet, idtyp, cp_index, idxprun, istage, iweight = params
    # print(params)
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = NE-idtyp*int(del_frac*istage*NE)
    N = newNI + newNE
    
    # sstrng = tuple([spkfold]+list(params))
    # spkk = np.load('%s/spikeData_%d_%d_%d_%d_%d_%d.npz'%sstrng)['data']    
    spike_matrix = spike_train_matrix(spkk, N)
    
    all_bin_list = [1, 10, 50, 100]
    mncs = []
    ffs = []
    for ibin, nbin in enumerate(all_bin_list):
        spktrain = rebin(spike_matrix, nbin)
        c = np.corrcoef(spktrain)
        mncaa = np.nanmean(c[np.triu_indices(N, 1)])
        mncei = np.nanmean(c[:newNI, newNI:])
        mncii = np.nanmean(c[:newNI, :newNI][np.triu_indices(newNI, 1)])
        mncee = np.nanmean(c[newNI:, newNI:][np.triu_indices(newNE, 1)])
        
        
        mncs = mncs + [mncaa, mncii, mncei, mncee]
        
        #ff
        f = fanoFactor(spkk, newNI, newNE, nbin)
        ffs = ffs + f
        
    
    return mncs + ffs
    
    
    
    
def firingRate(data, newNI, newNE):
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    node_ids = node_ids.astype(int)
    firing_rates = np.bincount(node_ids, minlength=network_size)
    recording_time = (simulation_time - start_record_time) # in seconds
    firing_rates = firing_rates/recording_time

    mean_rates = [
        np.mean(firing_rates), 
        np.mean(firing_rates[:newNI]),
        np.mean(firing_rates[newNI:])
    ]# for population, I population, Epopulation
    sd_rates = [
        np.std(firing_rates), 
        np.std(firing_rates[:newNI]),
        np.std(firing_rates[newNI:])
    ]# for population, I population, Epopulation
    
    mnrr = np.array([mean_rates, sd_rates])
    
    return mnrr

def asynchrony(vmm, newNI, newNE):
    '''
    vmdata should be in array format of (N0, Ndt)
    '''
    NN = newNI+newNE
    vms = np.reshape(vmm, (len(vmm)//NN, NN)).T

    nom_all = np.var(np.mean(vms, 1))
    denom_all = np.mean(np.var(vms, 1))
    asynch_all = np.sqrt(nom_all/denom_all)
    
    nom_inh = np.var(np.mean(vms[:newNI], 1))
    denom_inh = np.mean(np.var(vms[:newNI], 1))
    asynch_inh = np.sqrt(nom_inh/denom_inh)
    
    nom_exc = np.var(np.mean(vms[newNI:], 1))
    denom_exc = np.mean(np.var(vms[newNI:], 1))
    asynch_exc = np.sqrt(nom_exc/denom_exc)
    
    return [asynch_all, asynch_inh, asynch_exc]
    


    
    
    
def cvISI(data, newNI, newNE):    
    network_size = newNI + newNE
    node_ids, spike_times = data.T
    
    unique_nodes = np.unique(node_ids)
    cv_values = np.zeros(len(unique_nodes))
    for idx, node in enumerate(unique_nodes):
        node_spikes = spike_times[node_ids == node]   
        if len(node_spikes) > 1:  # as isi needs at least two spikes
            isis = np.diff(node_spikes)  
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_values[idx] = std_isi / mean_isi if mean_isi != 0 else 0
        else:
            cv_values[idx] = np.nan  
    mean_cv_all = np.nanmean(cv_values)
    mean_cv_inh = np.nanmean(cv_values[:newNI])
    mean_cv_exc = np.nanmean(cv_values[newNI:])
    mncv = np.array([mean_cv_all, mean_cv_inh, mean_cv_exc])
    return mncv



    
def dynPart(data, newNI, newNE):
    rr = firingRate(data, newNI, newNE)
    # ff = fanoFactor(data, newNI, newNE)
    cv = cvISI(data, newNI, newNE)
    return np.row_stack((rr, cv))    
    
    
    
    


def meanDegree(data, newNI, newNE): #<<<--- flag=unweighted to load
    '''
    - data is a list of sparse_data of unweighted adjacency matrix, number of Inhibitory and number of excitatory nodes.
    
    ''' 
    
    newNN = newNI + newNE
    
    in_degrees = np.bincount(data.row, minlength=data.shape[0])
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    sum_degrees = in_degrees + out_degrees
    
    mean_in_degrees = np.mean(in_degrees)
    mean_out_degrees = np.mean(out_degrees)
    mean_sum_degrees = np.mean(sum_degrees)
    
    mean_in_degrees_I = np.mean(in_degrees[:newNI])
    mean_out_degrees_I = np.mean(out_degrees[:newNI])
    mean_sum_degrees_I = np.mean(sum_degrees[:newNI])
    
    mean_in_degrees_E = np.mean(in_degrees[newNI:])
    mean_out_degrees_E = np.mean(out_degrees[newNI:])
    mean_sum_degrees_E = np.mean(sum_degrees[newNI:])
    
    std_in_degrees = np.std(in_degrees)
    std_out_degrees = np.std(out_degrees)
    std_sum_degrees = np.std(sum_degrees)
    
    std_in_degrees_I = np.std(in_degrees[:newNI])
    std_out_degrees_I = np.std(out_degrees[:newNI])
    std_sum_degrees_I = np.std(sum_degrees[:newNI])
    
    std_in_degrees_E = np.std(in_degrees[newNI:])
    std_out_degrees_E = np.std(out_degrees[newNI:])
    std_sum_degrees_E = np.std(sum_degrees[newNI:])
    
        
    mean_degree = [mean_sum_degrees, mean_sum_degrees_I, mean_sum_degrees_E]
    mean_in_degree = [mean_in_degrees, mean_in_degrees_I, mean_in_degrees_E]
    mean_out_degree = [mean_out_degrees, mean_out_degrees_I, mean_out_degrees_E]
    
    std_degree = [std_sum_degrees, std_sum_degrees_I, std_sum_degrees_E]
    std_in_degree = [std_in_degrees, std_in_degrees_I, std_in_degrees_E]
    std_out_degree = [std_out_degrees, std_out_degrees_I, std_out_degrees_E]
      
    mndg = np.array([
        mean_degree, 
        mean_in_degree, 
        mean_out_degree, 
        std_degree, 
        std_in_degree, 
        std_out_degree])
    
    return mndg

def meanEffectiveLinkWeight(data, newNI, newNE):
    # here data is weighted matrix
    newNN = newNI + newNE
    
    incoming = np.sum(data.toarray(), 1)
    outgoing = np.sum(data.toarray(), 0)
    projecting = incoming+outgoing
    
    mean_in_weight = np.mean(incoming)
    mean_out_weight = np.mean(outgoing)
    mean_sum_weight = np.mean(projecting)
    
    mean_in_weight_I = np.mean(incoming[:newNI])
    mean_out_weight_I = np.mean(outgoing[:newNI])
    mean_sum_weight_I = np.mean(projecting[:newNI])
    
    mean_in_weight_E = np.mean(incoming[newNI:])
    mean_out_weight_E = np.mean(outgoing[newNI:])
    mean_sum_weight_E = np.mean(projecting[newNI:])
    
    std_in_weight = np.std(incoming)
    std_out_weight = np.std(outgoing)
    std_sum_weight = np.std(projecting)
    
    std_in_weight_I = np.std(incoming[:newNI])
    std_out_weight_I = np.std(outgoing[:newNI])
    std_sum_weight_I = np.std(projecting[:newNI])
    
    std_in_weight_E = np.std(incoming[newNI:])
    std_out_weight_E = np.std(outgoing[newNI:])
    std_sum_weight_E = np.std(projecting[newNI:])
    
    mean_weight = [mean_sum_weight, mean_sum_weight_I, mean_sum_weight_E]
    mean_in_weight = [mean_in_weight, mean_in_weight_I, mean_in_weight_E]
    mean_out_weight = [mean_out_weight, mean_out_weight_I, mean_out_weight_E]
    
    std_weight = [std_sum_weight, std_sum_weight_I, std_sum_weight_E]
    std_in_weight = [std_in_weight, std_in_weight_I, std_in_weight_E]
    std_out_weight = [std_out_weight, std_out_weight_I, std_out_weight_E]
    
    mnwgt = [mean_weight, mean_in_weight, mean_out_weight]
    sdwgt = [std_weight, std_in_weight, std_out_weight]
    
    mnesw = np.array([
        mean_weight, 
        mean_in_weight, 
        mean_out_weight, 
        std_weight, 
        std_in_weight, 
        std_out_weight])
    
    return mnesw

def meanWeights(data, newNI):
    # here data is weighted matrix    
    iis = np.mean(data.toarray()[:newNI, :newNI])
    eis = np.mean(data.toarray()[:newNI, newNI:])
    ies = np.mean(data.toarray()[newNI:, :newNI:])
    ees = np.mean(data.toarray()[newNI:, newNI:])
    esw = ees+eis - iis - ies
    eff = iis+ees - eis - ies
    x = np.array([eff, esw, iis, eis, ies, ees] ) 
    return x

def statmatrix(mtx):
    mean_all = np.mean(mtx)
    std_all = np.std(mtx)
    xx2 = np.mean(mtx,1)
    xx1 = np.mean(mtx,0)
    mean_give = np.mean(xx1)
    std_give = np.std(xx1)
    mean_get = np.mean(xx2)
    std_get = np.std(xx2)
    return np.array([mean_all, mean_give, mean_get, std_all, std_give, std_get ])
    
    
    
       
def ieLinkAnalysis(data, newNI):
    # here data is weighted matrix  
    wmtx = data.toarray()
    amtx = (data.toarray() != 0).astype(int) 
    wmtcs = [
        wmtx[:newNI, :newNI], 
        wmtx[:newNI, newNI:], 
        wmtx[newNI:, :newNI:], 
        wmtx[newNI:, newNI:]]
    amtcs = [
        amtx[:newNI, :newNI], 
        amtx[:newNI, newNI:], 
        amtx[newNI:, :newNI:], 
        amtx[newNI:, newNI:]]
    
    
    apart = np.array([statmatrix(mtx) for mtx in amtcs])
    wpart = np.array([statmatrix(mtx) for mtx in wmtcs])
    
    # iis = np.sum()
    # eis = np.sum(mtx[:newNI, newNI:])
    # ies = np.sum(mtx[newNI:, :newNI:])
    # ees = np.sum(mtx[newNI:, newNI:])
    # esw = ees+eis - iis - ies
    # eff = iis+ees - eis - ies
    # x = np.array([eff, esw, iis, eis, ies, ees] )
    return np.array([apart, wpart])
    
    
def contributionToPairwiseSharing(data, newNI, newNE):
    newNN = newNI + newNE        
    out_degrees = np.bincount(data.col, minlength=data.shape[1])
    
    sharing_pairs = out_degrees*(out_degrees - 1)/2
    shared_by_I = sharing_pairs[:newNI]
    shared_by_E = sharing_pairs[newNI:]
    
    mean_shared = np.mean(sharing_pairs)/newNN
    mean_shared_I = np.mean(shared_by_I)/newNN
    mean_shared_E = np.mean(shared_by_E)/newNN
    
    mnsh = np.array([mean_shared, mean_shared_I, mean_shared_E])
    
    return mnsh
     
  


    
# def dynStrPart(dsparams, weight=None):
#     print(dsparams)
#     dpart = dynPart(dsparams)
#     spart = strPart(dsparams[:-1])
#     dscat = np.row_stack((dpart, spart))
#
#     mfold = mfolds[dsparams[0] in ['er', 'sw', 'sf']]
#     qfold = pfold+mfold+qntfold
#
#     strng = tuple([qfold]+list(dsparams[:-1])+[int(dsparams[-1])])
#
#     np.savetxt('%s/dynstr_%s%d%d%d%d%d.txt'%strng, dscat)
#     return
    
    
    

     


