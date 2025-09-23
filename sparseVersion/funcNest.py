import numpy as np
import nest, time
import scipy.io

from funcDegeneration import *
from funcAnalysis2 import *
from parameters import *




  
def simulateAndStore(params):
    inet, idtyp, cp_index, idxprun, istage, iweight = params
    netname = all_network_types[inet]
    tparam = (netname, idtyp, cp_index, idxprun, istage)

    bmtx = trimming(tparam)
    N = bmtx.shape[0]
    
    newNI = NI-idtyp*int(del_frac*istage*NI)
    newNE = N - newNI
    print(inet, idtyp, cp_index, idxprun, istage, iweight)

    

    qstrng = tuple([qntfold]+list(params))

    dictt = {}
    ### computing structural features -- deg, shared
    if iweight==0:
        dgg = meanDegree(bmtx.copy(), newNI, newNE)
        shd = contributionToPairwiseSharing(bmtx.copy(), newNI, newNE) # feature 2
        dgsh = np.row_stack((dgg, shd))
        dictt['dgsh'] = dgsh
        # np.savetxt('%s/deg_and_shared_%s%d%d%d%d%d.txt'%qstrng, dgsh)

    ### computing structural features, esw    
    scale = scalers[inet, cp_index, iweight]
    weight = scaledWeight(netname, scale)
    wmtx = weightedFromAdjacency(bmtx.copy(), newNI, weight)
    
    
    esw = meanEffectiveLinkWeight(wmtx.copy(), newNI, newNE) # feature 3
    dictt['esw'] = esw
    # np.savetxt('%s/esw_%s%d%d%d%d%d.txt'%qstrng, esw)

    linksta = ieLinkAnalysis(wmtx.copy(), newNI)
    dictt['linksta'] = linksta
    


    nest.ResetKernel()

    nest.set_verbosity('M_WARNING')
    nest.set(print_time=False)

    nrnall = nest.Create('iaf_psc_alpha', N)
    nest.Connect(nrnall, nrnall, 'all_to_all', syn_spec={'weight': wmtx.toarray(), 'delay': delay})


    bg = nest.Create('poisson_generator', params={'rate':p_rate})#, 'start': start, 'stop': stop})
    nest.Connect(bg, nrnall, syn_spec={'weight':J_bg, 'delay':delay})

    ms_simtime = simulation_time*1000
    ms_recstart = start_record_time*1000

    # mm = nest.Create('multimeter', 1, {'start': ms_recstart, 'stop': ms_simtime, "record_from": ["V_m"], "record_to": "memory"})
    
    mm = nest.Create("multimeter", 1, 
        {
            "start": ms_recstart, 
            "stop": ms_simtime,
            "record_from": [
                "V_m",
                "I_syn_ex", 
                "I_syn_in"]}
        )

    nest.Connect(mm, nrnall)
    

    spkD = nest.Create('spike_recorder', params={'start': ms_recstart, 'stop': ms_simtime})
    nest.Connect(nrnall, spkD)

    endbuild=time.time()

    # simulating
    nest.Simulate(ms_simtime)
    endsimulate= time.time()
    
    wmtx = None

    spkSend = nest.GetStatus(spkD)[0]['events']['senders']
    spkTime = nest.GetStatus(spkD)[0]['events']['times']

    data = np.column_stack((spkSend, spkTime))
    data[:,1] = data[:,1]-ms_recstart
    data[:,0] = data[:,0] - 1 ##### index from 0

    rvc = dynPart(data, newNI, newNE)
    dictt['rvc'] = rvc
    mncf = meanCorr(data, params)
    dictt['mnccff'] = mncf
    data = None
    
    vmvm = nest.GetStatus(mm)[0]['events']['V_m']
    asynch = asynchrony(vmvm, newNI, newNE)

    dictt['asynch'] = asynch
    vmvm = None
    
    
    leng = int(ms_simtime-ms_recstart)-1

    Isynex = np.reshape(nest.GetStatus(mm)[0]['events']['I_syn_ex'], (N, leng))
    Isynin = np.reshape(nest.GetStatus(mm)[0]['events']['I_syn_in'], (N, leng))
    
    a = analyzeCurrent(Isynex, Isynin, params)
    Isynex = None
    Isynin = None
    scipy.io.savemat('%s/qnt_%d_%d_%d_%d_%d_%d.mat'%qstrng, dictt)
    
    # np.savez_compressed('%s/spikeData_%d_%d_%d_%d_%d_%d.npz'%sstrng, data=data)


    return #dynstr
    



