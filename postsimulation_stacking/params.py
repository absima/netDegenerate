"""
Names, ordering indices, and labels for post-simulation stacked arrays.

This module defines the network/pruning display order, tuned scale tables,
variable names, math labels, and grouped index lists used by the downstream
stacking and feature-building scripts.
"""

import numpy as np
import scipy.io



        
wk = np.array([-3.0, -3.0, 1.5, 1.0])
wk = wk[[3,1,2,0]]
wg = np.array([-5.0, -5.0, 1.0, 1.0])
wg = wg[[3,1,2,0]]

onetnames = ['emp', 'er-I', 'er', 'sw', 'sf-I', 'sf']
netnames =  ['emp', 'er-I', 'sf-I', 'er', 'sw', 'sf']
dxnets = [onetnames.index(i) for i in netnames]



degens = ['syn', 'neuro']
osptypes = ['out', 'in', 'rnds', 'ord', 'res']
osptypes = ['s.axon', 's.dend', 's.rand', 's.cent', 's.peri'] # new name
sptypes = ['s.rand', 's.dend', 's.axon', 's.cent', 's.peri'] # new order
dxsptypes = [osptypes.index(i) for i in sptypes]

onptypes = ['iout', 'ideg', 'rndn', 'ddeg', 'dout']
onptypes = ['n.ax.peri', 'n.peri', 'n.rand', 'n.cent', 'n.ax.cent'] # new name
nptypes = ['n.rand', 'n.peri', 'n.ax.peri', 'n.cent', 'n.ax.cent'] # new order 
dxnptypes = [onptypes.index(i) for i in nptypes]

ptypes = [sptypes, nptypes]



tscales_l = np.array(
                [[ 2.36104176,  2.10117813,  8.36205398, 15.,          4.07917836,  8.22985433],
                [ 2.03397758,  1.85208968,  7.36691033, 15.,          3.29720195,  6.69365201],
                [ 1.74322534,  1.61947669,  6.51595704, 13.85699257,  2.70402919,  5.71200085],
                [ 1.48003433,  1.41982171,  5.80477279,  9.40490957,  2.20268297,  5.01522424],
                [ 1.26029526,  1.22338808,  5.06526478,  7.13113479,  1.74848692,  4.29025274],
                [ 1.08799983,  1.04531311,  4.66471615,  6.10926899,  1.35900594,  3.73250451],
                [ 0.88563154,  0.85364357,  4.15977286,  4.95400435,  1.06504039,  3.32877267],
                [ 0.71024377,  0.69222111,  3.79694545,  4.41396789,  0.76799608,  2.92281343],
                [ 0.52363655,  0.53185231,  3.40850918,  3.86667291,  0.54615268,  2.48892374],
                [ 0.32774549,  0.32708174,  3.05701863,  3.41669593,  0.30455865,  2.15034207]]).T

tscales_b = np.array(
                [[0.63281464, 0.5698673 , 1.92829821, 1.98056818, 1.38507591, 2.28388294],
                [0.51784524, 0.47380991, 1.383332  , 1.49430901, 0.96598979, 1.69858603],
                [0.41453699, 0.38935399, 1.12917462, 1.1317218 , 0.69012464, 1.21465548],
                [0.33603787, 0.31972571, 0.87096327, 0.88608737, 0.48513365, 0.92575807],
                [0.27360966, 0.26099724, 0.668916  , 0.67649131, 0.35033414, 0.68145037],
                [0.20852191, 0.20666759, 0.50507298, 0.50913976, 0.25003219, 0.49809397],
                [0.16127609, 0.15757984, 0.37492536, 0.38252939, 0.18029944, 0.36735692],
                [0.11645525, 0.1110061 , 0.25500683, 0.26349705, 0.11772736, 0.25066496],
                [0.07174878, 0.07450245, 0.16855403, 0.16924101, 0.07246907, 0.15395085],
                [0.03959269, 0.03731672, 0.08268426, 0.08302634, 0.03421887, 0.07404255]]).T
                                
                
varnames0 = [
    'deg', 
    'in-deg', 
    'out-deg', 
    'sd_deg', 
    'sd_in-deg', 
    'sd_out-deg', 
    'shared', ##
    'esw', 
    'in-esw', 
    'out-esw', 
    'sd_esw', 
    'sd_in-esw', 
    'sd_out-esw',
    'dEff',
    'specR',
    'mnwII',
    'mnwEI',
    'mnwIE',
    'mnwEE',
    'sigwII',
    'sigwEI',
    'sigwIE',
    'sigwEE',
    'synI',
    'synIi',
    'synIe',
    'rate', 
    'sd_rate', 
    'cv',
    'asyn',
    'ff', #'ffb1',
    'ffb10',
    'ffb50',
    'ffb100',
    'cc', #'ccb1',
    'ccb10',
    'ccb50',
    'ccb100',
    'cc_ei', # 'ccb1_ei',
    'ccb10_ei',
    'ccb50_ei',
    'ccb100_ei',
]   

varnames = [
    'deg', 
    'in-deg', 
    'out-deg', 
    'sd_deg', 
    'sd_in-deg', 
    'sd_out-deg', 
    'shared', ##
    'esw', 
    'in-esw', 
    'out-esw', 
    'sd_esw', 
    'sd_in-esw', 
    'sd_out-esw',
    #
    'dEff',
    'specR',
    'mnwII',
    'mnwEI',
    'mnwIE',
    'mnwEE',
    'sigwII',
    'sigwEI',
    'sigwIE',
    'sigwEE',
    'synI',
    'synIi',
    'synIe',
    #
    'rate', 
    'sd_rate', 
    'cv',
    'asyn',
    'ff', #'ffb1',
    # 'ffb10',
    # 'ffb50',
    # 'ffb100',
    'cc', #'ccb1',
    # 'ccb10',
    # 'ccb50',
    # 'ccb100',
    'cc_ei', # 'ccb1_ei',
    # 'ccb10_ei',
    # 'ccb50_ei',
    # 'ccb100_ei',
]    

ivarnames = [varnames0.index(j) for j in varnames]


mathvars = [
 '$\\mathbf{k}$',
 '$\\mathbf{k_i}$',
 '$\\mathbf{k_o}$',
 '$\\mathbf{\\sigma_k}$',
 '$\\mathbf{\\sigma_{k_i}}$',
 '$\\mathbf{\\sigma_{k_o}}$',
 '$\\mathbf{sh}$',
 '$\\mathbf{esw}$',
 '$\\mathbf{esw_i}$',
 '$\\mathbf{esw_o}$',
 '$\\mathbf{\\sigma_{esw}}$',
 '$\\mathbf{\\sigma_{esw_i}}$',
 '$\\mathbf{\\sigma_{esw_o}}$',
 '$\\mathbf{dEff}$',
 '$\\mathbf{specR}$',
 '$\\mathbf{mnw_{II}}$',
 '$\\mathbf{mnw_{EI}}$',
 '$\\mathbf{mnw_{IE}}$',
 '$\\mathbf{mnw_{EE}}$',
 '$\\mathbf{\\sigma_{w, II}}$',
 '$\\mathbf{\\sigma_{w, EI}}$',
 '$\\mathbf{\\sigma_{w, IE}}$',
 '$\\mathbf{\\sigma_{w, EE}}$',
 # '$\\mathbf{I}$',
 # '$\\mathbf{I_{i.Syn}}$',
 # '$\\mathbf{I_{e.Syn}}$',
 '$\\mathbf{I}$',
 '$\\mathbf{I}^{\\mathbf{I}}$',
 '$\\mathbf{I}^{\mathbf{E}}$',
 '$\\mathbf{\\lambda}$',
 '$\\mathbf{\\sigma_\\lambda}$',
 '$\\mathbf{CV}$',
 '$\\mathbf{SI}$',
 '$\\mathbf{FF}$',
 '$\\mathbf{cc}$',
 '$\\mathbf{ccEI}$']

vwpn = [
    'wEE', 
    'wEI', 
    'wIE', 
    'wII',
    'pEE', 
    'pEI', 
    'pIE', 
    'pII',
    'NE',
    'NI'
]

mvwpn =[
    '$\\mathbf{w_{EE}}$',
    '$\\mathbf{w_{EI}}$',
    '$\\mathbf{w_{IE}}$',
    '$\\mathbf{w_{II}}$',
    '$\\mathbf{p_{EE}}$',
    '$\\mathbf{p_{EI}}$',
    '$\\mathbf{p_{IE}}$',
    '$\\mathbf{p_{II}}$',
    '$\\mathbf{w_{NE}}$',
    '$\\mathbf{w_{NI}}$',
]



nvarnames = []
mvarnames = []
cases = ['', 'I', 'E']  # '' = no subscript

for ivar in range(len(varnames)):
    var = varnames[ivar]
    mvar = mathvars[ivar]
    for case in cases:
        if case == '':
            nvarnames.append(var)
            mvarnames.append(mvar)
        else:
            nvarnames.append(rf'{var}_{case}')
            mvarnames.append(mvar[:-1] + f'_{{\\mathbf{{{case}}}}}' + '$')
            # mvarnames.append(rf'$\mathbf{{{var}_{case}}}$')



nvarnames = np.reshape(nvarnames, (33,3))
mvarnames = np.reshape(mvarnames, (33,3))



dxstr1 = list(range(13)) # str 3 cat
dxstr2 = list(range(13,23)) # str 1 cat
dxcurr = list(range(23,26)) # current 3 cat
dxdyn1 = list(range(26,32)) # dyn 3 cat
dxdyn2 = list([32]) # dyn 1 cat






