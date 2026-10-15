'''
Network parameters (netParams) script for Spinal Dorsal Horn Network Model
from Sekiguchi et al. (2021)
'''

from netpyne import specs, sim
from neuron import h
import numpy as np
import cells
from spkt_gen import *
import json
import sys
sys.path.insert(0, 'spkt')  # adding path to spkt dir
import genrn

try: 
    from __main__ import cfg
except:
    from cfg_mechanical import cfg

PROB = .2  # probability of connection between pre and post populations
OG_INPUT = False  # whether to use original input spike trains or generate new ones
RATE_INCREASE = False  # whether to use increased input rates for Ab fiber
NUM_C_FIBERS = 50
NUM_AB_FIBERS = 20
AB_FIBER_RATE = 30  # Hz, mean rate of EACH Ab afferent
C_FIBER_RATE = 10
A_START = 1000
A_END = 1050


netParams = specs.NetParams()
_ab_mode = getattr(cfg, 'abInputMode', 'physiological')

#------------------------------------------------------------------------------
# Population
#------------------------------------------------------------------------------

### INPUT FREQUENCY FROM AFFERENT FIBERS WITH FIXED FACTOR ###
"""
with open('spkt/spkt_SAI_%s.json' %(cfg.freq), 'rb') as spkt_SAI: spkt_SAI = json.load(spkt_SAI)
with open('spkt/spkt_SAII_%s.json' %(cfg.freq), 'rb') as spkt_SAII: spkt_SAII = json.load(spkt_SAII)
with open('spkt/spkt_Ad_%s.json' %(cfg.freq), 'rb') as spkt_Ad: spkt_Ad = json.load(spkt_Ad)
"""
#with open('spkt/spkt_C_%s.json' %(cfg.freq), 'rb') as spkt_C: spkt_C = json.load(spkt_C)

### RETURN WHEN DONE WITH POISSON SWEEP ###
# with open('spkt/spkt.json', 'rb') as spkt_Ab: spkt_Ab = json.load(spkt_Ab)

_rate  = getattr(cfg, 'inputRate', 10.0)
_nAb   = getattr(cfg, 'numAb', 4)
_seed  = getattr(cfg, 'spktSeed', 1000)

if OG_INPUT:
    if RATE_INCREASE:
        with open('spkt/spkt_SAI_100mN_Ab10x.json', 'rb') as f: spkt_SAI  = json.load(f)
        with open('spkt/spkt_SAII_100mN_Ab10x.json', 'rb') as f: spkt_SAII = json.load(f)
    else:
        with open('spkt/spkt_SAI_100mN.json', 'rb') as f: spkt_SAI  = json.load(f)
        with open('spkt/spkt_SAII_100mN.json', 'rb') as f: spkt_SAII = json.load(f)
   # spkt_Ab = spkt_SAI + spkt_SAII          # 10 + 10 = 20 fibres
   # netParams.popParams['Ab'] = {'cellModel': 'VecStim', 'numCells': 20, 'spkTimes': spkt_Ab}
    netParams.popParams['Ab_SAI'] = {'cellModel': 'VecStim', 'numCells': 10, 'spkTimes': spkt_SAI}
    netParams.popParams['Ab_SAII'] = {'cellModel': 'VecStim', 'numCells': 10, 'spkTimes': spkt_SAII}
elif _ab_mode == 'poisson':
    # Homogeneous Poisson drive, rate-controlled by cfg.inputRate -- used by
    # poisson_sweep.py (and tests_cn.py) to build F-I curves. Each fiber gets
    # an independent realisation (seed = spktSeed + fiber index).
    def make_poisson_pop(rate, n_cells, t_stop, seed_base):
        trains = []
        for i in range(n_cells):
            seed = None if seed_base is None else seed_base + i
            trains.append(poisson_generator(rate=rate, t_start=0, t_stop=t_stop,
                                             seed=seed).tolist())
        return trains

    spkt_Ab = make_poisson_pop(_rate, _nAb, cfg.duration, _seed)
else:
    # inhomogeneous Poisson (thinning) drive, using the SAI/SAII cubic rate
    # time-course from spkt_gen.py, split evenly across cfg.numAb fibers
    rate_sai_vals,  t_sai  = rate_SAI()
    rate_saii_vals, t_saii = rate_SAII()

    _nSAI  = _nAb // 2
    _nSAII = _nAb - _nSAI

    def make_inhomog_pop(rate_vals, t_vals, n_cells, t_stop, seed_base):
        trains = []
        for i in range(n_cells):
            seed = None if seed_base is None else seed_base + i
            trains.append(inh_poisson_generator(rate=rate_vals, t=t_vals,
                                                 t_stop=t_stop, seed=seed))
        return trains

    spkt_SAI  = make_inhomog_pop(rate_sai_vals,  t_sai,  _nSAI,  cfg.duration, _seed)
    _seed_saii = None if _seed is None else _seed + _nSAI
    spkt_SAII = make_inhomog_pop(rate_saii_vals, t_saii, _nSAII, cfg.duration, _seed_saii)

    spkt_Ab = spkt_SAI + spkt_SAII  # combine SAI and SAII draws into one Ab population

 #   netParams.popParams['Ab'] = {'cellModel': 'VecStim', 'numCells': _nAb, 'spkTimes': spkt_Ab}

### GENERATE POISSON SPIKE TRAINS FOR C FIBERS ###
### GENERATE POISSON SPIKE TRAINS FOR Ab PULSE FIBERS (10 Hz, 2000-2030 ms) ###
NUM_AB_PULSE = 20
spkt_Ab_pulse = []
for i in range(NUM_AB_PULSE):
    spkt_Ab_pulse.append(poisson_generator(rate=AB_FIBER_RATE, t_start=A_START, t_stop=A_END, seed=None).tolist())
#netParams.popParams['Ab'] = {'cellModel': 'VecStim', 'numCells': NUM_AB_PULSE, 'spkTimes': spkt_Ab_pulse}

spkt_C = []
for i in range(NUM_C_FIBERS):
   spkt_C.append(poisson_generator(C_FIBER_RATE, t_stop=cfg.duration,seed= None).tolist())  # 10 Hz, 1000 ms, seed = i

spkt_C_pulse = make_volley_pop(n_cells=NUM_C_FIBERS, freq_hz=20, n_stim=5, jitter_ms=1.0, seed=None)

def make_poisson_pop(rate, n_cells, t_start, t_stop, seed_base):
        trains = []
        for i in range(n_cells):
            seed = None if seed_base is None else seed_base + i
            trains.append(poisson_generator(rate=rate, t_start=t_start, t_stop=t_stop,
                                             seed=seed).tolist())
        return trains

spkt_Ab_testing_30 = make_poisson_pop(rate=1, n_cells=_nAb, t_start=0, t_stop=1000, seed_base=_seed*30)
spkt_Ab_testing_20 = make_poisson_pop(rate=2, n_cells=_nAb, t_start=1000, t_stop=2000, seed_base=_seed*20)
spkt_Ab_testing_10 = make_poisson_pop(rate=4, n_cells=_nAb, t_start=2000, t_stop=3000, seed_base=_seed*10)
spkt_Ab_testing_5 = make_poisson_pop(rate=6, n_cells=_nAb, t_start=3000, t_stop=4000, seed_base=_seed*5)
spkt_Ab_testing_2 = make_poisson_pop(rate=8, n_cells=_nAb, t_start=4000, t_stop=5000, seed_base=_seed*2)
spkt_Ab_testing_1 = make_poisson_pop(rate=10, n_cells=_nAb, t_start=5000, t_stop=6000, seed_base=_seed*2)


blocks = [spkt_Ab_testing_1, spkt_Ab_testing_2, spkt_Ab_testing_5,
          spkt_Ab_testing_10, spkt_Ab_testing_20, spkt_Ab_testing_30]
spkt_Ab_staircase = [sum(trains, []) for trains in zip(*blocks)]

# below line was used to compare I neurons across both models with identical input
#spkt_Ab_30 = make_poisson_pop(rate=AB_FIBER_RATE, n_cells=4, t_start=1000, t_stop=1050, seed_base=1)


#netParams.popParams['C_PEP'] = {'cellModel': 'VecStim', 'numCells': NUM_C_FIBERS, 'spkTimes': spkt_C_pulse}  # input from peptidergic C fibers (pulse)
## PRIMARY AFFERENTS TIME-VARYING STIMULUS (NORMAL STIMULUS)
netParams.popParams['Ab'] = {'cellModel': 'VecStim', 'numCells': 20, 'spkTimes': spkt_Ab_staircase}  # input from Ab_slow adapting type I
#netParams.popParams['Ad'] = {'cellModel': 'VecStim', 'numCells': 20, 'spkTimes': spkt_Ad}  # input from Adelta
#netParams.popParams['C_PEP'] = {'cellModel': 'VecStim', 'numCells': NUM_C_FIBERS, 'spkTimes': spkt_C}  # input from peptidergic C fibers
#netParams.popParams['C_NP'] = {'cellModel': 'VecStim', 'numCells': 80, 'spkTimes': spkt_C}  # input from non-peptidergic C fibers

## SETTING OF SPINAL NEURONS
cells.PKCRule['conds'] = {'cellType': 'PKC'}
cells.INcellRule['conds'] = {'cellType': 'IN' }
cells.EXdelayedRule['conds'] = {'cellType': 'EXdl'}
#cells.CRRule['conds'] = {'cellType': 'CR'}
#cells.SOMRule['conds'] = {'cellType': 'SOM'}
#cells.EXinitialRule['conds'] = {'cellType': 'EXib'}
#cells.PROcellRule['conds'] = {'cellType': 'PRO'}

#cells.PROcellRule['secs']['soma']['threshold'] = 0

netParams.cellParams['PKC'] = cells.PKCRule
netParams.cellParams['VGLUT3Rule'] = cells.EXdelayedRule
netParams.cellParams['PVRule'] = cells.INcellRule
#netParams.cellParams['DORRule'] = cells.EXdelayedRule
#netParams.cellParams['TrCRule'] = cells.EXinitialRule
#netParams.cellParams['DYNRule'] = cells.INcellRule
#netParams.cellParams['SOMRule'] = cells.SOMRule
#netParams.cellParams['CRRule'] = cells.CRRule
#netParams.cellParams['ISLETRule'] = cells.INcellRule
#netParams.cellParams['NK1Rule'] = cells.PROcellRule

'''

## SETTING POPULATION OF SPINAL NEURONS
netParams.popParams['PKC' ] = {'cellType': 'PKC', 'numCells': 30} # PKCg+ neurons (excitatory)
netParams.popParams['VGLUT3'] = {'cellType': 'EXdl', 'numCells': 4} # VGLUT3+ neurons (excitatory)
netParams.popParams['PV'] = {'cellType': 'IN', 'numCells': 15} # PV+ neurons (inhibitory)
netParams.popParams['DOR' ] = {'cellType': 'EXdl', 'numCells': 30} # DOR+ neurons (excitatory)
netParams.popParams['TrC'] = {'cellType': 'EXib', 'numCells': 10} # Transient Central neurons (excitatory)
netParams.popParams['DYN'] = {'cellType': 'IN', 'numCells': 60} # Central/DYN+ neurons (inhibitory)
netParams.popParams['SOM'] = {'cellType': 'SOM', 'numCells': 15} # SOM+ neurons (excitatory)
netParams.popParams['CR'] = {'cellType': 'CR', 'numCells': 20} # CR+ neurons (excitatory)
netParams.popParams['ISLET'] = {'cellType': 'IN', 'numCells': 15} # Islet-type neurons (inhibitory)
netParams.popParams['NK1'] = {'cellType': 'PRO', 'numCells': 10} # NK1+ neurons (projection)

'''
#netParams.popParams['DOR' ] = {'cellType': 'EXdl', 'numCells': 180} # DOR+ neurons (excitatory)
netParams.popParams['PKC' ] = {'cellType': 'PKC', 'numCells': 30} # PKCg+ neurons (excitatory)
#netParams.popParams['VGLUT3'] = {'cellType': 'EXdl', 'numCells': 4} # VGLUT3+ neurons (excitatory)
netParams.popParams['PV'] = {'cellType': 'IN', 'numCells': 15} # PV+ neurons (inhibitory) 230 - (30+15+20) = 165


###################################################################################################################################
#   Synaptic Mechanisms
###################################################################################################################################
netParams.defaultThreshold = -30

netParams.synMechParams['AMPA'] = {'mod': 'AMPA_DynSyn'   , 'tau_rise': 0.1, 'tau_decay': 5            }
netParams.synMechParams['NMDA'] = {'mod': 'NMDA_DynSyn'   , 'tau_rise': 2  , 'tau_decay': 100          }
netParams.synMechParams['NK13'] = {'mod': 'NK1_DynSyn'    , 'tau_rise': 100, 'tau_decay': 1000         }
netParams.synMechParams['NK23'] = {'mod': 'NK1_DynSyn'    , 'tau_rise': 200, 'tau_decay': 3000         }
netParams.synMechParams['GABA'] = {'mod': 'GABAa_DynSyn'  , 'tau_rise': 0.1, 'tau_decay': 20, 'e': -70 }
netParams.synMechParams['GLY']  = {'mod': 'Glycine_DynSyn', 'tau_rise': 0.1, 'tau_decay': 10, 'e': -70 }

###################################################################################################################################
#   Connectivity Mechanisms   
###################################################################################################################################
# From Abeta Fibres to Spinal Interneurons
netParams.connParams['Ab_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'PKC'},
    'weight': cfg.Ab_EX_AMPA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['Ab_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'PKC'},
    'weight': cfg.Ab_EX_NMDA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['Ab_AMPA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'VGLUT3'},
    'weight': cfg.Ab_EX_AMPA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['Ab_NMDA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'VGLUT3'},
    'weight': cfg.Ab_EX_NMDA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['Ab_AMPA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'PV'},
    'weight': cfg.Ab_IN_AMPA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['Ab_NMDA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab'},
    'postConds': {'popLabel': 'PV'},
    'weight': cfg.Ab_IN_NMDA,
    'sec': 'dend',
    'probability': PROB,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'}


#### NEW FOR C FIBERS ###
# From Peptidergic C (C,TRPV1) to Spinal Interneurons
netParams.connParams['C_PEP_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.C_PKC_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.C_PKC_NMDA ,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}


'''
netParams.connParams['Ab_SAII_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.Ab_EX_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.Ab_EX_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 

netParams.connParams['Ab_SAII_AMPA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'VGLUT3'},  
    'weight': cfg.Ab_EX_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'VGLUT3'},  
    'weight': cfg.Ab_EX_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 

netParams.connParams['Ab_SAII_AMPA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PV'},  
    'weight': cfg.Ab_IN_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PV'},  
    'weight': cfg.Ab_IN_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 
'''

netParams.connParams['VGLUT3_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.VGLUT3_PKC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['VGLUT3_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.VGLUT3_PKC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['PV_GABA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PV'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.PV_GABA,          
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

netParams.connParams['PV_GLY->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PV'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.PV_GLY,        
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GLY'}


"""
netParams.connParams['Ab_SAI_AMPA->DYN'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAI'}, 
    'postConds': {'popLabel': 'DYN'},  
    'weight': cfg.Ab_IN_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.Ab_EX_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PKC'},  
    'weight': cfg.Ab_EX_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 

netParams.connParams['Ab_SAII_AMPA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'VGLUT3'},  
    'weight': cfg.Ab_EX_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->VGLUT3'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'VGLUT3'},  
    'weight': cfg.Ab_EX_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 

netParams.connParams['Ab_SAII_AMPA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PV'},  
    'weight': cfg.Ab_IN_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

netParams.connParams['Ab_SAII_NMDA->PV'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'PV'},  
    'weight': cfg.Ab_IN_NMDA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'NMDA'} 

netParams.connParams['Ab_SAII_AMPA->DYN'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ab_SAII'}, 
    'postConds': {'popLabel': 'DYN'},  
    'weight': cfg.Ab_IN_AMPA,           
    'sec': 'dend',
    'probability': 0.2,
    'delay': 1.0,
    'loc': 0.5,
    'synMech': 'AMPA'} 

# From Adelta to Spinal Interneurons
netParams.connParams['Ad_AMPA->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'DOR'},  
    'weight': cfg.Ad_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'AMPA'}

netParams.connParams['Ad_NMDA->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'DOR'},  
    'weight': cfg.Ad_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'NMDA'}

netParams.connParams['Ad_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.Ad_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'AMPA'}

netParams.connParams['Ad_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.Ad_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'NMDA'}

netParams.connParams['Ad_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.Ad_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'AMPA'}

netParams.connParams['Ad_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'Ad'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.Ad_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 5.0,
    'loc': 0.50,
    'synMech': 'NMDA'}

# From Peptidergic C (C,TRPV1) to Spinal Interneurons
netParams.connParams['C_PEP_AMPA->TrC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'TrC'},  
    'weight': cfg.C_TrC_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.C_EX_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.C_EX_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['C_PEP_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.C_EX_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.C_EX_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['C_PEP_AMPA->DYN'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'DYN'},  
    'weight': cfg.C_DYN_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_AMPA->ISLET'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'ISLET'},  
    'weight': cfg.C_ISLET_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_AMPA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.C_NK1_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_PEP_NMDA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.C_NK1_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['C_PEP_NK1->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_PEP'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.C_NK1_NK1,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NK13'}

# From Non-Peptidergic C (C,IB4) to Spinal Interneurons
netParams.connParams['C_NP_AMPA->TrC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'TrC'},  
    'weight': cfg.C_TrC_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_NP_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.C_EX_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_NP_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'SOM'},  
    'weight': cfg.C_EX_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['C_NP_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.C_EX_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_NP_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'CR'},  
    'weight': cfg.C_EX_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['C_NP_AMPA->DYN'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'DYN'},  
    'weight': cfg.C_DYN_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['C_NP_AMPA->ISLET'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'C_NP'}, 
    'postConds': {'popLabel': 'ISLET'},  
    'weight': cfg.C_ISLET_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 10.0,
    'loc': 0.5,
    'synMech': 'AMPA'}

    """

###################################################################################################################################
#   Connectivity Betweeen Spinal Neurons (inh>ex, ex>ex, ex/inh>projection)
###################################################################################################################################
# TO ePKC NEURONS:

"""
netParams.connParams['VGLUT3_AMPA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.VGLUT3_PKC_AMPA,           
    'probability': 0.8,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['VGLUT3_NMDA->PKC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'PKC'},  
    'weight': cfg.VGLUT3_PKC_NMDA,           
    'probability': 0.8,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}


# TO eDOR NEURONS
netParams.connParams['PV_GABA->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PV'}, 
    'postConds': {'popLabel':'DOR'},  
    'weight': cfg.PV_GABA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

netParams.connParams['PV_GLY->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PV'}, 
    'postConds': {'popLabel':'DOR'},  
    'weight': cfg.PV_GLY,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GLY'}

netParams.connParams['VGLUT3_AMPA->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'DOR'},  
    'weight': cfg.VGLUT3_PKC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['VGLUT3_NMDA->DOR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'DOR'},  
    'weight': cfg.VGLUT3_PKC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

# TO eTrC NEURONS
netParams.connParams['PKC_AMPA->TrC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PKC'}, 
    'postConds': {'popLabel':'TrC'},  
    'weight': cfg.PKC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['ISLET_GABA->TrC'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'ISLET'}, 
    'postConds': {'popLabel':'TrC'},  
    'weight': cfg.ISLET_GABA,       
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

# TO iISLET NEURONS
netParams.connParams['DYN_GABA->ISLET'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel':'ISLET'},  
    'weight': cfg.DYN_ISLET_GABA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

# TO iDYN NEURONS
netParams.connParams['ISLET_GABA->DYN'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'ISLET'}, 
    'postConds': {'popLabel':'DYN'},  
    'weight': cfg.ISLET_GABA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

# TO eSOM NEURONS
netParams.connParams['DYN_GABA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.DYN_EX_GABA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

netParams.connParams['DYN_GLY->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.DYN_EX_GLY,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GLY'}

netParams.connParams['TrC_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'TrC'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.TrC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['TrC_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'TrC'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.TrC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['PKC_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PKC'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.PKC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['PKC_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PKC'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.PKC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['VGLUT3_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.VGLUT3_SOM_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['VGLUT3_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'VGLUT3'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.VGLUT3_SOM_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['DOR_AMPA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DOR'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.DOR_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['DOR_NMDA->SOM'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DOR'}, 
    'postConds': {'popLabel':'SOM'},  
    'weight': cfg.DOR_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

# TO eCR NEURONS
netParams.connParams['DYN_GABA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.DYN_EX_GABA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

netParams.connParams['DYN_GLY->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.DYN_EX_GLY,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GLY'}

netParams.connParams['TrC_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'TrC'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.TrC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['TrC_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'TrC'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.TrC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['PKC_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PKC'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.PKC_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['PKC_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'PKC'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.PKC_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['DOR_AMPA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DOR'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.DOR_AMPA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['DOR_NMDA->CR'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DOR'}, 
    'postConds': {'popLabel':'CR'},  
    'weight': cfg.DOR_NMDA,           
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

# TO pNK1 NEURONS
netParams.connParams['SOM_AMPA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'SOM'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['SOM_NMDA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'SOM'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['SOM_NK1->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'SOM'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_NK1,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NK13'}

netParams.connParams['CR_AMPA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'CR'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_AMPA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'AMPA'}

netParams.connParams['CR_NMDA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'CR'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_NMDA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NMDA'}

netParams.connParams['CR_NK1->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'CR'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.EX_NK1_NK1,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'NK13'}

netParams.connParams['DYN_GABA->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.DYN_NK1_GABA,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GABA'}

netParams.connParams['DYN_GLY->NK1'] = {
    'oneSynPerNetcon': True,
    'preConds': {'popLabel': 'DYN'}, 
    'postConds': {'popLabel': 'NK1'},  
    'weight': cfg.DYN_NK1_GLY,
    'probability': 0.2,
    'sec': 'dend',
    'delay': 0.5, 
    'loc': 0.5,
    'synMech': 'GLY'}

    """