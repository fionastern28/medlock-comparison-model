'''
F-I sweep for ONE model. Drop a copy into each project directory, edit the
CONFIG block below, and run it there:

    cd ~/path/to/model_A && python fi_sweep_cn.py
    cd ~/path/to/model_B && python fi_sweep_cn.py

Each run writes  fi_<label>.csv  into the project folder. Copy both CSVs into a
new folder alongside plot_fi_overlay.py and run that to get the overlaid figure.

Running each model in its own directory means each uses its own compiled mod
files and its own cells/cfg modules, with no chance of one model's imports
shadowing the other's.

PAIRING: keep INPUT_RATES, SEEDS and numAb identical between the two models.
Same seed + same numAb => the two models see the exact same Poisson input
trains, which lets the plotter compute a paired difference. It checks this and
falls back to an unpaired overlay if they don't match.

REQUIREMENT: the netParams file must build its Ab population from
cfg.inputRate / cfg.spktSeed / cfg.duration (the make_poisson_pop block), not
from a fixed spkt JSON. Otherwise the input rate never changes and the curve
comes out flat.
'''

import os
import re
import csv
import sys
import importlib

import numpy as np

from netpyne import sim


# ===========================================================================
# CONFIG -- edit per project
# ===========================================================================
MODEL_LABEL       = 'medlock'                 # legend text; must be unique
NETPARAMS_MODULE  = 'netParams_mechanical'    # module basename, no .py
CFG_MODULE        = 'cfg_mechanical'          # module basename, no .py
TARGET_POP        = 'PV'                     # population to measure

# keep these IDENTICAL across the models you intend to compare
INPUT_RATES = [1, 3, 5, 7, 9, 12, 15, 20]#, 55, 60, 65, 70]  # Hz per afferent
SEEDS       = [1056, 3008, 5200, 601]  # RNG seeds for Poisson trains
DURATION    = 2000.0    # ms per run
TRANSIENT   = 500.0     # ms discarded at the start of each run
# ===========================================================================


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def slug(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def measure_pop_rate(pop_label, t_start, t_stop):
    '''Mean firing rate (Hz) of pop_label over [t_start, t_stop], per cell.'''
    if pop_label not in sim.net.allPops:
        raise KeyError("population '{}' not found; available: {}".format(
            pop_label, sorted(sim.net.allPops.keys())))

    gids = sim.net.allPops[pop_label]['cellGids']
    if len(gids) == 0:
        return np.nan

    spkt = np.asarray(sim.allSimData.get('spkt', []), dtype=float)
    spkid = np.asarray(sim.allSimData.get('spkid', []), dtype=float)
    if spkt.size == 0:
        return 0.0

    mask = (spkt >= t_start) & (spkt < t_stop) & np.isin(spkid, gids)
    window_s = (t_stop - t_start) / 1000.0
    return int(mask.sum()) / window_s / len(gids)

# commented out and replaced with other run_one per claud 10/20/26
'''
def run_one(input_rate, seed):
    #Run a single simulation, return (output_rate_Hz, numAb).
    cfgmod = importlib.import_module(CFG_MODULE)
    cfg = cfgmod.cfg

    cfg.inputRate = float(input_rate)
    cfg.spktSeed = seed
    cfg.duration = DURATION
    cfg.abInputMode = 'poisson' 

    # keep the sweep fast and quiet: no plots, no saved files, no traces
    cfg.analysis = {}
    cfg.recordTraces = {}
    cfg.saveJson = False
    cfg.savePickle = False
    cfg.verbose = False
    cfg.printPopAvgRates = False

    # reload so fresh Poisson trains are drawn with this rate/seed
    npmod = importlib.import_module(NETPARAMS_MODULE)
    importlib.reload(npmod)

    sim.createSimulate(netParams=npmod.netParams, simConfig=cfg)
    rate = measure_pop_rate(TARGET_POP, TRANSIENT, DURATION)
    n_ab = getattr(cfg, 'numAb', 4)
    sim.clearAll()

    return rate, n_ab
'''

def run_one(input_rate, seed):
    ''' Run a single simulation, return (output_rate_Hz, numAb).'''
    cfgmod = importlib.import_module(CFG_MODULE)
    cfg = cfgmod.cfg

    cfg.inputRate = float(input_rate)
    cfg.spktSeed = seed
    cfg.duration = DURATION
    cfg.abInputMode = 'poisson'
    cfg.numAb = 20

    # keep the sweep fast and quiet: no plots, no saved files, no traces
    cfg.analysis = {}
    cfg.recordTraces = {}
    cfg.saveJson = False
    cfg.savePickle = False
    cfg.verbose = False
    cfg.printPopAvgRates = False

    # reload so fresh Poisson trains are drawn with this rate/seed
    npmod = importlib.import_module(NETPARAMS_MODULE)
    importlib.reload(npmod)

    n_ab = getattr(cfg, 'numAb', 20)

    # Sweep network = Ab -> PV only, regardless of which pops are enabled
    # in netParams. Pop must be named 'Ab' to match the medlock connParams.
    netParams = npmod.netParams
    for pop in list(netParams.popParams.keys()):
        del netParams.popParams[pop]
    netParams.popParams['Ab'] = {'cellModel': 'VecStim', 'numCells': n_ab,
                                 'spkTimes': npmod.spkt_Ab}
    netParams.popParams['PV'] = {'cellType': 'IN', 'numCells': 15}

    sim.createSimulate(netParams=netParams, simConfig=cfg)
    rate = measure_pop_rate(TARGET_POP, TRANSIENT, DURATION)
    sim.clearAll()

    return rate, n_ab

def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fi_{}.csv'.format(slug(MODEL_LABEL)))

    rows = []
    n_ab = None
    total = len(INPUT_RATES) * len(SEEDS)
    done = 0

    for input_rate in INPUT_RATES:
        for seed in SEEDS:
            rate, n_ab = run_one(input_rate, seed)
            done += 1
            rows.append({
                'model': MODEL_LABEL,
                'pop': TARGET_POP,
                'num_ab': n_ab,
                'duration_ms': DURATION,
                'transient_ms': TRANSIENT,
                'input_rate_per_afferent_Hz': input_rate,
                'total_input_rate_Hz': input_rate * n_ab,
                'seed': seed,
                'output_rate_Hz': rate,
            })
            print('  [{:>3d}/{:<3d}] input {:>6.1f} Hz | seed {:>5d} | '
                  '{} = {:6.2f} Hz'.format(done, total, input_rate, seed,
                                           TARGET_POP, rate))

    fields = ['model', 'pop', 'num_ab', 'duration_ms', 'transient_ms',
              'input_rate_per_afferent_Hz', 'total_input_rate_Hz',
              'seed', 'output_rate_Hz']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print('\nwrote {}'.format(out_path))
    print('copy this next to the other model\'s CSV and run plot_fi_overlay.py')


if __name__ == '__main__':
    main()