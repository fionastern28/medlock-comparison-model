'''
Script for testing various parts of the model
come back to the meanings of what i have here...
eventually this will likely be the init
'''
'''
from netpyne import sim
from neuron import h
from spkt_gen_cn import generate_new_SPKT

for i in range(5):
    generate_new_SPKT(4, 10, 0, 5000, None)
                        
    simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='cfg_mechanical_cn.py', netParamsDefault='netParams_mechanical_cn.py')

    # Create, Simulate, and Analyze the Network:
    sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

'''

from collections import defaultdict
import numpy as np
from netpyne import sim
from spkt_gen import generate_new_SPKT
import importlib


N_RUNS = 1

SIM_DURATION = 2000 #ms
SIM_DURATION_SEC = SIM_DURATION / 1000.0
INPUT_RATE = 30


# Store firing rate for each cell in each run
firing_rates = defaultdict(list)
pop_firing_rates = defaultdict(list)

for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")

    cfgmod = importlib.import_module('cfg_mechanical')
    cfg = cfgmod.cfg

    cfg.inputRate = float(INPUT_RATE)
    cfg.spktSeed = run * 231
    cfg.duration = SIM_DURATION
    cfg.simLabel = f"Circuit_Trial_Medlock_run{run}"


    npmod = importlib.import_module('netParams_mechanical')
    importlib.reload(npmod)

    sim.createSimulateAnalyze(netParams=npmod.netParams, simConfig=cfg)

    all_gids = [c.gid for c in sim.net.cells]
    pop_of = {c.gid: c.tags['pop'] for c in sim.net.cells}

    gids_by_pop = defaultdict(list)
    for gid in all_gids:
        gids_by_pop[pop_of[gid]].append(gid)

    spike_counts = {gid: 0 for gid in all_gids}
    for gid in sim.allSimData['spkid']:
        spike_counts[int(gid)] += 1

    for gid in all_gids:
        firing_rates[gid].append(spike_counts[gid] / SIM_DURATION_SEC)

    # new for population average
    for pop, gids in gids_by_pop.items():
        pop_rate = np.mean([spike_counts[gid] / SIM_DURATION_SEC for gid in gids])
        pop_firing_rates[pop].append(pop_rate)

    sim.clearAll()
# Average firing rate across runs

# commented here is average firing rate per cell
'''
avg_firing_rate = {
    gid: np.mean(rates)
    for gid, rates in firing_rates.items()
}

print("Average firing rates (Hz):")
for gid in sorted(avg_firing_rate):
    print(f"Cell {gid} ({pop_of[gid]}): {avg_firing_rate[gid]:.2f} Hz")
'''

avg_pop_firing_rate = {
    pop: np.mean(rates)
    for pop, rates in pop_firing_rates.items()
}

print("Average firing rates by population (Hz):")
for pop in sorted(avg_pop_firing_rate):
    print(f"{pop}: {avg_pop_firing_rate[pop]:.2f} Hz")



