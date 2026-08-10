'''
Sweep the Ab (A-beta) input firing rate and record the PKC firing-rate response.

For each Ab input rate we regenerate the Ab VecStim spike file, (re)build and run
the network for a number of trials, measure the PKC population firing rate, then
plot PKC rate vs Ab input rate.
'''

import numpy as np
import matplotlib.pyplot as plt

from netpyne import sim
from spkt_gen import generate_new_SPKT


# ----------------------------- sweep settings -----------------------------
# Ab input rates to test, in Hz. NOTE: avoid 0 Hz -- poisson_generator divides
# by the rate (1.0/rate), so a 0 will blow up. Edit range/step as you like.
ab_rates = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

trials_per_rate = 3      # runs averaged per Ab rate (Ab spikes are random each run)
num_ab_cells    = 4      # must match numCells for the 'Ab' pop in netParams
SIM_DURATION    = 2000   # ms  (matches cfg.duration)
seed            = None   # None -> a different Poisson realization every run

SIM_DURATION_SEC = SIM_DURATION / 1000.0
# --------------------------------------------------------------------------


def pkc_rate_from_sim():
    """PKC population firing rate (Hz) from the just-completed simulation."""
    pkc_gids = set(sim.net.pops['PKC'].cellGids)
    n_pkc = max(len(pkc_gids), 1)
    spkids = list(sim.allSimData.get('spkid', []))
    n_spikes = sum(1 for gid in spkids if gid in pkc_gids)
    return n_spikes / (n_pkc * SIM_DURATION_SEC)


mean_pkc_rate = []
sem_pkc_rate  = []

for ab_rate in ab_rates:
    trial_rates = []
    for trial in range(trials_per_rate):
        print(f"Ab = {ab_rate} Hz   trial {trial + 1}/{trials_per_rate}")

        # Regenerate the Ab spike file at this input rate. This overwrites the
        # json that netParams loads, so it must happen BEFORE readCmdLineArgs.
        generate_new_SPKT(num_ab_cells, ab_rate, 0, SIM_DURATION, seed)

        simConfig, netParams = sim.readCmdLineArgs(
            simConfigDefault='cfg_mechanical.py',
            netParamsDefault='netParams_mechanical.py'
        )

        # Skip per-run plotting/saving during the sweep (faster, no popups).
        simConfig.analysis = {}
        simConfig.saveJson = False
        simConfig.savePickle = False

        sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)

        trial_rates.append(pkc_rate_from_sim())

    trial_rates = np.array(trial_rates)
    mean_pkc_rate.append(trial_rates.mean())
    # standard error of the mean across trials (0 if only one trial)
    sem_pkc_rate.append(
        trial_rates.std(ddof=1) / np.sqrt(len(trial_rates)) if len(trial_rates) > 1 else 0.0
    )
    print(f"  -> PKC mean rate = {mean_pkc_rate[-1]:.2f} Hz")


# ------------------------------- plot -------------------------------------
ab_rates      = np.array(ab_rates)
mean_pkc_rate = np.array(mean_pkc_rate)
sem_pkc_rate  = np.array(sem_pkc_rate)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(ab_rates, mean_pkc_rate, yerr=sem_pkc_rate,
            marker='o', capsize=3, linewidth=1.5)
ax.set_xlabel('Ab input firing rate (Hz)')
ax.set_ylabel('PKC firing rate (Hz)')
ax.set_title(f'PKC response vs Ab input  ({trials_per_rate} trials/point)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('pkc_vs_ab_rate.png', dpi=150)
print("saved figure -> pkc_vs_ab_rate.png")
plt.show()