'''
Average and plot Medlock soma voltage for PKC and PV (raw, spikes included).
Run from the medlock-comparison-model folder:  python3 avg_medlock.py
Needs a run with cfg.recordCells = ['PKC', 'PV'] so every cell's v_soma is saved.
'''
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob('data/Circuit_Trial_Medlock*_data.json'))
if not files:
    raise SystemExit('No Medlock data files found in data/')

V = {'PKC': [], 'PV': []}
for f in files:
    d = json.load(open(f))
    t = np.array(d['simData']['t'])
    vsoma = d['simData']['v_soma']
    for pop in V:
        gids = d['net']['pops'][pop]['cellGids']
        missing = [g for g in gids if f'cell_{g}' not in vsoma]
        if missing:
            print(f'WARNING {f}: {len(missing)}/{len(gids)} {pop} cells have no v_soma '
                  f"(set cfg.recordCells = ['PKC', 'PV'] and re-run)")
        V[pop] += [vsoma[f'cell_{g}'] for g in gids if f'cell_{g}' in vsoma]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, pop, col in zip(axes, V, ['#2a6fdb', '#d1495b']):
    if not V[pop]:
        ax.set_title(f'{pop}: no recorded cells')
        continue
    arr = np.array(V[pop])                    # (n_cells, n_timepoints)
    mean, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.fill_between(t, mean - sd, mean + sd, color=col, alpha=0.2, lw=0, label='± SD')
    ax.plot(t, mean, color=col, lw=0.8, label='mean')
    ax.set_title(f'Medlock {pop}: average soma voltage (n = {len(arr)} cells, {len(files)} run(s))')
    ax.set_ylabel('V (mV)')
    ax.legend(loc='upper right', frameon=False)
    print(f'{pop}: {arr.shape[0]} cells averaged, overall mean {mean.mean():.2f} mV')
axes[-1].set_xlabel('Time (ms)')
fig.tight_layout()
fig.savefig('data/medlock_avg_voltage.png', dpi=200)
print('Saved data/medlock_avg_voltage.png')
#plt.show()