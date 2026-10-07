import numpy as np
import json
import os

SCALE_FACTOR = 1 # temporary scale factor to test lower inputs!!


def rate_SAI():
    t = [x for x in np.arange(0, 10001, 1)]

    rate = []

    for i, key in enumerate(t):
        if i < len(t):
            freq_SAI = (-1.45433609113392e-10 * key ** 3 + 1.340603396708e-6 * key ** 2 - 0.00378224210238498 * key + 4.52737468545426) * 8 * SCALE_FACTOR
            rate.append(freq_SAI)

    return rate, t

def rate_SAII():
    t = [x for x in np.arange(0, 10001, 1)]

    rate = []

    for i, key in enumerate(t):
        if i < len(t):
            freq_SAII = (-1.45433609113392e-10 * key ** 3 + 1.340603396708e-6 * key ** 2 - 0.00378224210238498 * key + 4.52737468545426) * 8 * SCALE_FACTOR
            rate.append(freq_SAII)

    return rate, t

def poisson_generator(rate, t_start=0.0, t_stop=1000.0, seed=None):
    rng = np.random.RandomState(seed)
    n = (t_stop-t_start) / 1000.0 * rate
    number = np.ceil(n + 3*np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2*n), 100)
    if number > 0:
        isi = rng.exponential(1.0/rate, int(number)) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])
    spikes += t_start
    i = np.searchsorted(spikes, t_stop)
    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun
        
        t_last = spikes[-1] + rng.exponential(1.0/rate, 1)[0] * 1000.0
        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0/rate, 1)[0] * 1000.0
        
        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))
        
    return spikes

def inh_poisson_generator(rate, t, t_stop, seed=None):
    """
    Returns a SpikeTrain whose spikes are a realization of an inhomogeneous
    poisson process (dynamic rate). The implementation uses the thinning
    method.
    Inputs:
    -------
        rate   - an array of the rates (Hz) where rate[i] is active on interval
                    [t[i],t[i+1]]
        t      - an array specifying the time bins (in milliseconds) at which to
                    specify the rate
        t_stop - length of time to simulate process (in ms)
    Note:
    -----
        t_start=t[0]
    """
    rng = np.random.RandomState(seed)
    if np.shape(t) != np.shape(rate):
        raise ValueError('shape mismatch: t,rate must be of the same shape')
    # get max rate and generate poisson process to be thinned
    rmax = np.max(rate)
    ps = poisson_generator(rate=rmax, t_start=t[0], t_stop=t_stop, seed=None)
    # return empty if no spikes
    if len(ps) == 0:
        np.array([])

    # gen uniform rand on 0,1 for each spike
    rn = np.array(rng.uniform(0, 1, len(ps)))
    # instantaneous rate for each spike
    idx = np.searchsorted(t, ps) - 1
    spike_rate = np.array([rate[i] for i in idx])
    # thin and return spikes
    spike_train = ps[rn < spike_rate/rmax]
    return list(spike_train)

def generate_new_SPKT(num_cells, rate, tstart, tstop, seed):
    # --- generation parameters ---
    NUM_CELLS = num_cells
    RATE_HZ   = rate
    T_START   = tstart
    T_STOP    = tstop  # ms, matches cfg.duration in cfg_mechanical.py
    SEED_BASE = seed       # set to None for non-reproducible runs
 
    # each cell gets its own independent Poisson realization (different seed)
    spkt = []
    for cell_id in range(NUM_CELLS):
        seed = SEED_BASE + cell_id if SEED_BASE is not None else None
        spikes = poisson_generator(rate=RATE_HZ, t_start=T_START, t_stop=T_STOP, seed=seed)
        spkt.append(spikes.tolist())
 
    # save next to this script, in a "spkt" subfolder (matches project convention)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'spkt')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'spkt.json')
 
    with open(out_path, 'w') as f:
        json.dump(spkt, f)
 
    # quick sanity check
    for i, s in enumerate(spkt):
        print(f'cell {i}: {len(s)} spikes, mean rate = {len(s)/(T_STOP/1000.0):.2f} Hz')
 
    print(f'saved to {out_path}')


# from claude for testing windup
def make_volley_pop(n_cells, freq_hz, n_stim, t0=500.0, jitter_ms=1.0, seed=1000):
    rng = np.random.default_rng(seed)
    isi = 1000.0 / freq_hz
    stim_times = t0 + isi * np.arange(n_stim)
    return [sorted((stim_times + rng.normal(0, jitter_ms, n_stim)).tolist())
            for _ in range(n_cells)]
 