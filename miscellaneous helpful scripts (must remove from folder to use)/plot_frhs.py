'''Python port of FiringRateHist.m (Medlock et al. 2022) -- Fig. 2A right panel.

Pipeline (matches the original MATLAB exactly):
    0.025 ms binarized PSTH
    -> Gaussian kernel convolution (sigma = 100 ms, support +/-5 sigma)
    -> rescale so the trace mean equals the mean per-neuron firing rate (spk/s)

A fixed baseline run (data/100mN_data.json by default) is always plotted, drawn
solid; every file passed on the command line is overlaid on top of it with its
own dash pattern. Colour always encodes the cell group, never the run.

IMPORTANT: this script requires the REAL cell-to-population gid mapping, stored
in net['pops'][label]['cellGids']. That is only written to the output file if
cfg.saveDataInclude includes 'net', e.g. in cfg_mechanical.py:

    cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']

Guessing gid ranges from population order/size is NOT safe: it was tried and,
checked against ground-truth popRates, silently scrambled population identities
(a near-silent population appeared as one of the most active). So this script
refuses to guess and errors out instead.

Examples
--------
    # baseline alone, all 14 traces in 4 panels
    python3 plot_frh.py --cmap SDH-colormap.mat

    # one comparison against the baseline: baseline solid, 10mN dashed
    python3 plot_frh.py --data data/10mN_data.json \
        --cmap SDH-colormap.mat --groups AB,VGLUT3,PV,PKC

    # two comparisons with explicit legend labels, all on one axes
    python3 plot_frh.py --data data/10mN_data.json data/50mN_data.json \
        --labels 10mN 50mN --groups NK1 --single-panel --autoscale

    # different baseline, or none at all
    python3 plot_frh.py --baseline data/50mN_data.json --data data/10mN_data.json
    python3 plot_frh.py --no-baseline --data data/10mN_data.json data/50mN_data.json

    python3 plot_frhs.py --baseline data/Circuit_Trial_Medlock_data.json --data Circuit_Trial_data.json

    #update
    python3 plot_frhs.py --data data/Circuit_Trial_data.json --groups AB,VGLUT3,PV
    python3 plot_frhs.py --data data/Circuit_Trial_run0_data.json --groups AB,PV,PKC,VGLUT3 --autoscale
    python3 plot_frhs.py --data data/Circuit_Trial_data.json --groups PKC,AB,VGLUT3
    python3 plot_frhs.py  --groups AB,PV,PKC,VGLUT3 --autoscale

    python3 plot_frhs.py  --groups AB,PKC --autoscale --data data/Circuit_Trial_Medlock_data.json

    # repeated trials of one condition: mean trace +/- SD shading, vs. baseline
    python3 plot_frhs.py --trials data/Circuit_Trial_run*.json  --groups AB,VGLUT3,PV,PKC --autoscale

    # two conditions (e.g. medlock vs cn), each as its own mean +/- SD band,
    # overlaid on the same axes, no baseline
    python3 plot_frhs.py --no-baseline \
        --trials data/Circuit_Trial_Medlock_run*_data.json --trials-label medlock \
        --trials2 data/Circuit_Trial_run*_data.json --trials2-label cn \
        --groups AB,VGLUT3,PV,PKC --autoscale --out medlock_vs_cn_frh.png

        python3 plot_frhs.py --no-baseline --trials data/Circuit_Trial_Medlock_run*_data.json --trials-label medlock --groups AB,PKC,C,TRPV1 --autoscale --out medlock_vs_cn_frh.png

        python3 plot_frhs.py --no-baseline --data data/Circuit_Trial_Medlock_run0_data.json --autoscale
'''

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- constants --

END_TIME = 5.0     # s, simulated duration (EndTime in FiringRateHist.m)
RES      = 0.025   # ms, sampling resolution of simData.t
KWID     = 100.0   # ms, Gaussian kernel sigma (kWid in FiringRateHist.m)

# Default reference run, plotted unless --no-baseline is given. Override with
# --baseline if the file lives somewhere else.
#BASELINE = 'data/100mN_data.json'
BASELINE = 'data/Circuit_Trial_Medlock_data.json'

# The baseline is always solid; comparison files cycle through these. Colour is
# reserved for the cell group, so runs are distinguished by dash pattern only.
BASELINE_STYLE = '-'
COMPARE_STYLES = ['--', '-.', ':', (0, (5, 1, 1, 1, 1, 1)), (0, (7, 2))]

# Dedicated line styles + fill hatches for --trials/--trials2/... groups, chosen
# for maximum visual contrast between exactly a couple of overlaid mean+/-SD
# bands (COMPARE_STYLES above is tuned for --data run cycling and its first
# two patterns, dashed vs dash-dot, look too similar to each other at a glance).
TRIALS_STYLES = ['--', ':', '-.', (0, (3, 1, 1, 1))]
TRIALS_HATCHES = [None, '//', '\\\\', 'xx']

# Declaration order in netParams_mechanical.py. Used ONLY to index the colormap
# rows -- never to infer gids.
POP_ORDER = ['Ab_SAI', 'Ab_SAII', 'Ad', 'C_PEP', 'C_NP', 'PKC', 'VGLUT3', 'PV',
             'DOR', 'TrC', 'DYN', 'SOM', 'CR', 'ISLET', 'NK1']

# Some sim configs (e.g. Circuit_Trial_*) never split Ab into SAI/SAII and
# just have a single 'Ab' population. Colour-lookup only: reuse Ab_SAI's row
# so POP_ORDER/colormap indices for every other population stay unshifted.
COLOR_ALIAS = {'Ab': 'Ab_SAI'}

# (display label, constituent populations, subplot index)
# Mirrors ReCellNum / NewAffNames in FiringRateHist.m: only Abeta is merged;
# the two C-fiber populations stay separate.
GROUPS = [
    ('AB',      ['Ab_SAI', 'Ab_SAII'], 0),
    ('Ad',      ['Ad'],                0),
    ('C,TRPV1', ['C_PEP'],             0),
    ('C,IB4',   ['C_NP'],              0),
    ('PKC',     ['PKC'],               1),
    ('VGLUT3',  ['VGLUT3'],            1),
    ('DOR',     ['DOR'],               1),
    ('TrC',     ['TrC'],               1),
    ('SOM',     ['SOM'],               1),
    ('CR',      ['CR'],                1),
    ('PV',      ['PV'],                2),
    ('DYN',     ['DYN'],               2),
    ('ISLET',   ['ISLET'],             2),
    ('NK1',     ['NK1'],               3),
]

SUBPLOT_TITLES = ['Afferents', 'Excitatory interneurons',
                  'Inhibitory interneurons', 'Projection (pNK1)']
SUBPLOT_YLIM   = [(-5, 50), (-5, 150), (-5, 75), (-5, 50)]
SUBPLOT_YTICK  = [[0, 25, 50], [0, 50, 100, 150], [0, 25, 50, 75], [0, 25, 50]]

# Fallback population set for a group, tried per-run when the group's normal
# pops (above) aren't all present. E.g. some sim configs never split Ab into
# SAI/SAII and just save a single 'Ab' population.
GROUP_ALT_POPS = {'AB': ['Ab']}


def resolve_pops(name, pops, have):
    '''Return the pop list to use for this group on a run with population
    set `have`: the group's normal pops if all present, else its alt pops
    if all of those are present, else None (group unavailable on this run).
    '''
    if all(p in have for p in pops):
        return pops
    alt = GROUP_ALT_POPS.get(name)
    if alt and all(p in have for p in alt):
        return alt
    return None

# ------------------------------------------------------------------ loading --


def _missing_gids_msg(path):
    return (
        "\nNo per-cell gid mapping (net['pops'][label]['cellGids']) found in:\n"
        "  %s\n\n"
        "This script will not guess gid ranges from population order or size --\n"
        "that approach was verified to silently scramble population identities.\n\n"
        "Fix: add 'net' to cfg.saveDataInclude in cfg_mechanical.py:\n"
        "  cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']\n"
        "then rerun the simulation and pass the new output file to this script.\n"
        % path
    )


def load_sim(path):
    '''Return (spkt, spkid, t, gid_map, labels).

    gid_map is the ground-truth {popLabel: np.ndarray(gids)} read from
    net.pops[label].cellGids. Raises RuntimeError if it is absent.
    '''
    if path.endswith('.mat'):
        try:
            from scipy.io import loadmat
            m = loadmat(path, struct_as_record=False, squeeze_me=True)
            sd, net = m['simData'], m['net']
            spkt  = np.asarray(sd.spkt, float)
            spkid = np.asarray(sd.spkid, int)
            t     = np.asarray(sd.t, float)
            if not hasattr(net, 'pops'):
                raise RuntimeError(_missing_gids_msg(path))
            gid_map, labels = {}, []
            for label in net.pops._fieldnames:
                pop = getattr(net.pops, label)
                gids = getattr(pop, 'cellGids', None)
                if gids is None:
                    continue
                gid_map[label] = np.atleast_1d(np.asarray(gids, int))
                labels.append(label)
        except NotImplementedError:                      # MATLAB v7.3 / HDF5
            import mat73
            m = mat73.loadmat(path)
            sd, net = m['simData'], m['net']
            spkt  = np.asarray(sd['spkt'], float)
            spkid = np.asarray(sd['spkid'], int)
            t     = np.asarray(sd['t'], float)
            if 'pops' not in net:
                raise RuntimeError(_missing_gids_msg(path))
            gid_map, labels = {}, []
            for label, pop in net['pops'].items():
                if 'cellGids' not in pop:
                    continue
                gid_map[label] = np.atleast_1d(np.asarray(pop['cellGids'], int))
                labels.append(label)
    else:
        with open(path) as fh:
            blob = json.load(fh)
        sd = blob['simData']
        spkt  = np.asarray(sd['spkt'], float)
        spkid = np.asarray(sd['spkid'], int)
        if 't' in sd:
            t = np.asarray(sd['t'], float)
        else:
            # Older/incomplete runs can be missing simData['t'] entirely --
            # NetPyNE only ever writes it if cfg.recordTraces was non-empty
            # at the time of the run (see netpyne/sim/setup.py:setupRecording).
            # Reconstruct a matching time axis from cfg.duration (always saved
            # in simConfig) and this script's fixed sampling resolution (RES),
            # rather than requiring a full rerun just to get the raster/rate
            # data, which doesn't depend on recordTraces at all.
            duration = blob.get('simConfig', {}).get('duration')
            if duration is None:
                raise RuntimeError(
                    "\n%s has no simData['t'] and no simConfig['duration'] to "
                    "reconstruct it from -- can't recover a time axis for this "
                    "file." % path
                )
            t = np.arange(0, float(duration) + RES / 2, RES)
            print("NOTE: %s has no simData['t'] (cfg.recordTraces was empty "
                  "when it was generated); reconstructed a %.3f ms-resolution "
                  "axis from cfg.duration=%s instead of rerunning."
                  % (path, RES, duration))
        net = blob.get('net', {})
        if 'pops' not in net:
            raise RuntimeError(_missing_gids_msg(path))
        gid_map, labels = {}, []
        for label, pop in net['pops'].items():
            gids = pop.get('cellGids')
            if not gids:
                continue
            gid_map[label] = np.atleast_1d(np.asarray(gids, int))
            labels.append(label)

    if not gid_map:
        raise RuntimeError(_missing_gids_msg(path))
    return spkt, spkid, t, gid_map, labels


def run_label(path):
    '''Legend label for a run: file stem with a trailing _data stripped.'''
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-5] if stem.endswith('_data') else stem


# ------------------------------------------------------------------- kernel --


def gaussian_kernel(sigma_ms=KWID, res=RES):
    '''Port of the kernel construction in KernelPSTH.m.'''
    t = np.arange(-5 * sigma_ms, 5 * sigma_ms + res / 2, res)
    g = np.exp(-t ** 2 / (2 * sigma_ms ** 2)) / np.sqrt(2 * np.pi * sigma_ms ** 2)
    return g / (np.abs(g).sum() * res)


def kernel_psth(psth, g):
    '''Port of the conv + trim in KernelPSTH.m (1-based -> 0-based indexing).

    fftconvolve rather than a direct convolution: the kernel is 40001 samples
    and the PSTH is 200000, so a direct conv is ~8e9 MACs. Results agree to
    floating-point error.
    '''
    full = fftconvolve(psth, g, mode='full')
    half = int(round(len(g) / 2.0))
    trimmed = full[half - 1: len(full) - half + 1]
    return trimmed[:len(psth)]


def compute_trace(spkt, spkid, gids, t, g, end_time=END_TIME):
    '''Return (trace, n_cells, mean_rate) for one group of populations.'''
    times = spkt[np.isin(spkid, gids)]

    counts, _ = np.histogram(np.sort(times), bins=t)
    psth = (counts > 0).astype(float)          # PSTH_.m binarisation

    k = kernel_psth(psth, g)
    k[np.isnan(k)] = 0.0

    n_cells = len(gids)
    fr_neuron = psth.sum() / end_time / n_cells   # spk/s per neuron
    mk = k.mean()
    scale = fr_neuron / mk if (mk != 0.0 and np.isfinite(mk)) else 0.0
    if not np.isfinite(scale):
        scale = 0.0
    return k * scale, n_cells, fr_neuron


# --------------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--data', nargs='*', default=[],
                    help='NetPyNE outputs (.json or .mat) to compare against '
                         'the baseline; each gets its own dash pattern')
    ap.add_argument('--trials', nargs='*', default=[],
                    help='repeated-trial NetPyNE outputs of ONE condition '
                         '(e.g. runs of tests_cn.py with different seeds); '
                         'plotted as a single mean trace +/- SD shaded band, '
                         'instead of one line per file')
    ap.add_argument('--trials-label', default=None,
                    help='legend label for the trial-average line '
                         '(default: "trial avg (n=N)")')
    ap.add_argument('--trials2', nargs='*', default=[],
                    help='repeated-trial NetPyNE outputs of a SECOND condition '
                         '(e.g. the other model), plotted as its own mean +/- SD '
                         'shaded band overlaid on the first --trials group')
    ap.add_argument('--trials2-label', default=None,
                    help='legend label for the second trial-average line '
                         '(default: "trial avg 2 (n=N)")')
    ap.add_argument('--baseline', default=BASELINE,
                    help='reference run, always plotted solid (default: %s)'
                         % BASELINE)
    ap.add_argument('--no-baseline', action='store_true',
                    help='plot only the files given to --data')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='legend labels for the comparison runs, in the same '
                         'order as --data (default: file stems)')
    ap.add_argument('--baseline-label', default=None,
                    help='legend label for the baseline '
                         "(default: its file stem + ' (Medlock)')")
    ap.add_argument('--cmap', default='SDH-colormap.mat',
                    help='SDH-colormap.mat containing Cellcmap')
    ap.add_argument('--sigma', type=float, default=KWID,
                    help='Gaussian kernel sigma in ms (default 100)')
    ap.add_argument('--end-time', type=float, default=END_TIME,
                    help='simulated duration in s, used to normalise rates '
                         '(default 5)')
    ap.add_argument('--groups', nargs='+', default=None,
                    help='space-separated group labels to plot, '
                         'e.g. AB VGLUT3 PV PKC (default: all)')
    ap.add_argument('--single-panel', action='store_true',
                    help='overlay all selected groups on one axes')
    ap.add_argument('--autoscale', action='store_true',
                    help='let matplotlib pick y-limits instead of the .m values')
    ap.add_argument('--out', default='fig2A_right.png')
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.data):
        print('ERROR: --labels has %d entries but --data has %d'
              % (len(args.labels), len(args.data)), file=sys.stderr)
        sys.exit(1)

    # ---- assemble the run list: baseline first, then comparisons ----------
    plan = []                                   # (path, label, linestyle)
    if not args.no_baseline:
        if not os.path.exists(args.baseline):
            print('ERROR: baseline not found: %s\n'
                  'Pass --baseline PATH to point at it, or --no-baseline to '
                  'plot without it.' % args.baseline, file=sys.stderr)
            sys.exit(1)
        plan.append((args.baseline,
                     args.baseline_label
                     or '%s (baseline)' % run_label(args.baseline),
                     BASELINE_STYLE))

    for i, path in enumerate(args.data):
        if not args.no_baseline and \
                os.path.abspath(path) == os.path.abspath(args.baseline):
            print('WARNING: %s is the baseline, not plotting it twice' % path)
            continue
        plan.append((path,
                     args.labels[i] if args.labels else run_label(path),
                     COMPARE_STYLES[i % len(COMPARE_STYLES)]))

    if not plan and not args.trials and not args.trials2:
        print('ERROR: no runs to plot -- pass files to --data, --trials, '
              'or --trials2', file=sys.stderr)
        sys.exit(1)

    # ---- load every run ---------------------------------------------------
    runs = []
    for path, label, style in plan:
        try:
            spkt, spkid, t, gid_map, labels = load_sim(path)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        span = (t[-1] - t[0]) / 1000.0          # ms -> s
        if abs(span - args.end_time) > 1e-6:
            print('WARNING: %s spans %.3f s but rates are normalised by '
                  '%.3f s (see --end-time)' % (path, span, args.end_time))

        have = set(labels)
        missing_pops = [p for p in POP_ORDER if p not in have]
        if missing_pops:
            print('WARNING: population(s) absent from net.pops in %s: %s'
                  % (path, ', '.join(missing_pops)))

        runs.append({
            'path':  path,
            'label': label,
            'spkt':  spkt,
            'spkid': spkid,
            't':     t,
            'gids':  gid_map,
            'have':  have,
            'style': style,
        })

    # ---- load every trial file (repeated trials of ONE condition) ---------
    trial_runs = []
    trials_style  = TRIALS_STYLES[len(args.data) % len(TRIALS_STYLES)]
    trials_hatch  = TRIALS_HATCHES[len(args.data) % len(TRIALS_HATCHES)]
    trials_label = args.trials_label or 'trial avg (n=%d)' % len(args.trials)
    for path in args.trials:
        try:
            spkt, spkid, t, gid_map, labels = load_sim(path)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        span = (t[-1] - t[0]) / 1000.0          # ms -> s
        if abs(span - args.end_time) > 1e-6:
            print('WARNING: %s spans %.3f s but rates are normalised by '
                  '%.3f s (see --end-time)' % (path, span, args.end_time))

        have = set(labels)
        missing_pops = [p for p in POP_ORDER if p not in have]
        if missing_pops:
            print('WARNING: population(s) absent from net.pops in %s: %s'
                  % (path, ', '.join(missing_pops)))

        trial_runs.append({
            'path':  path,
            'spkt':  spkt,
            'spkid': spkid,
            't':     t,
            'gids':  gid_map,
            'have':  have,
        })

    if trial_runs:
        t0 = trial_runs[0]['t']
        for tr in trial_runs[1:]:
            if len(tr['t']) != len(t0) or not np.allclose(tr['t'], t0):
                print('WARNING: %s has a different time axis than %s -- '
                      'trial traces will be truncated to the shortest '
                      'common length before averaging'
                      % (tr['path'], trial_runs[0]['path']))
                break

    # ---- load every trial2 file (repeated trials of a SECOND condition) ---
    trial_runs2 = []
    _trials2_idx = (len(args.data) + (1 if args.trials else 0))
    trials2_style = TRIALS_STYLES[_trials2_idx % len(TRIALS_STYLES)]
    trials2_hatch = TRIALS_HATCHES[_trials2_idx % len(TRIALS_HATCHES)]
    trials2_label = args.trials2_label or 'trial avg 2 (n=%d)' % len(args.trials2)
    for path in args.trials2:
        try:
            spkt, spkid, t, gid_map, labels = load_sim(path)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        span = (t[-1] - t[0]) / 1000.0          # ms -> s
        if abs(span - args.end_time) > 1e-6:
            print('WARNING: %s spans %.3f s but rates are normalised by '
                  '%.3f s (see --end-time)' % (path, span, args.end_time))

        have = set(labels)
        missing_pops = [p for p in POP_ORDER if p not in have]
        if missing_pops:
            print('WARNING: population(s) absent from net.pops in %s: %s'
                  % (path, ', '.join(missing_pops)))

        trial_runs2.append({
            'path':  path,
            'spkt':  spkt,
            'spkid': spkid,
            't':     t,
            'gids':  gid_map,
            'have':  have,
        })

    if trial_runs2:
        t0 = trial_runs2[0]['t']
        for tr in trial_runs2[1:]:
            if len(tr['t']) != len(t0) or not np.allclose(tr['t'], t0):
                print('WARNING: %s has a different time axis than %s -- '
                      'trial traces will be truncated to the shortest '
                      'common length before averaging'
                      % (tr['path'], trial_runs2[0]['path']))
                break

    # ---- select groups ----------------------------------------------------
    wanted = None
    if args.groups:
        wanted = [s.strip() for s in args.groups if s.strip()]
        known = {name for name, _, _ in GROUPS}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            print('ERROR: unknown group label(s): %s' % ', '.join(unknown),
                  file=sys.stderr)
            print('Valid labels: %s'
                  % ', '.join(name for name, _, _ in GROUPS), file=sys.stderr)
            sys.exit(1)

    # A group survives if at least one run has all of its populations; runs
    # that are missing them are skipped individually at plot time.
    active_groups, skipped_groups = [], []
    for name, pops, sub in GROUPS:
        if wanted is not None and name not in wanted:
            continue
        if any(resolve_pops(name, pops, r['have']) for r in runs) or \
                any(resolve_pops(name, pops, r['have']) for r in trial_runs) or \
                any(resolve_pops(name, pops, r['have']) for r in trial_runs2):
            active_groups.append((name, pops, sub))
        else:
            skipped_groups.append(name)
    if skipped_groups:
        print('WARNING: skipping trace(s) with missing data in every run: %s'
              % ', '.join(skipped_groups))
    if not active_groups:
        print('ERROR: nothing to plot', file=sys.stderr)
        sys.exit(1)

    # ---- colours ----------------------------------------------------------
    if os.path.exists(args.cmap):
        from scipy.io import loadmat
        cellcmap = np.asarray(loadmat(args.cmap)['Cellcmap'], float)
    else:
        cellcmap = plt.cm.tab20(np.linspace(0, 1, len(POP_ORDER)))[:, :3]
        print('WARNING: %s not found, using placeholder colours' % args.cmap)

    def group_colour(pops):
        '''Cellcmap row of the group's last constituent pop -- matches the
        unqNewClr construction in FiringRateHist.m.'''
        key = pops[-1]
        idx = POP_ORDER.index(COLOR_ALIAS.get(key, key))
        return cellcmap[idx % len(cellcmap)]

    # ---- layout: keep only panels that have traces ------------------------
    if args.single_panel:
        used_subs = [0]
        sub_to_ax = {sub: 0 for _, _, sub in active_groups}
    else:
        used_subs = sorted({sub for _, _, sub in active_groups})
        sub_to_ax = {sub: i for i, sub in enumerate(used_subs)}

    g = gaussian_kernel(args.sigma)
    fig, axes = plt.subplots(len(used_subs), 1,
                             figsize=(6.5, max(2.6 * len(used_subs), 3.0)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    panel_groups = {i: [] for i in range(len(used_subs))}   # legend bookkeeping
    xmax = 0.0

    print('\n%-14s %-9s %5s %12s %12s'
          % ('run', 'group', 'n', 'mean spk/s', 'peak spk/s'))
    for name, pops, sub in active_groups:
        colour = group_colour(pops)
        ax_i = sub_to_ax[sub]
        if name not in [n for n, _ in panel_groups[ax_i]]:
            panel_groups[ax_i].append((name, colour))

        for r in runs:
            use_pops = resolve_pops(name, pops, r['have'])
            if use_pops is None:
                print('%-14s %-9s %5s %12s %12s'
                      % (r['label'], name, '-', 'absent', '-'))
                continue
            gids = np.concatenate([r['gids'][p] for p in use_pops])
            trace, n_cells, mean_rate = compute_trace(
                r['spkt'], r['spkid'], gids, r['t'], g, args.end_time)
            print('%-14s %-9s %5d %12.3f %12.3f'
                  % (r['label'], name, n_cells, mean_rate, trace.max()))

            tt = r['t'][:len(trace)]
            xmax = max(xmax, tt[-1])
            axes[ax_i].plot(tt, trace, color=colour, lw=1.5,
                            ls=r['style'], label='_nolegend_')

        if trial_runs:
            traces, n_cells, mean_rates = [], None, []
            for tr in trial_runs:
                use_pops = resolve_pops(name, pops, tr['have'])
                if use_pops is None:
                    print('%-14s %-9s %5s %12s %12s'
                          % (os.path.basename(tr['path']), name, '-',
                             'absent', '-'))
                    continue
                gids = np.concatenate([tr['gids'][p] for p in use_pops])
                trace, n_cells, mean_rate = compute_trace(
                    tr['spkt'], tr['spkid'], gids, tr['t'], g, args.end_time)
                traces.append(trace)
                mean_rates.append(mean_rate)

            if traces:
                min_len = min(len(x) for x in traces)
                arr = np.stack([x[:min_len] for x in traces])
                mean_trace = arr.mean(axis=0)
                std_trace = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 \
                    else np.zeros(min_len)
                print('%-14s %-9s %5d %12.3f %12.3f'
                      % (trials_label, name, n_cells, np.mean(mean_rates),
                         mean_trace.max()))

                tt = trial_runs[0]['t'][:min_len]
                xmax = max(xmax, tt[-1])
                axes[ax_i].fill_between(tt, mean_trace - std_trace,
                                        mean_trace + std_trace,
                                        facecolor=colour, edgecolor=colour,
                                        hatch=trials_hatch,
                                        alpha=0.22, lw=0, zorder=1)
                axes[ax_i].plot(tt, mean_trace, color=colour, lw=1.8,
                                ls=trials_style, label='_nolegend_')

        if trial_runs2:
            traces, n_cells, mean_rates = [], None, []
            for tr in trial_runs2:
                use_pops = resolve_pops(name, pops, tr['have'])
                if use_pops is None:
                    print('%-14s %-9s %5s %12s %12s'
                          % (os.path.basename(tr['path']), name, '-',
                             'absent', '-'))
                    continue
                gids = np.concatenate([tr['gids'][p] for p in use_pops])
                trace, n_cells, mean_rate = compute_trace(
                    tr['spkt'], tr['spkid'], gids, tr['t'], g, args.end_time)
                traces.append(trace)
                mean_rates.append(mean_rate)

            if traces:
                min_len = min(len(x) for x in traces)
                arr = np.stack([x[:min_len] for x in traces])
                mean_trace = arr.mean(axis=0)
                std_trace = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 \
                    else np.zeros(min_len)
                print('%-14s %-9s %5d %12.3f %12.3f'
                      % (trials2_label, name, n_cells, np.mean(mean_rates),
                         mean_trace.max()))

                tt = trial_runs2[0]['t'][:min_len]
                xmax = max(xmax, tt[-1])
                axes[ax_i].fill_between(tt, mean_trace - std_trace,
                                        mean_trace + std_trace,
                                        facecolor=colour, edgecolor=colour,
                                        hatch=trials2_hatch,
                                        alpha=0.22, lw=0, zorder=2)
                axes[ax_i].plot(tt, mean_trace, color=colour, lw=1.8,
                                ls=trials2_style, label='_nolegend_')
    print()

    # ---- axis cosmetics ---------------------------------------------------
    for ax_i, sub in enumerate(used_subs):
        ax = axes[ax_i]
        ax.set_xlim(0, xmax)
        if not args.autoscale and not args.single_panel:
            ax.set_ylim(*SUBPLOT_YLIM[sub])
            ax.set_yticks(SUBPLOT_YTICK[sub])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', labelsize=11)
        ax.grid(True, color='0.85', lw=0.5, zorder=0)
        ax.set_axisbelow(True)

        # colour legend: one entry per group, drawn solid
        handles = [Line2D([], [], color=c, lw=1.5, label=n)
                   for n, c in panel_groups[ax_i]]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper right')
        if not args.single_panel:
            ax.set_title(SUBPLOT_TITLES[sub], fontsize=10, loc='left')

    # style legend: one entry per run (plus one for the trial average),
    # drawn in neutral grey
    n_style_entries = len(runs) + (1 if trial_runs else 0) + (1 if trial_runs2 else 0)
    if n_style_entries > 1:
        run_handles = [Line2D([], [], color='0.25', lw=1.5, ls=r['style'],
                              label=r['label']) for r in runs]
        if trial_runs:
            run_handles.append(Line2D([], [], color='0.25', lw=1.5,
                                      ls=trials_style, label=trials_label))
        if trial_runs2:
            run_handles.append(Line2D([], [], color='0.25', lw=1.5,
                                      ls=trials2_style, label=trials2_label))
        fig.legend(handles=run_handles, loc='upper center',
                   ncol=min(n_style_entries, 4), frameon=False, fontsize=9)
        top = 0.88
    elif runs:
        stem = run_label(runs[0]['path'])
        n_present, n_total = len(runs[0]['have']), len(POP_ORDER)
        tag = '' if n_present == n_total else \
            '  (PARTIAL: %d/%d populations)' % (n_present, n_total)
        axes[0].set_title('%s%s' % (stem, tag), fontsize=10, loc='left')
        top = 0.90
    elif trial_runs:
        axes[0].set_title(trials_label, fontsize=10, loc='left')
        top = 0.90
    else:
        axes[0].set_title(trials2_label, fontsize=10, loc='left')
        top = 0.90

    xticks = [x for x in range(0, int(xmax) + 1, 1000)]
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([x // 1000 for x in xticks])
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.text(0.02, 0.5, 'Firing rate (spk/s)', rotation=90, va='center',
             fontsize=12)
    fig.subplots_adjust(left=0.15, right=0.97, top=top, bottom=0.12, hspace=0.3)
    fig.savefig(args.out, dpi=300)
    print('saved %s' % args.out)


if __name__ == '__main__':
    main()