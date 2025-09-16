import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_gamma_peaks(t, gamma):
    gmean = np.nanmean(gamma)
    gstd = np.nanstd(gamma)
    thresh = gmean + gstd
    peaks = [t[i] for i in range(1, len(gamma) - 1)
             if gamma[i] > thresh and gamma[i] > gamma[i-1] and gamma[i] > gamma[i+1]]
    return np.array(peaks)


def build_schedule_times(cfg):
    se = cfg.get('exploration', {}).get('spacing_effect', {})
    if not se or not se.get('enabled', False):
        return np.array([])
    from spacing_schedule import build_spacing_schedule
    sched = build_spacing_schedule(cfg['system']['T'], se.get('start_interval', 2.0), se.get('growth', 1.5), se.get('num_blocks', 8), se.get('order', ['gamma','G']))
    return np.array(sched.get('gamma_peaks', []))


def find_local_peak_around(t, gamma, t0, dt, window_steps=2):
    # map t0 to index
    if len(t) == 0:
        return None
    i0 = int(round(t0 / max(dt, 1e-9)))
    i0 = max(1, min(len(t)-2, i0))
    w = int(max(1, window_steps))
    lo = max(1, i0 - w)
    hi = min(len(t)-2, i0 + w)
    if hi <= lo:
        return None
    idx = lo + np.argmax(gamma[lo:hi+1])
    return float(t[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', default='spacing_report')
    ap.add_argument('--window_steps', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    t = df['t'].values
    dt = np.median(np.diff(t)) if len(t)>1 else 0.1
    gamma = pd.to_numeric(df.get('gamma', pd.Series(np.nan, index=df.index)), errors='coerce').values

    # global peaks (for context)
    peaks_global = detect_gamma_peaks(t, gamma)

    cfg = json.load(open(args.cfg))
    sched_times = build_schedule_times(cfg)

    matches = []
    for st in sched_times:
        lp = find_local_peak_around(t, gamma, st, dt, window_steps=args.window_steps)
        if lp is not None:
            matches.append({'scheduled': float(st), 'detected': lp, 'delta': float(lp - st)})

    # plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, gamma, lw=0.8, label='gamma(t)')
    if len(peaks_global):
        ax.scatter(peaks_global, np.interp(peaks_global, t, gamma), c='tab:orange', s=10, label='global peaks')
    if len(sched_times):
        ax.vlines(sched_times, ymin=np.nanmin(gamma), ymax=np.nanmax(gamma), colors='tab:green', alpha=0.3, label='scheduled')
    if matches:
        md = np.array([m['detected'] for m in matches])
        ax.scatter(md, np.interp(md, t, gamma), c='tab:blue', s=12, label='local around scheduled')
    ax.set_title('Gamma spacing: scheduled vs detected (local window)')
    ax.set_xlabel('t')
    ax.legend(loc='upper right')
    fig.tight_layout()
    png = os.path.join(args.out, f"spacing_overlay_{os.path.splitext(os.path.basename(args.csv))[0]}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    summary = {
        'csv': args.csv,
        'cfg': args.cfg,
        'num_detected_global': int(len(peaks_global)),
        'num_scheduled': int(len(sched_times)),
        'matches': matches[:20]
    }
    jpath = os.path.join(args.out, f"spacing_summary_{os.path.splitext(os.path.basename(args.csv))[0]}.json")
    json.dump(summary, open(jpath, 'w'), indent=2)
    print({'png': png, 'summary': jpath})


if __name__ == '__main__':
    main() 