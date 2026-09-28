"""Drowsiness and reaction time along the session.

The advisor asked for the smoothed P(drowsy) curve across the session together
with the median reaction time in the first against the last half hour. Two
layers are available without any new training: the reaction times, which come
straight from the prepared dataset, and the model's scores for the windows that
carry a strict label. The intermediate windows are not scored -- evaluation
drops them -- so the curve has gaps wherever the labelling rule abstained.

    python cloud/session_trajectory.py --out <dir>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_predictions import smooth

SESSION_STRIDE = 86400.0


def rt_profile(data_dir, n_bins=14, span_min=70.0):
    """Median reaction time against position in the session, pooled over sessions.

    Each session is placed on a common axis by its own start, and the median is
    taken across sessions per bin, so a long session does not outweigh a short
    one.
    """
    edges = np.linspace(0, span_min * 60, n_bins + 1)
    per_session = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
        b = torch.load(f)
        o = b["onsets"].numpy()
        rt = b["rt"].numpy()
        t = o - o.min()
        row = [np.median(rt[(t >= a) & (t < c)]) if ((t >= a) & (t < c)).sum() >= 3 else np.nan
               for a, c in zip(edges[:-1], edges[1:])]
        per_session.append(row)
    arr = np.array(per_session, dtype=float)
    centres = (edges[:-1] + edges[1:]) / 2 / 60
    return centres, np.nanmedian(arr, axis=0), np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0)


def score_profile(results, tag, n_bins=14, span_min=70.0):
    """Mean smoothed P(drowsy) against position in the session."""
    edges = np.linspace(0, span_min * 60, n_bins + 1)
    rows, label_rows = [], []
    for f in glob.glob(os.path.join(results, f"{tag}_fold*.json")):
        if f.endswith(".devrun.json"):
            continue
        r = json.load(open(f))
        pw = r.get("per_window")
        if not pw:
            continue
        o = np.asarray(pw["onset"], float)
        s = smooth(o, np.asarray(pw["score"], float), 60.0)
        l = np.asarray(pw["label"], int)
        # split back into sessions, which were pushed a day apart on the axis
        for sess in np.unique(np.floor(o / SESSION_STRIDE)):
            m = np.floor(o / SESSION_STRIDE) == sess
            t = o[m] - o[m].min()
            rows.append([s[m][(t >= a) & (t < c)].mean() if ((t >= a) & (t < c)).sum() >= 3 else np.nan
                         for a, c in zip(edges[:-1], edges[1:])])
            label_rows.append([l[m][(t >= a) & (t < c)].mean() if ((t >= a) & (t < c)).sum() >= 3 else np.nan
                               for a, c in zip(edges[:-1], edges[1:])])
    centres = (edges[:-1] + edges[1:]) / 2 / 60
    return centres, np.nanmean(np.array(rows, float), axis=0), np.nanmean(np.array(label_rows, float), axis=0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "..", "datasets", "downstream", "sadt_ts"))
    ap.add_argument("--results", default=os.path.join(here, "..", "downstream", "results_sadt"))
    ap.add_argument("--tag", default="linear_attn25")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tc, rt_med, rt_lo, rt_hi = rt_profile(args.data)
    sc, score, prev = score_profile(args.results, args.tag)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    ax1.plot(tc, rt_med, "o-", color="#444444", label="TR mediano")
    ax1.fill_between(tc, rt_lo, rt_hi, color="#444444", alpha=0.15, label="quartis")
    ax1.set_ylabel("Tempo de reação (s)")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.25)

    ax2.plot(sc, score, "o-", color="#1f77b4", label="P(sonolento) predita, suavizada")
    ax2.plot(sc, prev, "s--", color="#d62728", label="Proporção rotulada sonolenta")
    ax2.set_xlabel("Tempo desde o início da sessão (min)")
    ax2.set_ylabel("Probabilidade")
    ax2.legend(fontsize=7, frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "session-trajectory.pdf"))
    plt.close(fig)

    # os números que a pergunta pede, em texto
    print(f"{'min':>6} {'TR med':>8} {'P(sono)':>9} {'prev':>7}")
    for a, b, c, d in zip(tc, rt_med, score, prev):
        print(f"{a:>6.1f} {b:>8.3f} {c:>9.3f} {d:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
