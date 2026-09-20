"""Smooth per-window predictions over time and re-score.

Drowsiness is a slow state: the labelling rule itself averages reaction time over
90 seconds to decide it. A single three-second window is therefore a noisy sample
of the thing being predicted -- it can catch a momentary arousal inside a drowsy
stretch, or a lapse inside an alert one. Part of the ceiling a per-window metric
reports is that mismatch of timescale rather than the model.

Averaging each window's score with those of its neighbours in time uses the
state's slowness where it actually helps, at the decision rather than at the
input. It is also what the application wants: nobody needs a verdict on an
operator every three seconds, they need a state estimate over minutes.

This runs offline on the result JSONs -- no GPU, no retraining.

    python smooth_predictions.py --strategy layerwise_best
    python smooth_predictions.py --strategy layerwise_best --windows 0 30 60 120
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def balanced_accuracy(label, score, threshold=0.5):
    pred = score >= threshold
    pos, neg = label == 1, label == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (pred[pos].mean() + (~pred[neg]).mean())


def roc_auc(label, score):
    """Rank-based AUROC, so no sklearn import is needed for one number."""
    pos, neg = label == 1, label == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks within ties, or equal scores would bias the statistic.
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def smooth(onset, score, half_width_s):
    """Average each window's score with those within +/- half_width seconds.

    Centred rather than backward-looking: this is an offline state estimate over
    a recording, not a causal detector. A deployment that has to decide live
    would use the backward half, and score lower.
    """
    if half_width_s <= 0:
        return score.copy()
    order = np.argsort(onset)
    t, s = onset[order], score[order]
    left = np.searchsorted(t, t - half_width_s, side="left")
    right = np.searchsorted(t, t + half_width_s, side="right")
    cum = np.concatenate([[0.0], np.cumsum(s)])
    out = (cum[right] - cum[left]) / (right - left)
    result = np.empty_like(score)
    result[order] = out
    return result


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results",
                    default=os.path.join(here, "..", "downstream", "results_sadt"))
    ap.add_argument("--strategy", required=True,
                    help="result file prefix, e.g. layerwise_best")
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[0, 15, 30, 60, 120, 300],
                    help="half-widths in seconds; 0 is the unsmoothed baseline")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.results, f"{args.strategy}_fold*.json"))
             if not f.endswith(".devrun.json")]
    if not files:
        print(f"no results matching {args.strategy} under {args.results}", file=sys.stderr)
        return 1

    records = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        if "per_window" not in r:
            continue
        pw = r["per_window"]
        records.append((r["test_subject"], np.asarray(pw["onset"], dtype=np.float64),
                        np.asarray(pw["label"], dtype=int),
                        np.asarray(pw["score"], dtype=np.float64)))
    if not records:
        print(f"{len(files)} results found, none carrying per-window scores. "
              f"They predate the change that stores them.", file=sys.stderr)
        return 1

    print(f"{args.strategy}: {len(records)} subjects with per-window scores")
    print(f"{'half-width':>11}  {'BAC':>7}  {'AUROC':>7}  {'vs 0 s':>8}")
    baseline = None
    for hw in args.windows:
        bacs, aucs = [], []
        for _sub, onset, label, score in records:
            sm = smooth(onset, score, hw)
            bacs.append(balanced_accuracy(label, sm))
            aucs.append(roc_auc(label, sm))
        bac, auc = np.nanmean(bacs), np.nanmean(aucs)
        if baseline is None:
            baseline = bac
        print(f"{hw:>9.0f} s  {bac:>7.4f}  {auc:>7.4f}  {(bac - baseline) * 100:>+7.2f} pp")

    print("\nHalf-widths are one-sided, so 60 s averages over a two-minute span.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
