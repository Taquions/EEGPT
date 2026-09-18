"""Summarise a Leave-One-Subject-Out campaign, and compare two of them.

Per-subject balanced accuracy is the unit of analysis: it is what the paired
Wilcoxon test consumes, so the table is printed one row per subject rather than
as a single average. A subject that is hard drags every strategy down equally,
which is exactly why the comparison is paired.

    python summarise_folds.py --strategy linear
    python summarise_folds.py --strategy lora --against linear
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(results_dir, strategy):
    """Return {test_subject: record} for one strategy."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{strategy}_fold*.json"))):
        if path.endswith(".devrun.json"):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        out[int(rec["test_subject"])] = rec
    return out


def table(records, strategy):
    subs = sorted(records)
    bac = np.array([records[s]["balanced_accuracy"] for s in subs])
    auc = np.array([records[s]["roc_auc"] for s in subs])

    print(f"=== {strategy}: {len(subs)} folds ===")
    print(" sub    BAC   AUROC    acc   n_test  drowsy   min  trainable")
    for s in subs:
        r = records[s]
        print(f"{s:4d}  {r['balanced_accuracy']:.4f}  {r['roc_auc']:.4f}  "
              f"{r['accuracy']:.4f}  {r['n_test']:6d}  {r['n_test_drowsy']:6d}  "
              f"{r['train_seconds'] / 60:5.1f}  "
              f"{r['trainable_head'] + r['trainable_encoder']:9d}")

    print(f"\nBAC   mean {bac.mean():.4f}  sd {bac.std(ddof=1):.4f}  "
          f"median {np.median(bac):.4f}  min {bac.min():.4f}  max {bac.max():.4f}")
    print(f"AUROC mean {auc.mean():.4f}  sd {auc.std(ddof=1):.4f}")
    below = [s for s in subs if records[s]["balanced_accuracy"] <= 0.5]
    if below:
        print(f"at or below chance: {below}")
    total_min = sum(records[s]["train_seconds"] for s in subs) / 60
    print(f"compute: {total_min:.0f} min over {len(subs)} folds "
          f"({total_min / max(len(subs), 1):.1f} min per fold)")
    return bac


def compare(a_records, a_name, b_records, b_name):
    """Paired Wilcoxon on the subjects both campaigns covered."""
    from scipy.stats import wilcoxon

    shared = sorted(set(a_records) & set(b_records))
    if len(shared) < 6:
        print(f"\nonly {len(shared)} shared subjects -- too few for a "
              f"meaningful signed-rank test")
        return
    a = np.array([a_records[s]["balanced_accuracy"] for s in shared])
    b = np.array([b_records[s]["balanced_accuracy"] for s in shared])
    diff = a - b

    print(f"\n=== {a_name} vs {b_name}, paired over {len(shared)} subjects ===")
    print(f"mean BAC {a.mean():.4f} vs {b.mean():.4f}  "
          f"(difference {diff.mean() * 100:+.2f} pp)")
    print(f"{a_name} wins on {int((diff > 0).sum())} subjects, "
          f"loses on {int((diff < 0).sum())}, ties on {int((diff == 0).sum())}")

    if np.allclose(diff, 0):
        print("identical on every subject; no test to run")
        return
    stat, p = wilcoxon(a, b)
    print(f"Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"at alpha=0.05 the difference is {verdict}")
    # The work sets 1.5 pp as the smallest difference worth acting on, so
    # significance alone is not the bar.
    if diff.mean() * 100 >= 1.5:
        print("and it clears the 1.5 pp practical threshold")
    else:
        print("but it does not clear the 1.5 pp practical threshold")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results",
                    default=os.path.join(here, "..", "downstream", "results_sadt"))
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--against", help="second strategy to compare against")
    args = ap.parse_args()

    records = load(args.results, args.strategy)
    if not records:
        print(f"no {args.strategy} results under {args.results}", file=sys.stderr)
        return 1
    table(records, args.strategy)

    if args.against:
        other = load(args.results, args.against)
        if not other:
            print(f"\nno {args.against} results to compare against", file=sys.stderr)
            return 1
        table(other, args.against)
        compare(records, args.strategy, other, args.against)
    return 0


if __name__ == "__main__":
    sys.exit(main())
