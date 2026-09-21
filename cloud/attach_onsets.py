"""Write onset timestamps into result files that were produced without them.

The prepared sessions gained an `onsets` key after several campaigns had already
run, and the training script falls back to zeros when the key is absent. Runs
made before that change therefore carry per-window scores whose timestamps are
all zero, which makes every window a neighbour of every other and collapses any
smoothing to a constant.

The windows can be recovered rather than recomputed. The test loader does not
shuffle, so a result's per-window arrays are in exactly the order
`load_subjects` produced for the held-out subject: sessions in sorted filename
order, windows in trial order within each session. Reading the timestamps back
out of the prepared dataset and checking that the label sequence matches element
for element is enough to attach them safely -- a mismatch means the dataset on
disk is not the one the run used, and the file is left alone.

    python cloud/attach_onsets.py --data datasets/downstream/sadt_ts \
        --results downstream/results_sadt --strategy layerwise_pw
"""

import argparse
import collections
import glob
import json
import os
import sys

import torch

SESSION_TIME_STRIDE_S = 86400.0


def subject_onsets(data_dir):
    """Per subject, the onsets and labels in the order training concatenates them."""
    per_subject = {}
    session_index = collections.defaultdict(int)
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".pt") or name.startswith("."):
            continue
        blob = torch.load(os.path.join(data_dir, name), map_location="cpu")
        sub = int(blob["subject"])
        if "onsets" not in blob:
            raise SystemExit(f"{name} has no onsets; re-run prepare_SADT.py first")
        o = blob["onsets"].double() + SESSION_TIME_STRIDE_S * session_index[sub]
        session_index[sub] += 1
        y = blob["y"]
        if sub in per_subject:
            po, py = per_subject[sub]
            per_subject[sub] = (torch.cat([po, o]), torch.cat([py, y]))
        else:
            per_subject[sub] = (o, y)
    return per_subject


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "datasets", "downstream", "sadt_ts"))
    ap.add_argument("--results", default=os.path.join(here, "..", "downstream", "results_sadt"))
    ap.add_argument("--strategy", required=True, help="result file prefix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    per_subject = subject_onsets(args.data)
    files = sorted(f for f in glob.glob(os.path.join(args.results, f"{args.strategy}_fold*.json"))
                   if not f.endswith(".devrun.json"))
    if not files:
        print(f"no results matching {args.strategy}", file=sys.stderr)
        return 1

    patched = skipped = mismatched = 0
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        pw = r.get("per_window")
        if pw is None:
            skipped += 1
            continue
        if any(t != 0.0 for t in pw["onset"]):
            skipped += 1
            continue
        sub = int(r["test_subject"])
        onset, label = per_subject[sub]
        # The stored run dropped intermediate trials from the test set, so
        # compare against the strict subset the same way the loader built it.
        keep = label >= 0
        onset, label = onset[keep], label[keep]
        stored = torch.tensor(pw["label"])
        if len(stored) != len(label) or not torch.equal(stored, label.to(stored.dtype)):
            print(f"subject {sub}: label sequence differs "
                  f"({len(stored)} stored vs {len(label)} on disk) -- left alone")
            mismatched += 1
            continue
        pw["onset"] = [round(float(t), 3) for t in onset]
        patched += 1
        if not args.dry_run:
            with open(f, "w") as fh:
                json.dump(r, fh)

    verb = "would patch" if args.dry_run else "patched"
    print(f"{verb} {patched} files, skipped {skipped}, {mismatched} mismatched")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
