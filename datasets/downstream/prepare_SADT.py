"""Preprocess the raw SADT recordings into windows labelled alert/drowsy.

Input is what `download_SADT.py` fetches: 62 EEGLAB `.set` sessions from the
Sustained-Attention Driving Task (Cao et al., 2019), 27 subjects, 500 Hz.

Output is one `.pt` per session holding a tensor of windows plus their labels,
which the training scripts group by subject for Leave-One-Subject-Out.

Labelling follows Wei et al., "Toward Drowsiness Detection Using Non-Hair-Bearing
EEG-Based BCIs", IEEE TNSRE 26(2):400-406, 2018 -- the de-facto standard for this
dataset, restated verbatim by Cui et al. (Methods 2021, TNNLS 2022) and Li et al.
(2023). Note that the rule is *not* in the data descriptor: Cao et al. 2019 define
no thresholds at all, so cite Wei et al. for it.

Pipeline
--------
1. Keep the 30 scalp channels: drop `vehicle position` (behavioural, but see
   step 3) and the mastoid references A1/A2 -- which sit at indices 23 and 29,
   interleaved rather than trailing, so positional slicing takes the wrong set.
   Rename the old 10-20 labels T3/T4/T5/T6 to their 10-10 equivalents
   T7/T8/P7/P8; EEGPT's channel vocabulary carries only the latter.
2. Convert to microvolts and low-pass at 38 Hz, then resample 500 -> 256 Hz.
3. Reject trials whose vehicle trajectory was not flat before the deviation
   onset, per the tutorial shipped with the dataset. See `trial_is_clean`.
4. Recover reaction time per trial: 251/252 mark the lane departure, 253 the
   onset of the participant's response, 254/255 its offset.
5. Label each trial against the session's alert baseline (see `label_trials`).
6. Cut a window ending at the deviation onset. The window is strictly *before*
   the event: EEG after the onset contains the motor response itself, so
   classifying drowsiness from it would be circular. Every published work on
   this dataset does the same, with 3 s the settled convention.
7. Euclidean Alignment per session, then channel-wise z-score per window.

Usage
-----
    python prepare_SADT.py                      # defaults, all sessions
    python prepare_SADT.py --jobs 4             # parallel over sessions
    python prepare_SADT.py --window-s 4.0       # EEGPT's pretraining geometry
"""

import argparse
import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

warnings.filterwarnings("ignore")

# --- acquisition constants ---------------------------------------------------

SFREQ_IN = 500.0
SFREQ_OUT = 256.0
LOW_PASS_HZ = 38.0

DEV_ONSET = ("251", "252")      # lane departure, left / right
RESP_ONSET = "253"              # participant starts correcting
RESP_OFFSET = ("254", "255")    # correction finished; 255 = uncorrected trial,
                                # undocumented in the paper, present only in
                                # subjects 54 and 55.

TRAJECTORY_CHANNEL = "VEHICLE POSITION"
# SADT predates the 10-10 naming; EEGPT's vocabulary does not know T3/T4/T5/T6.
RENAME_10_20 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
# A1/A2 are the mastoid references. Dropping them rather than re-referencing to
# them follows Cui et al. and Li et al.; Wu et al. (arXiv:1809.00929) instead
# re-reference to averaged earlobes. Both are citable, this one is the majority.
DROP_CHANNELS = ("A1", "A2", TRAJECTORY_CHANNEL)

# --- labelling constants (Wei et al., 2018) ----------------------------------

BASELINE_PERCENTILE = 5.0       # "alert RT" = 5th percentile of local RTs
ALERT_RATIO = 1.5
DROWSY_RATIO = 2.5
GLOBAL_RT_WINDOW_S = 90.0       # backward-looking, ending at the deviation onset

LABEL_ALERT, LABEL_DROWSY, LABEL_DISCARD = 0, 1, -1


def parse_session(filename):
    """`s01_051017m.set` -> (1, 's01_051017m')."""
    stem = os.path.basename(filename)[: -len(".set")]
    m = re.match(r"s(\d+)_", stem)
    if not m:
        raise ValueError(f"cannot parse a subject id out of {filename!r}")
    return int(m.group(1)), stem


def load_session(path):
    """Read one .set. Returns (eeg_raw, trajectory, trajectory_sfreq).

    The trajectory is pulled out at the acquisition rate and before filtering,
    since it is a quantised position signal rather than EEG and is only used to
    judge whether a trial is behaviourally clean.
    """
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - SFREQ_IN) > 1e-6:
        raise ValueError(f"{path}: expected {SFREQ_IN} Hz, found {raw.info['sfreq']}")

    traj = None
    for ch in raw.ch_names:
        if ch.upper().strip() == TRAJECTORY_CHANNEL:
            # MNE scales every channel as if it were EEG in volts; undo that to
            # recover the original 0-255 lane-position quantisation.
            traj = raw.get_data(picks=[ch])[0] * 1e6
            break

    raw.drop_channels([ch for ch in raw.ch_names if ch.upper().strip() in DROP_CHANNELS])
    raw.rename_channels({ch: RENAME_10_20[ch.upper()] for ch in raw.ch_names
                         if ch.upper() in RENAME_10_20})
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})

    raw.apply_function(lambda d: d * 1e6)              # V -> uV
    raw.filter(l_freq=None, h_freq=LOW_PASS_HZ, verbose="ERROR")
    raw.resample(SFREQ_OUT, verbose="ERROR")

    return raw, traj, SFREQ_IN


def trial_is_clean(traj, sfreq, onset_s, window_s, max_std):
    """Was the car steady in its lane in the interval we are about to cut?

    The tutorial shipped with the dataset rejects trials whose trajectory is not
    flat before the deviation onset. That criterion turns out to be what
    separates the odd cluster of near-zero reaction times from real behaviour:
    on one session those trials have a pre-onset trajectory standard deviation
    of 2.71 against 0.04 for the rest, i.e. the car was already moving and the
    driver was still correcting the previous event. Their event triplets are
    perfectly well formed, so no amount of marker checking finds them -- only
    the behavioural signal does.

    The distribution is strongly bimodal, so the threshold is not delicate.
    """
    if traj is None:
        return True
    end = int(round(onset_s * sfreq))
    start = end - int(round(window_s * sfreq))
    if start < 0 or end > len(traj):
        return False
    return float(traj[start:end].std()) < max_std


def extract_trials(annotations, traj, traj_sfreq, args):
    """Pair each deviation onset with its response and keep the clean trials.

    A valid trial is the triplet the lab's own code assumes throughout:
    251|252 -> 253 -> 254|255. Trials are rejected, and counted separately, for
    three reasons that push the class balance in different directions: a broken
    triplet, a trajectory that was not flat beforehand, and a reaction time
    outside the plausible range.
    """
    desc = list(annotations.description)
    onset = list(annotations.onset)

    trials = {"ok": [], "broken_triplet": 0, "dirty_trajectory": 0,
              "too_fast": 0, "no_response": 0}

    for i, (d, t) in enumerate(zip(desc, onset)):
        if d not in DEV_ONSET:
            continue
        if i + 2 >= len(desc) or desc[i + 1] != RESP_ONSET or desc[i + 2] not in RESP_OFFSET:
            trials["broken_triplet"] += 1
            continue

        rt = onset[i + 1] - t
        if rt > args.max_rt:
            trials["no_response"] += 1
            continue
        if rt < args.min_rt:
            trials["too_fast"] += 1
            continue
        if not trial_is_clean(traj, traj_sfreq, t, args.window_s, args.max_traj_std):
            trials["dirty_trajectory"] += 1
            continue

        trials["ok"].append((t, rt))

    return trials


def label_trials(trials):
    """Label each trial alert/drowsy against the session's own alert baseline.

    Wei et al. (2018): the alert RT is the 5th percentile of the session's local
    reaction times; a trial is alert when both its own RT and the average over
    the preceding 90 s stay below 1.5x that, drowsy when both exceed 2.5x, and
    is left out in between. Requiring both is what stops one quick response
    inside a drowsy stretch from being read as alertness.
    """
    times = np.array([t for t, _ in trials], dtype=np.float64)
    local = np.array([rt for _, rt in trials], dtype=np.float64)

    baseline = float(np.percentile(local, BASELINE_PERCENTILE))
    if not np.isfinite(baseline) or baseline <= 0:
        return None, None, None

    glob = np.empty_like(local)
    for i, t in enumerate(times):
        window = local[(times >= t - GLOBAL_RT_WINDOW_S) & (times <= t)]
        glob[i] = window.mean() if window.size else local[i]

    labels = np.full(len(local), LABEL_DISCARD, dtype=np.int64)
    labels[(local < ALERT_RATIO * baseline) & (glob < ALERT_RATIO * baseline)] = LABEL_ALERT
    labels[(local > DROWSY_RATIO * baseline) & (glob > DROWSY_RATIO * baseline)] = LABEL_DROWSY

    return labels, baseline, glob


def cut_windows(data, times, labels, window_samples, sfreq, keep_intermediate=False):
    """Cut one window per trial, ending at the deviation onset.

    Trials whose reaction time falls between the alert and drowsy thresholds are
    kept when asked for, carrying label -1. They are more numerous than the
    labelled ones -- 10,727 against 11,012 over the dataset -- so discarding them
    throws away more data than it keeps. A training run can turn them into soft
    targets from the reaction-time ratio; evaluation must still use only the
    strict classes, or the metric stops meaning what it means everywhere else.
    """
    X, y, keep_idx = [], [], []
    for i, (t, lab) in enumerate(zip(times, labels)):
        if lab == LABEL_DISCARD and not keep_intermediate:
            continue
        end = int(round(t * sfreq))
        start = end - window_samples
        if start < 0 or end > data.shape[1]:
            continue
        X.append(data[:, start:end])
        y.append(lab)
        keep_idx.append(i)

    if not X:
        return None, None, None
    return np.stack(X), np.array(y, dtype=np.int64), np.array(keep_idx, dtype=np.int64)


def euclidean_align(X):
    """Whiten by the session's mean spatial covariance (He & Wu, 2020).

    Puts every session on a common covariance scale, which is what makes windows
    from different sessions and subjects comparable to a model that never sees a
    session identifier.
    """
    cov = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    cov += np.eye(cov.shape[0]) * 1e-10 * np.trace(cov) / cov.shape[0]
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    inv_sqrt = V @ np.diag(w ** -0.5) @ V.T
    return np.einsum("cd,ndt->nct", inv_sqrt, X)


def zscore_channelwise(X):
    """Per window, per channel: zero mean and unit variance over time."""
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-8)


def process_session(path, args):
    """Full pipeline for one session. Returns a summary dict."""
    subject, stem = parse_session(path)
    raw, traj, traj_sfreq = load_session(path)
    channels = list(raw.ch_names)

    rejected = extract_trials(raw.annotations, traj, traj_sfreq, args)
    trials = rejected.pop("ok")
    base = {"session": stem, "subject": subject, "n_trials": len(trials), **rejected}

    if len(trials) < args.min_trials:
        return {**base, "status": "too few clean trials"}

    labels, baseline, glob = label_trials(trials)
    if labels is None:
        return {**base, "status": "no usable baseline"}

    times = np.array([t for t, _ in trials])
    local = np.array([rt for _, rt in trials])

    window_samples = int(round(args.window_s * SFREQ_OUT))
    X, y, idx = cut_windows(raw.get_data(), times, labels, window_samples, SFREQ_OUT,
                            args.keep_intermediate)
    if X is None:
        return {**base, "status": "no windows survived"}

    if args.euclidean_alignment:
        X = euclidean_align(X)
    if args.zscore:
        X = zscore_channelwise(X)

    torch.save({
        "X": torch.from_numpy(X.astype(np.float32)),
        "y": torch.from_numpy(y),
        "rt": torch.from_numpy(local[idx].astype(np.float32)),
        "rt_global": torch.from_numpy(glob[idx].astype(np.float32)),
        # Reaction time as a multiple of the session's alert baseline. This is
        # what the binary rule thresholds at 1.5 and 2.5, kept as a continuous
        # value so a training run can use the intermediate trials.
        "ratio": torch.from_numpy((local[idx] / baseline).astype(np.float32)),
        "alert_baseline": baseline,
        "subject": subject,
        "session": stem,
        "channels": channels,
        "sfreq": SFREQ_OUT,
        "window_s": args.window_s,
    }, os.path.join(args.out, f"{stem}.pt"))

    return {
        **base, "status": "ok",
        "alert_baseline_s": round(baseline, 4),
        "windows_alert": int((y == LABEL_ALERT).sum()),
        "windows_drowsy": int((y == LABEL_DROWSY).sum()),
        "windows_intermediate": int((y == LABEL_DISCARD).sum()),
        "trials_discarded": int((labels == LABEL_DISCARD).sum()),
        "n_channels": len(channels),
    }


def _worker(args_tuple):
    path, args = args_tuple
    try:
        return process_session(path, args)
    except Exception as exc:  # noqa: BLE001 - one bad session must not stop the run
        subject, stem = parse_session(path)
        return {"session": stem, "subject": subject, "status": f"ERROR: {exc}"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.join(here, "sadt_raw"))
    ap.add_argument("--out", default=os.path.join(here, "sadt"))
    ap.add_argument("--window-s", type=float, default=3.0,
                    help="window length in seconds, ending at the deviation onset "
                         "(default: 3, the convention in the literature on this dataset)")
    ap.add_argument("--max-traj-std", type=float, default=0.3,
                    help="reject a trial whose pre-onset trajectory varies more than this "
                         "(lane units; the distribution is bimodal around ~0.05 and ~3)")
    ap.add_argument("--min-rt", type=float, default=0.3,
                    help="reaction times below this are rejected, per the dataset's "
                         "own tutorial (seconds)")
    ap.add_argument("--max-rt", type=float, default=10.0,
                    help="reaction times above this are treated as no response (seconds)")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="also write the trials whose reaction time falls between the "
                         "alert and drowsy thresholds, labelled -1")
    ap.add_argument("--min-trials", type=int, default=20,
                    help="skip sessions with fewer clean trials than this")
    ap.add_argument("--no-euclidean-alignment", dest="euclidean_alignment",
                    action="store_false")
    ap.add_argument("--no-zscore", dest="zscore", action="store_false")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    sessions = sorted(f for f in os.listdir(args.raw) if f.endswith(".set"))
    if not sessions:
        print(f"no .set files under {args.raw}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    print(f"{len(sessions)} sessions -> {args.out}")
    print(f"window {args.window_s}s @ {SFREQ_OUT:.0f} Hz ending at deviation onset, "
          f"EA={args.euclidean_alignment}, zscore={args.zscore}", flush=True)

    paths = [os.path.join(args.raw, f) for f in sessions]
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_worker, (p, args)): p for p in paths}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(paths)}] {r['session']}: {r['status']}", flush=True)
    else:
        for i, p in enumerate(paths, 1):
            r = _worker((p, args))
            results.append(r)
            print(f"[{i}/{len(paths)}] {r['session']}: {r['status']}", flush=True)

    results.sort(key=lambda r: r["session"])
    ok = [r for r in results if r["status"] == "ok"]

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": vars(args), "sessions": results}, fh, indent=2)

    print("\n=== summary ===")
    print(f"sessions written: {len(ok)}/{len(results)}")
    for r in results:
        if r["status"] != "ok":
            print(f"  skipped {r['session']}: {r['status']}")

    if not ok:
        return 1

    wa = sum(r["windows_alert"] for r in ok)
    wd = sum(r["windows_drowsy"] for r in ok)
    print(f"subjects: {len(sorted({r['subject'] for r in ok}))}")
    print(f"windows: {wa + wd} total, {wa} alert, {wd} drowsy "
          f"({wd / max(wa + wd, 1):.1%} drowsy)")
    print("trials rejected -- "
          f"broken triplet: {sum(r['broken_triplet'] for r in ok)}, "
          f"dirty trajectory: {sum(r['dirty_trajectory'] for r in ok)}, "
          f"too fast: {sum(r['too_fast'] for r in ok)}, "
          f"no response: {sum(r['no_response'] for r in ok)}")
    print(f"trials labelled but intermediate: {sum(r['trials_discarded'] for r in ok)}")

    per_sub = {}
    for r in ok:
        a, d = per_sub.get(r["subject"], (0, 0))
        per_sub[r["subject"]] = (a + r["windows_alert"], d + r["windows_drowsy"])
    empty = sorted(s for s, (a, d) in per_sub.items() if min(a, d) == 0)
    thin = sorted(s for s, (a, d) in per_sub.items() if 0 < min(a, d) < 50)
    if empty:
        print(f"subjects with an empty class (unusable as a LOSO test fold): {empty}")
    if thin:
        print(f"subjects with under 50 windows in one class: {thin}")
    print(f"\nsummary.json written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
