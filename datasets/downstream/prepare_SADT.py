"""Preprocess the raw SADT recordings into windows labelled alert/drowsy.

Input is what `download_SADT.py` fetches: 62 EEGLAB `.set` sessions from the
Sustained-Attention Driving Task (Cao et al., 2019), 27 subjects, 500 Hz.

Output is one `.pt` per session holding a tensor of windows plus their labels,
which the training scripts group by subject for Leave-One-Subject-Out.

Pipeline
--------
1. Keep the 30 scalp channels: drop `vehicle position` (behavioural) and the
   mastoid references A1/A2. Rename the old 10-20 labels T3/T4/T5/T6 to their
   10-10 equivalents T7/T8/P7/P8, which is what EEGPT's channel vocabulary
   carries -- without this `prepare_chan_ids` raises an assertion.
2. Convert to microvolts and low-pass at 38 Hz, then resample 500 -> 256 Hz to
   match the rate EEGPT was pretrained on.
3. Recover reaction time per trial from the event codes: 251/252 mark the onset
   of the lane departure, 253 the onset of the participant's response.
4. Label each trial against the session's own alert baseline (see `label_trials`).
5. Cut a window ending at the deviation onset. The window is strictly *before*
   the event: EEG after the onset contains the motor response itself, so
   classifying drowsiness from it would be circular.
6. Euclidean Alignment per session, then channel-wise z-score per window.

Usage
-----
    python prepare_SADT.py                      # defaults, all sessions
    python prepare_SADT.py --jobs 4             # parallel over sessions
    python prepare_SADT.py --windows-per-trial 3
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

DEV_ONSET = ("251", "252")   # lane departure, left / right
RESP_ONSET = "253"           # participant starts correcting
RESP_OFFSET = "254"          # correction finished

# SADT predates the 10-10 naming; EEGPT's vocabulary does not know T3/T4/T5/T6.
RENAME_10_20 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
# Not scalp EEG: A1/A2 are the mastoid references, the last one is behavioural.
DROP_CHANNELS = ("A1", "A2", "VEHICLE POSITION")

# --- labelling constants -----------------------------------------------------

# The dataset's own tutorial defines the alert baseline as the trimmed mean of
# the fastest 10 % of trials in the session.
BASELINE_QUANTILE = 0.10
BASELINE_TRIM = 0.10
# Thresholds on the ratio to that baseline, as used in the SADT literature.
ALERT_RATIO = 1.5
DROWSY_RATIO = 2.5
# Window over which the moving ("global") reaction time is averaged.
GLOBAL_RT_WINDOW_S = 90.0

LABEL_ALERT, LABEL_DROWSY, LABEL_DISCARD = 0, 1, -1


def parse_session(filename):
    """`s01_051017m.set` -> (1, 's01_051017m')."""
    stem = os.path.basename(filename)[: -len(".set")]
    m = re.match(r"s(\d+)_", stem)
    if not m:
        raise ValueError(f"cannot parse a subject id out of {filename!r}")
    return int(m.group(1)), stem


def load_session(path, verbose=False):
    """Read one .set, keep the scalp channels, filter and resample."""
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")

    drop = [ch for ch in raw.ch_names if ch.upper().strip() in DROP_CHANNELS]
    raw.drop_channels(drop)
    raw.rename_channels({ch: RENAME_10_20[ch.upper()] for ch in raw.ch_names
                         if ch.upper() in RENAME_10_20})
    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})

    if abs(raw.info["sfreq"] - SFREQ_IN) > 1e-6:
        raise ValueError(f"{path}: expected {SFREQ_IN} Hz, found {raw.info['sfreq']}")

    raw.apply_function(lambda d: d * 1e6)              # V -> uV
    raw.filter(l_freq=None, h_freq=LOW_PASS_HZ, verbose="ERROR")
    raw.resample(SFREQ_OUT, verbose="ERROR")

    if verbose:
        print(f"  {len(raw.ch_names)} channels, {raw.n_times / SFREQ_OUT / 60:.1f} min")
    return raw


def extract_trials(annotations, min_rt_s, max_rt_s):
    """Pair each deviation onset with the response that follows it.

    Two kinds of trial carry no usable reaction time, and both are counted
    rather than silently dropped, because each one shifts the class balance in
    a different direction:

    *No response* -- the correction never arrives before the next deviation, or
    arrives implausibly late. In a drowsiness dataset this is not noise, it is
    the extreme of the phenomenon being measured.

    *Implausibly fast* -- these recordings contain a cluster of responses at
    exactly 0.016 s (eight samples at the acquisition rate), which is an order
    of magnitude below the floor of visuomotor reaction and is a marker
    artefact, not behaviour. They matter more than their count suggests: the
    alert baseline is built from the fastest trials, so leaving them in drags
    it towards zero and makes almost every real trial look drowsy by ratio.
    """
    desc = list(annotations.description)
    onset = list(annotations.onset)

    trials, no_response, too_fast = [], 0, 0
    for i, (d, t) in enumerate(zip(desc, onset)):
        if d not in DEV_ONSET:
            continue
        rt = None
        for dj, tj in zip(desc[i + 1:], onset[i + 1:]):
            if dj in DEV_ONSET:      # next trial started, this one went unanswered
                break
            if dj == RESP_ONSET:
                rt = tj - t
                break
        if rt is None or rt > max_rt_s:
            no_response += 1
            continue
        if rt < min_rt_s:
            too_fast += 1
            continue
        trials.append((t, rt))

    return trials, no_response, too_fast


def label_trials(trials):
    """Label each trial alert/drowsy against the session's own alert baseline.

    Both the trial's own reaction time and the moving average over the preceding
    90 s have to agree, which is what keeps a single fast response inside a
    drowsy stretch from being read as alertness.
    """
    from scipy.stats import trim_mean

    times = np.array([t for t, _ in trials], dtype=np.float64)
    local = np.array([rt for _, rt in trials], dtype=np.float64)

    n_fastest = max(1, int(round(len(local) * BASELINE_QUANTILE)))
    fastest = np.sort(local)[:n_fastest]
    baseline = trim_mean(fastest, BASELINE_TRIM) if len(fastest) > 2 else fastest.mean()
    if not np.isfinite(baseline) or baseline <= 0:
        return None, None, None

    glob = np.empty_like(local)
    for i, t in enumerate(times):
        window = local[(times >= t - GLOBAL_RT_WINDOW_S) & (times <= t)]
        glob[i] = window.mean() if window.size else local[i]

    labels = np.full(len(local), LABEL_DISCARD, dtype=np.int64)
    alert = (local < ALERT_RATIO * baseline) & (glob < ALERT_RATIO * baseline)
    drowsy = (local > DROWSY_RATIO * baseline) & (glob > DROWSY_RATIO * baseline)
    labels[alert] = LABEL_ALERT
    labels[drowsy] = LABEL_DROWSY

    return labels, baseline, glob


def cut_windows(data, times, labels, window_samples, n_per_trial, sfreq):
    """Cut `n_per_trial` windows ending at each deviation onset, 50 % overlapped.

    Window k covers [onset - (k/2 + 1) * window, onset - (k/2) * window), so the
    first one ends exactly at the onset and each further one steps half a window
    back into the pre-event interval.
    """
    stride = window_samples // 2
    X, y, keep_idx = [], [], []

    for i, (t, lab) in enumerate(zip(times, labels)):
        if lab == LABEL_DISCARD:
            continue
        end0 = int(round(t * sfreq))
        for k in range(n_per_trial):
            end = end0 - k * stride
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
    raw = load_session(path)
    channels = list(raw.ch_names)

    trials, no_response, too_fast = extract_trials(
        raw.annotations, args.min_rt, args.max_rt)
    if len(trials) < args.min_trials:
        return {"session": stem, "subject": subject, "status": "too few trials",
                "n_trials": len(trials), "no_response": no_response,
                "too_fast": too_fast}

    labels, baseline, glob = label_trials(trials)
    if labels is None:
        return {"session": stem, "subject": subject, "status": "no usable baseline",
                "n_trials": len(trials), "no_response": no_response,
                "too_fast": too_fast}

    times = np.array([t for t, _ in trials])
    local = np.array([rt for _, rt in trials])

    window_samples = int(round(args.window_s * SFREQ_OUT))
    data = raw.get_data()
    X, y, idx = cut_windows(data, times, labels, window_samples,
                            args.windows_per_trial, SFREQ_OUT)
    if X is None:
        return {"session": stem, "subject": subject, "status": "no windows survived",
                "n_trials": len(trials), "no_response": no_response,
                "too_fast": too_fast}

    if args.euclidean_alignment:
        X = euclidean_align(X)
    if args.zscore:
        X = zscore_channelwise(X)

    out_path = os.path.join(args.out, f"{stem}.pt")
    torch.save({
        "X": torch.from_numpy(X.astype(np.float32)),
        "y": torch.from_numpy(y),
        "rt": torch.from_numpy(local[idx].astype(np.float32)),
        "rt_global": torch.from_numpy(glob[idx].astype(np.float32)),
        "alert_baseline": float(baseline),
        "subject": subject,
        "session": stem,
        "channels": channels,
        "sfreq": SFREQ_OUT,
    }, out_path)

    return {
        "session": stem, "subject": subject, "status": "ok",
        "n_trials": len(trials), "no_response": no_response,
        "too_fast": too_fast,
        "alert_baseline_s": round(float(baseline), 4),
        "trials_alert": int((labels == LABEL_ALERT).sum()),
        "trials_drowsy": int((labels == LABEL_DROWSY).sum()),
        "trials_discarded": int((labels == LABEL_DISCARD).sum()),
        "windows_alert": int((y == LABEL_ALERT).sum()),
        "windows_drowsy": int((y == LABEL_DROWSY).sum()),
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
    ap.add_argument("--window-s", type=float, default=4.0,
                    help="window length in seconds (default: 4, matching EEGPT's pretraining)")
    ap.add_argument("--windows-per-trial", type=int, default=1,
                    help="windows per trial, stepping back by half a window each time")
    ap.add_argument("--min-rt", type=float, default=0.15,
                    help="reaction times below this are marker artefacts, not behaviour "
                         "(seconds; the recordings contain a cluster at 0.016 s)")
    ap.add_argument("--max-rt", type=float, default=10.0,
                    help="reaction times above this are treated as no response (seconds)")
    ap.add_argument("--min-trials", type=int, default=20,
                    help="skip sessions with fewer usable trials than this")
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
    print(f"window {args.window_s}s x {args.windows_per_trial} per trial @ {SFREQ_OUT:.0f} Hz, "
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

    if ok:
        wa = sum(r["windows_alert"] for r in ok)
        wd = sum(r["windows_drowsy"] for r in ok)
        nr = sum(r["no_response"] for r in ok)
        tf = sum(r["too_fast"] for r in ok)
        disc = sum(r["trials_discarded"] for r in ok)
        subs = sorted({r["subject"] for r in ok})
        print(f"subjects: {len(subs)} {subs}")
        print(f"windows: {wa + wd} total, {wa} alert, {wd} drowsy "
              f"({wd / max(wa + wd, 1):.1%} drowsy)")
        print(f"trials discarded as intermediate: {disc}")
        print(f"trials with no usable response: {nr}")
        print(f"trials rejected as implausibly fast (< {args.min_rt}s): {tf}")
        per_sub = {}
        for r in ok:
            a, d = per_sub.get(r["subject"], (0, 0))
            per_sub[r["subject"]] = (a + r["windows_alert"], d + r["windows_drowsy"])
        thin = [s for s, (a, d) in per_sub.items() if min(a, d) < 20]
        if thin:
            print(f"subjects with under 20 windows in one class: {thin}")
        print(f"\nsummary.json written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
