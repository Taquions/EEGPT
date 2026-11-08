"""Classical spectral baseline: band power plus logistic regression, LOSO.

Answers "and against classical methods?". The windows are exactly the ones the
EEGPT runs scored (same labelling and trial selection as the main variant), but
without Euclidean alignment and without the per-window z-score, since the
z-score divides out the absolute power this baseline is built on:

    python datasets/downstream/prepare_SADT.py --out sadt_noea_ts \
        --no-euclidean-alignment --no-zscore

Two feature sets and two normalisations:
  compact   log theta, log alpha and log theta/alpha, averaged over channels
  full      log absolute and log relative power in delta/theta/alpha/beta,
            per channel (30 x 8 = 240 features)
  none      features as they are
  session   features standardised within each session, the analogue of the
            per-session Euclidean alignment the EEGPT pipeline uses -- equally
            transductive and non-causal, so the comparison is like for like

Metrics, smoothing and the comparison against the six EEGPT arms follow
final_tables.py: threshold 0.5, centred mean over +/- 60 s, Wilcoxon per subject.

    python cloud/spectral_baseline.py --data datasets/downstream/sadt_noea_ts
"""

import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_tables import ARMS, HALF_WIDTH, holm, load, ved
from smooth_predictions import balanced_accuracy, roc_auc, smooth

SESSION_STRIDE = 86400.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def band_powers(X, sfreq):
    """(n, c, t) -> (n, c, len(BANDS)) absolute power, Welch with 1 s segments."""
    f, p = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
    return np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS.values()], -1)


def features(bp, kind):
    eps = 1e-30
    if kind == "compact":
        th, al = bp[..., 1].mean(1), bp[..., 2].mean(1)
        return np.column_stack([np.log(th + eps), np.log(al + eps), np.log(th / (al + eps) + eps)])
    rel = bp / (bp.sum(-1, keepdims=True) + eps)
    n = bp.shape[0]
    return np.concatenate([np.log(bp + eps).reshape(n, -1), np.log(rel + eps).reshape(n, -1)], 1)


def load_sessions(data_dir, subjects):
    """Per subject: list of (onset, label, band power), sessions in sorted filename order."""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
        b = torch.load(f, map_location="cpu")
        s = int(b["subject"])
        if s not in subjects:
            continue
        o = b["onsets"].double().numpy() + SESSION_STRIDE * len(out[s])
        out[s].append((o, b["y"].numpy().astype(int), band_powers(b["X"].numpy().astype(np.float64), b["sfreq"])))
    return out


def subject_matrix(sessions, kind, norm):
    feats, ons, ys = [], [], []
    for o, y, bp in sessions:
        F = features(bp, kind)
        if norm == "session":
            F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        feats.append(F); ons.append(o); ys.append(y)
    return np.concatenate(feats), np.concatenate(ons), np.concatenate(ys)


def loso(S, subs, kind, norm, C):
    mats = {s: subject_matrix(S[s], kind, norm) for s in subs}
    preds = {}
    for s in subs:
        tr = [u for u in subs if u != s]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        clf.fit(np.vstack([mats[u][0] for u in tr]), np.concatenate([mats[u][2] for u in tr]))
        F, o, y = mats[s]
        preds[s] = (o, y, clf.predict_proba(F)[:, 1])
    return preds


C_GRID = (0.01, 0.1, 1.0, 10.0)


def eegpt_folds(S, subs, folds, kind):
    """The EEGPT split per fold: fit on the same 20 training subjects, pick C by
    pooled window-level BAC (threshold 0.5) on the same 4 validation subjects,
    score the same test subject. Validation never sees the test subject."""
    mats = {s: subject_matrix(S[s], kind, "none") for s in subs}
    stack = lambda group, k: (np.vstack if k == 0 else np.concatenate)([mats[u][k] for u in group])
    preds, chosen = {}, {}
    for s in subs:
        valid = list(folds[s])
        train = [u for u in subs if u != s and u not in valid]
        Xtr, ytr, Xva, yva = stack(train, 0), stack(train, 2), stack(valid, 0), stack(valid, 2)
        best = None
        for C in C_GRID:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
            clf.fit(Xtr, ytr)
            v = balanced_accuracy(yva, clf.predict_proba(Xva)[:, 1])
            if best is None or v > best[0]:
                best = (v, C, clf)
        F, o, y = mats[s]
        preds[s] = (o, y, best[2].predict_proba(F)[:, 1])
        chosen[s] = (best[1], best[0], len(train))
    return preds, chosen


def scores(preds, subs, hw):
    b = np.array([balanced_accuracy(preds[s][1], smooth(preds[s][0], preds[s][2], hw)) for s in subs])
    a = np.array([roc_auc(preds[s][1], smooth(preds[s][0], preds[s][2], hw)) for s in subs])
    return b, a


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "datasets", "downstream", "sadt_noea_ts"))
    ap.add_argument("--C", type=float, default=1.0,
                    help="fixed C for the 24-subject LOSO variants (not tuned)")
    ap.add_argument("--protocol", choices=["loso24", "eegpt"], default="eegpt",
                    help="eegpt: same 20 train / 4 valid / 1 test split as the EEGPT runs, "
                         "C chosen on validation; loso24: train on all 24 others, fixed C")
    args = ap.parse_args()

    names = [n for n, _ in ARMS]
    D = {n: load(t) for n, t in ARMS}
    subs = sorted(set.intersection(*[set(D[n]) for n in names]))
    S = load_sessions(args.data, set(subs))

    ref = D[names[1]]
    for s in subs:
        o = np.concatenate([x[0] for x in S[s]])
        y = np.concatenate([x[1] for x in S[s]])
        if len(o) != len(ref[s][0]) or not np.allclose(o, ref[s][0], atol=1e-3) or not (y == ref[s][1]).all():
            raise SystemExit(f"subject {s}: windows differ from the EEGPT runs; wrong --data?")
    print(f"{len(subs)} sujeitos, janelas identicas as das execucoes EEGPT\n")

    eeg = {n: (scores({s: D[n][s][:3] for s in subs}, subs, 0.0),
               scores({s: D[n][s][:3] for s in subs}, subs, HALF_WIDTH)) for n in names}

    if args.protocol == "eegpt":
        folds = {s: tuple(D[names[1]][s][3]["valid_subjects"]) for s in subs}
        for n in names:
            if any(tuple(D[n][s][3]["valid_subjects"]) != folds[s] for s in subs):
                raise SystemExit(f"{n}: validation subjects differ from the reference arm")
        preds, chosen = eegpt_folds(S, subs, folds, "full")
        print(f"=== espectral full, sem normalizacao, folds do EEGPT, C em {C_GRID} pela BAC de validacao ===")
        print("C escolhido por fold: " + "  ".join(f"{s}:{c[0]:g}" for s, c in chosen.items()))
        print("contagem: " + ", ".join(f"C={C:g}: {sum(c[0] == C for c in chosen.values())}" for C in C_GRID)
              + f"; sujeitos de treino por fold: {sorted({c[2] for c in chosen.values()})}")
        spec = {0.0: scores(preds, subs, 0.0), HALF_WIDTH: scores(preds, subs, HALF_WIDTH)}
        for hw, lab in [(0.0, "crua"), (HALF_WIDTH, "60 s")]:
            b, a = spec[hw]
            print(f"{lab}: BAC {b.mean():.4f} +- {b.std(ddof=1):.4f}   AUROC {a.mean():.4f} +- {a.std(ddof=1):.4f}")
        print()
        rows = []
        for hw, lab in [(HALF_WIDTH, "suav"), (0.0, "crua")]:
            for k, metric in [(0, "BAC"), (1, "AUROC")]:
                ref_v = spec[hw][k]
                for n in names:
                    v = eeg[n][0 if hw == 0.0 else 1][k]
                    d = v - ref_v
                    rows.append([f"{metric} {lab}", n, v.mean(), d.mean(),
                                 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue, ved(v, ref_v)])
        fams = sorted({r[0] for r in rows}, key=[r[0] for r in rows].index)
        for f in fams:
            idx = [i for i, r in enumerate(rows) if r[0] == f]
            for i, p in zip(idx, holm([rows[i][4] for i in idx])):
                rows[i].append(p)
        for i, p in enumerate(holm([r[4] for r in rows])):
            rows[i].append(p)
        print(f"{'familia':<12}{'braco EEGPT':<36}{'EEGPT':>8}{'EEGPT-espec pp':>16}{'p':>9}{'p_Holm':>9}{'p_glob24':>10}  V/E/D (EEGPT)")
        for r in rows:
            print(f"{r[0]:<12}{r[1]:<36}{r[2]:>8.4f}{100*r[3]:>16.2f}{r[4]:>9.4f}{r[6]:>9.4f}{r[7]:>10.4f}  {r[5]}")
        return

    for kind in ("compact", "full"):
        for norm in ("none", "session"):
            preds = loso(S, subs, kind, norm, args.C)
            (br, ar), (bs, as_) = scores(preds, subs, 0.0), scores(preds, subs, HALF_WIDTH)
            print(f"=== espectral {kind}, normalizacao {norm}, C={args.C} ===")
            print(f"crua:  BAC {br.mean():.4f} +- {br.std(ddof=1):.4f}   AUROC {ar.mean():.4f} +- {ar.std(ddof=1):.4f}")
            print(f"60 s:  BAC {bs.mean():.4f} +- {bs.std(ddof=1):.4f}   AUROC {as_.mean():.4f} +- {as_.std(ddof=1):.4f}")
            print(f"{'braco EEGPT (60 s) menos espectral':<38}{'dBAC pp':>9}{'p':>9}  V/E/D  {'dAUROC pp':>10}{'p':>9}  V/E/D")
            for n in names:
                b, a = eeg[n][1]
                print(f"{n:<38}{100*(b-bs).mean():>9.2f}{wilcoxon(b-bs).pvalue:>9.4f}  {ved(b, bs)}"
                      f"  {100*(a-as_).mean():>10.2f}{wilcoxon(a-as_).pvalue:>9.4f}  {ved(a, as_)}")
            print()


if __name__ == "__main__":
    main()
