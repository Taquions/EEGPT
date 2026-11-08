"""CPU-only analyses requested in the advisor's second review, on existing results.

Nothing here trains an encoder. Every number comes from the per-window scores
already stored in the result JSONs, so the script can be rerun after any new
campaign and stays consistent with final_tables.py.

  1. time-on-task baseline: logistic regression on time since session start, LOSO;
     its AUROC per subject is simply AUROC(label, time), no fitting needed
  2. does the model add anything beyond the clock: AUROC within 10-min bins of
     each session, and a LOSO logistic regression on [time, score] against time alone
  3. paired differences between arms: 95 % bootstrap CI (resampling subjects),
     TOST at a 1.5 pp margin via the 90 % CI, and the smallest margin for which
     equivalence holds
  4. Holm over all four families at once (60 comparisons)
  5. positives per subject, and the main comparisons without subjects that have
     few drowsy windows
  6. causal smoothing: the same window length, but only the past

    python cloud/review_analyses.py > review_analyses.txt
"""

import itertools
import os
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_tables import ARMS, HALF_WIDTH, holm, load, ved
from smooth_predictions import balanced_accuracy, roc_auc, smooth

SESSION_STRIDE = 86400.0
MARGIN = 0.015
BIN_S = 600.0
N_BOOT = 10_000


def causal_smooth(onset, score, half_width_s):
    """Mean over [t - 2w, t]: same window length as the centred one, past only.

    Sessions are 86400 s apart on the onset axis, so a window never reaches
    into the previous session."""
    if half_width_s <= 0:
        return score.copy()
    order = np.argsort(onset)
    t, s = onset[order], score[order]
    left = np.searchsorted(t, t - 2 * half_width_s, side="left")
    right = np.searchsorted(t, t, side="right")
    cum = np.concatenate([[0.0], np.cumsum(s)])
    out = np.empty_like(score)
    out[order] = (cum[right] - cum[left]) / (right - left)
    return out


def session_time(onset):
    return np.mod(onset, SESSION_STRIDE)


def boot_ci(d, level, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), (N_BOOT, len(d)))
    q = (1 - level) / 2
    return np.quantile(d[idx].mean(1), [q, 1 - q])


def wilcoxon_p(d):
    return 1.0 if np.allclose(d, 0) else wilcoxon(d, zero_method="wilcox").pvalue


def per_subject(d, subs, hw, smoother=smooth):
    b = np.array([balanced_accuracy(d[s][1], smoother(d[s][0], d[s][2], hw)) for s in subs])
    a = np.array([roc_auc(d[s][1], smoother(d[s][0], d[s][2], hw)) for s in subs])
    return b, a


def time_baseline(d, subs):
    """LOSO logistic regression on time since session start (minutes)."""
    X = {s: session_time(d[s][0])[:, None] / 60 for s in subs}
    y = {s: d[s][1] for s in subs}
    bac, auc = [], []
    for s in subs:
        tr = [u for u in subs if u != s]
        clf = LogisticRegression(class_weight="balanced").fit(
            np.vstack([X[u] for u in tr]), np.concatenate([y[u] for u in tr]))
        bac.append(balanced_accuracy(y[s], clf.predict_proba(X[s])[:, 1]))
        auc.append(roc_auc(y[s], X[s][:, 0]))
    return np.array(bac), np.array(auc)


def time_plus_score(d, subs, hw):
    """LOSO logistic regression on [time, smoothed score]: does the score add to the clock?"""
    X = {s: np.column_stack([session_time(d[s][0]) / 60, smooth(d[s][0], d[s][2], hw)]) for s in subs}
    y = {s: d[s][1] for s in subs}
    bac, auc = [], []
    for s in subs:
        tr = [u for u in subs if u != s]
        clf = LogisticRegression(class_weight="balanced").fit(
            np.vstack([X[u] for u in tr]), np.concatenate([y[u] for u in tr]))
        p = clf.predict_proba(X[s])[:, 1]
        bac.append(balanced_accuracy(y[s], p))
        auc.append(roc_auc(y[s], p))
    return np.array(bac), np.array(auc)


def binned_auc(onset, label, score, bin_s=BIN_S, min_t=0.0):
    """Mean AUROC over (session, 10-min bin) cells that hold both classes.

    Within a bin the prevalence is roughly constant, so a score that only tracks
    time on task has little to rank; the cell weights are the cell sizes."""
    t = session_time(onset)
    cell = (onset // SESSION_STRIDE) * 1000 + t // bin_s
    vals, w = [], []
    for c in np.unique(cell[t >= min_t]):
        m = (cell == c) & (t >= min_t)
        if 0 < label[m].sum() < m.sum():
            vals.append(roc_auc(label[m], score[m]))
            w.append(m.sum())
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=w))


def pairwise(vals, names, label, alpha_sig=0.05):
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = vals[i], vals[j]
        d = a - b
        lo95, hi95 = boot_ci(d, 0.95)
        lo90, hi90 = boot_ci(d, 0.90)
        rows.append(dict(family=label, a=names[i], b=names[j], mean=d.mean(),
                         lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90,
                         p=wilcoxon_p(d), ved=ved(a, b),
                         equiv=(lo90 > -MARGIN) and (hi90 < MARGIN),
                         min_margin=max(abs(lo90), abs(hi90))))
    return rows


def print_rows(rows, padj_key="p_holm"):
    print(f"{'A':<36}{'B':<36}{'dif pp':>8}{'IC95 pp':>18}{'IC90 pp':>18}"
          f"{'p':>9}{'p_Holm':>9}{'p_glob':>9}  V/E/D   TOST1.5  marg.min")
    for r in rows:
        print(f"{r['a']:<36}{r['b']:<36}{100*r['mean']:>8.2f}"
              f"   [{100*r['lo95']:>6.2f},{100*r['hi95']:>6.2f}]"
              f"   [{100*r['lo90']:>6.2f},{100*r['hi90']:>6.2f}]"
              f"{r['p']:>9.4f}{r['p_holm']:>9.4f}{r['p_glob']:>9.4f}  {r['ved']:<7} "
              f"{'sim' if r['equiv'] else 'nao':<8} {100*r['min_margin']:.2f}")


def families(D, names, subs):
    fam = {}
    for hw, tag in [(HALF_WIDTH, "suav"), (0.0, "crua")]:
        ps = [per_subject(D[n], subs, hw) for n in names]
        fam[f"BAC {tag}"] = [p[0] for p in ps]
        fam[f"AUROC {tag}"] = [p[1] for p in ps]
    return fam


def run_pairwise(fam, names):
    allrows = []
    for label, vals in fam.items():
        rows = pairwise(vals, names, label)
        for r, p in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p
        allrows += rows
    for r, p in zip(allrows, holm([r["p"] for r in allrows])):
        r["p_glob"] = p
    return allrows


def main():
    names = [n for n, _ in ARMS]
    D = {n: load(t) for n, t in ARMS}
    subs = sorted(set.intersection(*[set(D[n]) for n in names]))
    ref = D[names[1]]
    print(f"{len(subs)} sujeitos: {subs}\n")

    # ---- 5a. positives per subject
    print("=== Janelas por sujeito ===")
    print(f"{'suj':>4}{'n':>6}{'sonol':>7}{'alerta':>8}{'preval':>8}{'sessoes':>8}{'dur_max_min':>12}")
    npos = {}
    for s in subs:
        o, y = ref[s][0], ref[s][1]
        npos[s] = int(y.sum())
        print(f"{s:>4}{len(y):>6}{int(y.sum()):>7}{int((y == 0).sum()):>8}{y.mean():>8.3f}"
              f"{len(np.unique(o // SESSION_STRIDE)):>8}{session_time(o).max() / 60:>12.1f}")
    print()

    # ---- 1. time baseline
    tb_bac, tb_auc = time_baseline(ref, subs)
    print("=== Baseline de tempo de tarefa (LOSO, regressao logistica sobre minutos desde o inicio da sessao) ===")
    print(f"BAC (limiar 0,5): {tb_bac.mean():.4f} +- {tb_bac.std(ddof=1):.4f}")
    print(f"AUROC = AUROC(rotulo, tempo): {tb_auc.mean():.4f} +- {tb_auc.std(ddof=1):.4f}")
    print(f"{'braco':<36}{'BAC suav':>10}{'dif pp':>8}{'p':>9}  V/E/D  {'AUROC suav':>11}{'dif pp':>8}{'p':>9}  V/E/D")
    for n in names:
        b, a = per_subject(D[n], subs, HALF_WIDTH)
        print(f"{n:<36}{b.mean():>10.4f}{100*(b-tb_bac).mean():>8.2f}{wilcoxon_p(b-tb_bac):>9.4f}  {ved(b, tb_bac)}"
              f"  {a.mean():>11.4f}{100*(a-tb_auc).mean():>8.2f}{wilcoxon_p(a-tb_auc):>9.4f}  {ved(a, tb_auc)}")
    print("por sujeito (AUROC tempo):", " ".join(f"{s}:{v:.2f}" for s, v in zip(subs, tb_auc)))
    print()

    # ---- 2. beyond the clock
    print("=== O modelo acrescenta algo alem do relogio? ===")
    print("AUROC dentro de faixas de 10 min de cada sessao (media ponderada por janelas, por sujeito; depois media entre sujeitos)")
    print(f"{'braco':<36}{'tudo crua':>10}{'tudo suav':>10}{'>=20min suav':>13}{'tempo, tudo':>12}{'n_suj':>6}")
    tb_bin = np.array([binned_auc(ref[s][0], ref[s][1], session_time(ref[s][0])) for s in subs])
    tb_bin20 = np.array([binned_auc(ref[s][0], ref[s][1], session_time(ref[s][0]), min_t=1200) for s in subs])
    for n in names:
        d = D[n]
        raw = np.array([binned_auc(d[s][0], d[s][1], d[s][2]) for s in subs])
        sm = np.array([binned_auc(d[s][0], d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH)) for s in subs])
        sm20 = np.array([binned_auc(d[s][0], d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH), min_t=1200) for s in subs])
        print(f"{n:<36}{np.nanmean(raw):>10.4f}{np.nanmean(sm):>10.4f}{np.nanmean(sm20):>13.4f}"
              f"{np.nanmean(tb_bin):>12.4f}{int(np.isfinite(sm).sum()):>6}")
    print(f"(tempo dentro das faixas: tudo {np.nanmean(tb_bin):.4f}, >=20 min {np.nanmean(tb_bin20):.4f})")
    print()
    print("LOSO logistica [tempo, pontuacao suavizada] contra so tempo")
    print(f"{'braco':<36}{'BAC':>8}{'dBAC pp':>9}{'p':>9}  V/E/D  {'AUROC':>8}{'dAUROC pp':>10}{'p':>9}  V/E/D")
    for n in names:
        b, a = time_plus_score(D[n], subs, HALF_WIDTH)
        print(f"{n:<36}{b.mean():>8.4f}{100*(b-tb_bac).mean():>9.2f}{wilcoxon_p(b-tb_bac):>9.4f}  {ved(b, tb_bac)}"
              f"  {a.mean():>8.4f}{100*(a-tb_auc).mean():>10.2f}{wilcoxon_p(a-tb_auc):>9.4f}  {ved(a, tb_auc)}")
    print()

    # ---- 3 + 4. CIs, TOST, global Holm
    print("=== Diferencas pareadas: IC bootstrap por sujeito, TOST 1,5 pp, Holm por familia e global (60) ===")
    rows = run_pairwise(families(D, names, subs), names)
    for fam in ["BAC suav", "AUROC suav", "BAC crua", "AUROC crua"]:
        print(f"--- {fam}")
        print_rows([r for r in rows if r["family"] == fam])
    print()

    # ---- 5b. without subjects with few positives
    for k in (10, 20):
        keep = [s for s in subs if npos[s] >= k]
        print(f"=== Excluindo sujeitos com < {k} janelas sonolentas: {len(keep)} sujeitos "
              f"(fora: {[s for s in subs if npos[s] < k]}) ===")
        fam = families(D, names, keep)
        for label in fam:
            print(f"  {label}: " + "  ".join(f"{n.split(',')[0].replace('Cabeca ', '')}={v.mean():.4f}"
                                            for n, v in zip(names, fam[label])))
        rows_k = run_pairwise(fam, names)
        sig = [r for r in rows_k if r["p_holm"] < 0.05]
        print(f"  significativas (Holm por familia): {len(sig)}")
        for r in sig:
            print(f"    {r['family']:<11} {r['a']} vs {r['b']}: {100*r['mean']:+.2f} pp, p_Holm {r['p_holm']:.4f}")
        print()

    # pooled (window-weighted) metrics: small subjects no longer weigh as much
    print("=== Media ponderada pelo numero de janelas (BAC suav / AUROC suav) ===")
    w = np.array([len(ref[s][1]) for s in subs], float)
    for n in names:
        b, a = per_subject(D[n], subs, HALF_WIDTH)
        print(f"{n:<36}{np.average(b, weights=w):>8.4f}{np.average(a, weights=w):>8.4f}")
    print()

    # ---- 6. causal smoothing
    print("=== Suavizacao causal [t-2w, t] contra centrada [t-w, t+w], mesmo comprimento ===")
    print(f"{'braco':<36}{'w s':>5}{'BAC centr':>10}{'BAC causal':>11}{'AUROC centr':>12}{'AUROC causal':>13}")
    for n in names:
        for hw in (30.0, 60.0, 90.0):
            bc, ac = per_subject(D[n], subs, hw)
            bk, ak = per_subject(D[n], subs, hw, causal_smooth)
            print(f"{n:<36}{hw:>5.0f}{bc.mean():>10.4f}{bk.mean():>11.4f}{ac.mean():>12.4f}{ak.mean():>13.4f}")
    print("Comparacoes na causal (w=60): Holm por familia")
    fam = {}
    ps = [per_subject(D[n], subs, HALF_WIDTH, causal_smooth) for n in names]
    fam["BAC causal"] = [p[0] for p in ps]
    fam["AUROC causal"] = [p[1] for p in ps]
    rows_c = run_pairwise(fam, names)
    for r in rows_c:
        if r["p_holm"] < 0.05:
            print(f"  {r['family']:<12} {r['a']} vs {r['b']}: {100*r['mean']:+.2f} pp, p_Holm {r['p_holm']:.4f}, {r['ved']}")


if __name__ == "__main__":
    main()
