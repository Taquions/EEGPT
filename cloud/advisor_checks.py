"""CPU-only checks raised in the advisor's review, run on the existing result JSONs.

1. Per-subject counts (windows, drowsy, alert, prevalence).
2. Time-on-task baseline: logistic regression on the window's position in its
   session, LOSO, plus the AUROC of the raw position itself.
3. Model AUROC stratified by session time, so a model that only tracked the
   clock would fall to chance inside each stratum.
4. Paired differences with subject-bootstrap 95 % CIs and a TOST at +-1.5 pp
   (equivalence at 5 % <=> the 90 % CI lies inside the margin).
5. Holm over all four families (60 comparisons) instead of within each.
6. The same comparisons without the subjects with few drowsy windows.
7. Causal (past-only) smoothing.

    python cloud/advisor_checks.py
"""

import itertools
import os
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import ARMS, load, HALF_WIDTH
from smooth_predictions import smooth, balanced_accuracy, roc_auc
from attach_onsets import SESSION_TIME_STRIDE_S

MARGIN_PP = 1.5
MIN_DROWSY = 20
N_BOOT = 10_000


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def boot_ci(d, level, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), (N_BOOT, len(d)))
    q = (1 - level) / 2
    return np.quantile(d[idx].mean(1), [q, 1 - q])


def session_time(onset):
    return np.mod(onset, SESSION_TIME_STRIDE_S)


def causal_smooth(onset, score, width_s):
    """Mean over the past width_s seconds, current window included."""
    order = np.argsort(onset)
    t, s = onset[order], score[order]
    left = np.searchsorted(t, t - width_s, side="left")
    right = np.arange(1, len(t) + 1)
    cum = np.concatenate([[0.0], np.cumsum(s)])
    out = np.empty_like(score)
    out[order] = (cum[right] - cum[left]) / (right - left)
    return out


def metric_table(D, subs, fn):
    return {n: np.array([fn(D[n][s]) for s in subs]) for n in D}


def compare(M, names, label):
    rows = []
    for a, b in itertools.combinations(names, 2):
        _, p = wilcoxon(M[a], M[b])
        rows.append((label, a, b, (M[a] - M[b]) * 100, p))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    D = {n: load(os.path.join(here, "..", "downstream", "results_sadt"), t) for n, t, _ in ARMS}
    subs = sorted(set.intersection(*[set(d) for d in D.values()]))
    names = [n for n, _, _ in ARMS]
    ref = D[names[0]]

    print("## 1. Janelas por sujeito\n")
    print("| sujeito | janelas | sonolentas | alertas | prevalencia |")
    print("|---|---|---|---|---|")
    pos = {}
    for s in subs:
        y = ref[s][1]
        pos[s] = int((y == 1).sum())
        print(f"| {s} | {len(y)} | {pos[s]} | {int((y == 0).sum())} | {100 * y.mean():.1f} % |")
    few = [s for s in subs if pos[s] < MIN_DROWSY]
    print(f"\nSujeitos com menos de {MIN_DROWSY} janelas sonolentas: {few}")

    print("\n## 2. Baseline de tempo de tarefa\n")
    X = {s: session_time(ref[s][0])[:, None] / 60 for s in subs}
    Y = {s: ref[s][1] for s in subs}
    tb_bac, tb_auc = [], []
    for s in subs:
        tr = [u for u in subs if u != s]
        clf = LogisticRegression(class_weight="balanced").fit(
            np.vstack([X[u] for u in tr]), np.concatenate([Y[u] for u in tr]))
        tb_bac.append(balanced_accuracy(Y[s], clf.predict_proba(X[s])[:, 1]))
        tb_auc.append(roc_auc(Y[s], X[s][:, 0]))
    tb_bac, tb_auc = np.array(tb_bac), np.array(tb_auc)
    print(f"tempo -> BAC@0,5 {tb_bac.mean():.4f} +- {tb_bac.std(ddof=1):.4f}, "
          f"AUROC {tb_auc.mean():.4f} +- {tb_auc.std(ddof=1):.4f}")
    print("\n| braco | escala | BAC | vs tempo (pp) | p | AUROC | vs tempo (pp) | p |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        for hw in (0.0, HALF_WIDTH):
            b = np.array([balanced_accuracy(D[n][s][1], smooth(D[n][s][0], D[n][s][2], hw)) for s in subs])
            a = np.array([roc_auc(D[n][s][1], smooth(D[n][s][0], D[n][s][2], hw)) for s in subs])
            print(f"| {n} | {'crua' if hw == 0 else '60 s'} | {b.mean():.4f} | {100 * (b - tb_bac).mean():+.2f} | "
                  f"{wilcoxon(b, tb_bac).pvalue:.2g} | {a.mean():.4f} | {100 * (a - tb_auc).mean():+.2f} | "
                  f"{wilcoxon(a, tb_auc).pvalue:.2g} |")

    print("\n## 3. AUROC estratificada por tempo de sessao (faixas de 10 min)\n")
    print("Por sujeito, AUROC dentro de cada faixa com >= 3 sonolentas e >= 3 alertas, "
          "media das faixas; depois media entre sujeitos. 'tempo' e o proprio instante usado como pontuacao.\n")
    edges = np.arange(0, 130 * 60, 600)

    def strat_auc(y, t, sc):
        v = []
        for lo in edges:
            m = (t >= lo) & (t < lo + 600)
            if (y[m] == 1).sum() >= 3 and (y[m] == 0).sum() >= 3:
                v.append(roc_auc(y[m], sc[m]))
        return np.mean(v) if v else np.nan

    def late_auc(y, t, sc):
        m = t >= 20 * 60
        return roc_auc(y[m], sc[m]) if (y[m] == 1).sum() >= 3 and (y[m] == 0).sum() >= 3 else np.nan

    print("| pontuacao | escala | AUROC global | AUROC estratificada | AUROC apos 20 min | sujeitos (estr./apos 20) |")
    print("|---|---|---|---|---|---|")
    st = {s: session_time(ref[s][0]) for s in subs}
    g = np.array([roc_auc(Y[s], st[s]) for s in subs])
    e = np.array([strat_auc(Y[s], st[s], st[s]) for s in subs])
    l = np.array([late_auc(Y[s], st[s], st[s]) for s in subs])
    print(f"| tempo | -- | {np.nanmean(g):.4f} | {np.nanmean(e):.4f} | {np.nanmean(l):.4f} | "
          f"{np.isfinite(e).sum()}/{np.isfinite(l).sum()} |")
    for n in names:
        for hw in (0.0, HALF_WIDTH):
            sc = {s: smooth(D[n][s][0], D[n][s][2], hw) for s in subs}
            g = np.array([roc_auc(Y[s], sc[s]) for s in subs])
            e = np.array([strat_auc(Y[s], st[s], sc[s]) for s in subs])
            l = np.array([late_auc(Y[s], st[s], sc[s]) for s in subs])
            print(f"| {n} | {'crua' if hw == 0 else '60 s'} | {np.nanmean(g):.4f} | {np.nanmean(e):.4f} | "
                  f"{np.nanmean(l):.4f} | {np.isfinite(e).sum()}/{np.isfinite(l).sum()} |")

    fams = {
        "BAC suavizada": lambda d: balanced_accuracy(d[1], smooth(d[0], d[2], HALF_WIDTH)),
        "BAC crua": lambda d: balanced_accuracy(d[1], d[2]),
        "AUROC suavizada": lambda d: roc_auc(d[1], smooth(d[0], d[2], HALF_WIDTH)),
        "AUROC crua": lambda d: roc_auc(d[1], d[2]),
    }
    M = {f: metric_table(D, subs, fn) for f, fn in fams.items()}

    print(f"\n## 4. IC 95 % bootstrap por sujeito e TOST (margem {MARGIN_PP} pp)\n")
    print("Equivalente a 5 % se o IC 90 % cabe em +-margem. 'Margem minima' = maior |limite| do IC 90 %.\n")
    for f in ("BAC suavizada", "BAC crua"):
        print(f"### {f}\n")
        print("| A | B | dif (pp) | IC 95 % | IC 90 % | equivalente a 1,5 pp | margem minima (pp) |")
        print("|---|---|---|---|---|---|---|")
        for a, b in itertools.combinations(names, 2):
            d = (M[f][a] - M[f][b]) * 100
            c95, c90 = boot_ci(d, 0.95), boot_ci(d, 0.90)
            eq = "sim" if (c90[0] > -MARGIN_PP and c90[1] < MARGIN_PP) else "nao"
            print(f"| {a} | {b} | {d.mean():+.2f} | [{c95[0]:+.2f}, {c95[1]:+.2f}] | "
                  f"[{c90[0]:+.2f}, {c90[1]:+.2f}] | {eq} | {np.abs(c90).max():.2f} |")
        print()

    print("## 5. Holm global sobre as quatro familias (60 comparacoes)\n")
    rows = [r for f in fams for r in compare(M[f], names, f)]
    within = {}
    for f in fams:
        fr = [r for r in rows if r[0] == f]
        for r, pa in zip(fr, holm([r[4] for r in fr])):
            within[(f, r[1], r[2])] = pa
    glob_adj = holm([r[4] for r in rows])
    print("| familia | A | B | dif (pp) | p bruto | p Holm na familia | p Holm global |")
    print("|---|---|---|---|---|---|---|")
    for r, pg in sorted(zip(rows, glob_adj), key=lambda z: z[0][4]):
        pw = within[(r[0], r[1], r[2])]
        if pw < 0.05 or pg < 0.05:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3].mean():+.2f} | {r[4]:.2g} | {pw:.3g}{' *' if pw < 0.05 else ''} | "
                  f"{pg:.3g}{' *' if pg < 0.05 else ''} |")
    print("\n(so linhas significativas em pelo menos um dos dois criterios)")

    print(f"\n## 6. Sem os sujeitos com menos de {MIN_DROWSY} janelas sonolentas ({len(subs) - len(few)} sujeitos)\n")
    keep = np.array([s not in few for s in subs])
    print("| familia | A | B | dif (pp) | p | p Holm na familia | dif com todos (pp) |")
    print("|---|---|---|---|---|---|---|")
    for f in fams:
        Mk = {n: v[keep] for n, v in M[f].items()}
        fr = compare(Mk, names, f)
        for r, pa in zip(fr, holm([r[4] for r in fr])):
            if "atenção, congelado" in r[1] or pa < 0.05:
                full = (M[f][r[1]] - M[f][r[2]]).mean() * 100
                print(f"| {f} | {r[1]} | {r[2]} | {r[3].mean():+.2f} | {r[4]:.2g} | "
                      f"{pa:.3g}{' *' if pa < 0.05 else ''} | {full:+.2f} |")

    print("\n## 7. Suavizacao causal (so passado)\n")
    print("| braco | crua BAC/AUROC | centrada +-60 s | passado 60 s | passado 120 s |")
    print("|---|---|---|---|---|")
    for n in names:
        cells = []
        for fn in (lambda d: d[2], lambda d: smooth(d[0], d[2], HALF_WIDTH),
                   lambda d: causal_smooth(d[0], d[2], 60.0), lambda d: causal_smooth(d[0], d[2], 120.0)):
            b = np.mean([balanced_accuracy(D[n][s][1], fn(D[n][s])) for s in subs])
            a = np.mean([roc_auc(D[n][s][1], fn(D[n][s])) for s in subs])
            cells.append(f"{b:.4f} / {a:.4f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
