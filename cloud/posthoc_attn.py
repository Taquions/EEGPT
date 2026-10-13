"""Post-hoc analyses of the frozen attention head, from per-window predictions.

Regenerates the calibration, noise-floor, dispersion and effect-size numbers of
the thesis that were first computed ad hoc. No GPU, no retraining: everything
is read from the result JSONs, under the conventions of final_tables.py
(60 s centred smoothing, threshold 0.5, BAC per subject averaged over subjects).

    python cloud/posthoc_attn.py                      # final 't' campaign
    python cloud/posthoc_attn.py --tag linear_attn25 --noise-tag linear_attn-temp \
        --tta-tag linear_attn-tta                     # numbers in the thesis

A --tag without the trailing 't' also switches the two attention PEFT arms of
the dispersion and Holm tables to their pre-'t' runs, which is what the thesis
tables were built on.
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_tables import ARMS, HALF_WIDTH, holm, load, metrics, ved
from smooth_predictions import balanced_accuracy, roc_auc, smooth

warnings.filterwarnings("ignore", module="scipy")  # exact-p fallbacks on ties

EXCLUDED = {14, 55}
DAY = 86400.0  # SESSION_TIME_STRIDE_S in train_EEGPT_SADT.py: session k starts at k days
GRID = np.linspace(0, 1, 1001)  # threshold grid for the leave-one-out transfer
PCTS = np.linspace(0, 100, 101)


def subjects(d):
    return sorted(set(d) - EXCLUDED)


def smoothed(d, subs):
    return {s: (d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH)) for s in subs}


def oracle(y, v):
    return max(balanced_accuracy(y, v, t) for t in np.unique(v))


def wil(a, b):
    """Wilcoxon p and two effect sizes r = |Z| / sqrt(N).

    r is the thesis definition: Z from the statistic with the mean and variance
    of N = all subjects, although scipy drops zero differences from T. With ties
    that inflates |Z| (attn+LW vs attn+LoRA, 5 ties: r = 0.358 at p = 0.74).
    r0 recomputes Z on the non-zero differences only, consistent with p."""
    w = wilcoxon(a, b)
    n = len(a)
    m = int((a != b).sum())
    z = lambda k: (w.statistic - k * (k + 1) / 4) / np.sqrt(k * (k + 1) * (2 * k + 1) / 24)
    return w.pvalue, abs(z(n)) / np.sqrt(n), abs(z(m)) / np.sqrt(n)


def row(name, x, ref, p=True):
    pv = f"{wilcoxon(x, ref).pvalue:.4f}" if p else "--"
    return f"| {name} | {x.mean():.4f} | {(x.mean() - ref.mean()) * 100:+.2f} pp | {pv} | {ved(x, ref)} |"


def saerens(p, prior_train=0.5, tol=1e-8):
    """EM prior adjustment (Saerens et al., 2002). The training prior is 0.5
    because the loss is class-balanced."""
    pi = prior_train
    for _ in range(1000):
        a = p * pi / prior_train
        q = a / (a + (1 - p) * (1 - pi) / (1 - prior_train))
        if abs(q.mean() - pi) < tol:
            break
        pi = q.mean()
    return q, pi


def gmm(y, v, **kw):
    """Unsupervised 2-component mixture on one subject's scores; drowsy is the
    posterior of the higher-mean component."""
    g = GaussianMixture(2, **kw).fit(v.reshape(-1, 1))
    return balanced_accuracy(y, g.predict_proba(v.reshape(-1, 1))[:, int(np.argmax(g.means_.ravel()))])


def session_time(onset):
    """Seconds since the first scored window of the window's own session."""
    ses = np.floor(onset / DAY)
    t = onset - ses * DAY
    for k in np.unique(ses):
        t[ses == k] -= t[ses == k].min()
    return t


def calibration(tag):
    d = load(tag)
    subs = subjects(d)
    S = smoothed(d, subs)
    fixed = np.array([balanced_accuracy(*S[s]) for s in subs])
    print(f"\n## Calibracao de limiar -- {tag}, {len(subs)} sujeitos, suavizado {HALF_WIDTH:.0f} s\n")
    print("| esquema | BAC | vs 0,5 fixo | p | V/E/D |")
    print("|---|---|---|---|---|")
    print(f"| limiar fixo 0,5 | {fixed.mean():.4f} | -- | -- | -- |")

    val = np.array([balanced_accuracy(*S[s], d[s][3]["threshold"]) for s in subs])
    print(row("limiar da validacao (escolhido em pontuacoes cruas)", val, fixed))

    # threshold maximising pooled BAC over the windows of the other 24 subjects
    loo = []
    for s in subs:
        y = np.concatenate([S[o][0] for o in subs if o != s])
        v = np.concatenate([S[o][1] for o in subs if o != s])
        t = GRID[int(np.argmax([balanced_accuracy(y, v, t) for t in GRID]))]
        loo.append(balanced_accuracy(*S[s], t))
    print(row("max. BAC nos outros 24 (agregado, grade 0,001)", np.array(loo), fixed))

    # percentile maximising the mean per-subject BAC of the other 24
    P = {s: np.array([balanced_accuracy(*S[s], np.percentile(S[s][1], q)) for q in PCTS]) for s in subs}
    pct = np.array([P[s][int(np.argmax(np.mean([P[o] for o in subs if o != s], axis=0)))] for s in subs])
    print(row("percentil aprendido nos outros 24", pct, fixed))

    g = np.array([gmm(*S[s], random_state=0, n_init=5) for s in subs])
    print(row("mistura de 2 gaussianas (n_init=5, seed 0)", g, fixed))
    for name, raw in (("EM de prior (Saerens), pontuacoes cruas", True),
                      ("EM de prior (Saerens), suavizadas", False)):
        b, err = [], []
        for s in subs:
            y, v = S[s]
            q, pi = saerens(np.clip(d[s][2] if raw else v, 1e-6, 1 - 1e-6))
            b.append(balanced_accuracy(y, smooth(d[s][0], q, HALF_WIDTH) if raw else q))
            err.append(abs(pi - y.mean()))
        print(row(f"{name} [erro de prevalencia {np.mean(err) * 100:.1f} pp]", np.array(b), fixed))
    orac = np.array([oracle(*S[s]) for s in subs])
    print(row("oraculo por sujeito (inatingivel)", orac, fixed))

    # the thesis' -1.40 pp is this raw-score delta, not the smoothed row above
    rt = np.array([d[s][3]["balanced_accuracy"] for s in subs])
    rh = np.array([d[s][3]["balanced_accuracy_at_half"] for s in subs])
    print(f"\nLimiar da validacao sobre pontuacoes cruas: {rt.mean():.4f} contra {rh.mean():.4f} "
          f"do 0,5 cru ({(rt.mean() - rh.mean()) * 100:+.2f} pp). A tese reportava como BAC desse "
          f"esquema o 0,5 suavizado somado a essa diferenca crua ({fixed.mean() + rt.mean() - rh.mean():.4f}), "
          f"nao a BAC suavizada da linha acima.")

    dg = (g - fixed) * 100
    alt = [np.mean([gmm(*S[s], random_state=k) for s in subs]) - fixed.mean() for k in range(10)]
    print(f"\nMistura de gaussianas: mediana do ganho {np.median(dg):+.2f} pp, de {dg.min():+.1f} "
          f"a {dg.max():+.1f} pp. Com n_init=1 e seeds 0-9 a media varia de "
          f"{min(alt) * 100:+.2f} a {max(alt) * 100:+.2f} pp (seed 2: {alt[2] * 100:+.2f}).")

    print("\n### Calibracao no inicio da propria sessao\n")
    print("Limiar = max. BAC nas janelas de cada sessao ate N min (pontuacoes suavizadas na "
          "gravacao inteira); avaliado no restante, contra 0,5 no mesmo restante.\n")
    print("| N min | sujeitos com 2 classes no inicio | avaliaveis | BAC calibrado | BAC 0,5 mesmo trecho | dif | p | V/E/D |")
    print("|---|---|---|---|---|---|---|---|")
    both = lambda y: y.min() == 0 and y.max() == 1
    for mins in (5, 30):
        cal, fx, n0 = [], [], 0
        for s in subs:
            (y, v), m = S[s], session_time(d[s][0]) < mins * 60
            n0 += both(y[m])
            if not (both(y[m]) and both(y[~m])):
                continue
            c = np.unique(v[m])
            t = c[int(np.argmax([balanced_accuracy(y[m], v[m], t) for t in c]))]
            cal.append(balanced_accuracy(y[~m], v[~m], t))
            fx.append(balanced_accuracy(y[~m], v[~m]))
        cal, fx = np.array(cal), np.array(fx)
        p = f"{wilcoxon(cal, fx).pvalue:.4f}" if len(cal) > 5 else "--"
        print(f"| {mins} | {n0}/{len(subs)} | {len(cal)} | {cal.mean():.4f} | {fx.mean():.4f} | "
              f"{(cal.mean() - fx.mean()) * 100:+.2f} pp | {p} | {ved(cal, fx)} |")

    # the thesis' example is the largest gap among the well-ordered subjects;
    # the overall largest gap is a subject that also ranks poorly
    a = np.array([roc_auc(*S[s]) for s in subs])
    gap_ = np.where(a >= 0.95, a - fixed, -np.inf)
    i, j = int(np.argmax(gap_)), int(np.argmax(a - fixed))
    print(f"\nSujeito extremo (maior AUROC - BAC entre AUROC >= 0,95, suavizado): sujeito {subs[i]}, "
          f"AUROC {a[i]:.4f}, BAC {fixed[i]:.4f}. Sem o corte de AUROC: sujeito {subs[j]}, "
          f"AUROC {a[j]:.4f}, BAC {fixed[j]:.4f}.")
    return d, subs, orac.mean() - fixed.mean()


def noise_floor(tag, noise_tag, d):
    e = load(noise_tag)
    common = sorted(set(subjects(d)) & set(subjects(e)))
    print(f"\n## Piso de ruido -- AUROC crua, {tag} contra {noise_tag}\n")
    if not common:
        print(f"Familia {noise_tag} ausente; secao pulada.")
        return
    a = np.array([roc_auc(d[s][1], d[s][2]) for s in common])
    b = np.array([roc_auc(e[s][1], e[s][2]) for s in common])
    diff = sum(not np.array_equal(rankdata(d[s][2]), rankdata(e[s][2])) for s in common)
    print(f"{len(common)} sujeitos: {a.mean():.4f} contra {b.mean():.4f}, "
          f"{(b.mean() - a.mean()) * 100:+.2f} pp, p = {wilcoxon(b, a).pvalue:.4f}, V/E/D {ved(b, a)}; "
          f"ordenacao crua difere em {diff}/{len(common)} sujeitos.")
    T = [e[s][3]["temperature"] for s in common if "temperature" in e[s][3]]
    if T:
        print(f"Temperatura ajustada ({noise_tag}): mediana {np.median(T):.2f} "
              f"(de {min(T):.2f} a {max(T):.2f}).")


def tta_gap(tta_tag, gap):
    print(f"\n## Folga ate o oraculo -- {tta_tag}\n")
    d = load(tta_tag)
    subs = subjects(d)
    if not subs:
        print(f"Familia {tta_tag} ausente; secao pulada.")
        return
    S = smoothed(d, subs)
    b = np.array([balanced_accuracy(*S[s]) for s in subs])
    o = np.array([oracle(*S[s]) for s in subs])
    print(f"{len(subs)} sujeitos: BAC {b.mean():.4f}, oraculo {o.mean():.4f}, folga "
          f"{gap * 100:.2f} -> {(o.mean() - b.mean()) * 100:.2f} pp.")


def arm_tables(tag):
    legacy = not tag.endswith("t")
    arms = [(n, tag if t == "linear_attn25t" else t[:-1] if legacy and t.endswith("t") else t)
            for n, t in ARMS]
    D = {n: load(t) for n, t in arms}
    subs = sorted(set.intersection(*[set(subjects(d)) for d in D.values()]))
    print(f"\n## Dispersao entre sujeitos -- {len(subs)} sujeitos\n")
    print("| braco | execucao | media | dp cru | dp suav. | mediana | Q1 - Q3 | min - max |")
    print("|---|---|---|---|---|---|---|---|")
    M = {}
    for n, t in arms:
        raw = metrics(D[n], subs, 0.0)[0]
        M[n] = b = metrics(D[n], subs, HALF_WIDTH)[0]
        q1, q3 = np.percentile(b, [25, 75])
        print(f"| {n} | {t} | {b.mean():.4f} | {raw.std(ddof=1):.4f} | {b.std(ddof=1):.4f} | "
              f"{np.median(b):.4f} | {q1:.3f} - {q3:.3f} | {b.min():.3f} - {b.max():.3f} |")
    print("\nBAC suavizada a 0,5; dp cru e da BAC sobre pontuacoes cruas.")

    print(f"\n## Wilcoxon pareado com Holm e tamanho de efeito (suavizado)\n")
    rows = []
    for a, b in itertools.combinations([n for n, _ in arms], 2):
        if M[a].mean() > M[b].mean():  # A is the arm with the higher mean
            a, b = b, a
        p, r, r0 = wil(M[b], M[a])
        rows.append([b, a, (M[b].mean() - M[a].mean()) * 100, p, ved(M[b], M[a]), r, r0])
    adj = holm([r[3] for r in rows])
    print("| A | B | dif (pp) | p bruto | p Holm | V/E/D de A | r | r sem empates |")
    print("|---|---|---|---|---|---|---|---|")
    for r, pa in sorted(zip(rows, adj), key=lambda x: x[0][3]):
        mark = " *" if pa < 0.05 else ""
        print(f"| {r[0]} | {r[1]} | {r[2]:+.2f} | {r[3]:.4f} | {pa:.4f}{mark} | {r[4]} | {r[5]:.3f} | {r[6]:.3f} |")
    print("\nr = |Z|/sqrt(N), N = sujeitos. A coluna r segue a tese: Z com media e variancia "
          "de N, embora o T do Wilcoxon descarte as diferencas nulas; com empates isso infla r. "
          "'r sem empates' usa o N efetivo no Z e e coerente com o p.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="linear_attn25t")
    ap.add_argument("--noise-tag", default="linear_attn-tempt")
    ap.add_argument("--tta-tag", default="linear_attn-ttat")
    args = ap.parse_args()

    print(f"# Analises post-hoc -- cabeca de atencao congelada ({args.tag})")
    d, _, gap = calibration(args.tag)
    noise_floor(args.tag, args.noise_tag, d)
    tta_gap(args.tta_tag, gap)
    arm_tables(args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
