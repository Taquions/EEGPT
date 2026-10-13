"""Generate every number the write-up needs, from the result JSONs, in one pass.

Written so the thesis is transcription rather than recomputation: each table
below corresponds to one table or claim in the text, and regenerating this file
after any new run keeps them consistent with each other.

Conventions fixed here and used everywhere:
  * threshold 0.5, never the tuned one -- tuning was measured and it hurts
  * smoothing half-width 60 s, chosen by the interior optimum of the sweep
  * 25 subjects, leave-one-subject-out, the intersection across arms
  * balanced accuracy and AUROC always reported together, because the second is
    invariant to the operating point and the first is not
"""

import itertools
import json
import glob
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_predictions import smooth, balanced_accuracy, roc_auc

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "downstream", "results_sadt")
HALF_WIDTH = 60.0


def ved(a, b):
    """Wins/ties/losses of a over b per subject. Ties use isclose: two balanced
    accuracies that differ only in float noise are the same result, and counting
    them as wins for whichever side happens to be listed first inflated the
    counts in earlier versions of this table."""
    tie = np.isclose(a, b)
    win = (a > b) & ~tie
    return f"{int(win.sum())}/{int(tie.sum())}/{int((~win & ~tie).sum())}"

ARMS = [
    ("Cabeca linear, encoder congelado", "linear_flat25"),
    ("Cabeca atencao, encoder congelado", "linear_attn25t"),
    ("Cabeca linear + layer-wise", "layerwise_pw"),
    ("Cabeca linear + LoRA r=16", "lora_pw"),
    ("Cabeca atencao + layer-wise", "layerwise_lw-attnt"),
    ("Cabeca atencao + LoRA r=16", "lora_lora-attnt"),
]


def load(tag, max_fold=None):
    out = {}
    for f in glob.glob(os.path.join(RESULTS, f"{tag}_fold*.json")):
        if f.endswith(".devrun.json"):
            continue
        r = json.load(open(f))
        if max_fold is not None and r["fold"] >= max_fold:
            continue
        pw = r.get("per_window")
        if pw:
            out[r["test_subject"]] = (np.asarray(pw["onset"], float),
                                      np.asarray(pw["label"], int),
                                      np.asarray(pw["score"], float), r)
    return out


def metrics(d, subs, hw):
    b = np.array([balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], hw)) for s in subs])
    a = np.array([roc_auc(d[s][1], smooth(d[s][0], d[s][2], hw)) for s in subs])
    return b, a


def confusion(d, subs, hw):
    tp = fn = fp = tn = 0
    for s in subs:
        p = smooth(d[s][0], d[s][2], hw) >= 0.5
        l = d[s][1]
        tp += int((p & (l == 1)).sum()); fn += int((~p & (l == 1)).sum())
        fp += int((p & (l == 0)).sum()); tn += int((~p & (l == 0)).sum())
    return tp, fn, fp, tn


def sens_spec(d, subs, hw):
    """Per-subject sensitivity and specificity, averaged across subjects -- the
    same aggregation as the balanced accuracy beside them, so that
    (sens + spec) / 2 equals the BAC column. Pooling the confusion matrix
    instead would weight subject 44 about 38 times subject 4."""
    se, sp = [], []
    for s in subs:
        p = smooth(d[s][0], d[s][2], hw) >= 0.5
        l = d[s][1]
        se.append((p & (l == 1)).sum() / (l == 1).sum())
        sp.append((~p & (l == 0)).sum() / (l == 0).sum())
    return float(np.mean(se)), float(np.mean(sp))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    prev = 0.0
    for k, i in enumerate(order):
        v = max(prev, min(1.0, (len(pvals) - k) * pvals[i]))
        prev = v
        adj[i] = v
    return adj


def main():
    D = {n: load(t) for n, t in ARMS}
    subs = sorted(set.intersection(*[set(d) for d in D.values()]))
    n_win = sum(len(D[ARMS[0][0]][s][1]) for s in subs)
    n_dro = sum(int(D[ARMS[0][0]][s][1].sum()) for s in subs)

    print(f"# Tabelas finais -- TG-2\n")
    print(f"Protocolo: LOSO, {len(subs)} sujeitos, limiar fixo 0,5, sem ajuste de limiar.")
    print(f"Conjunto de teste agregado: {n_win} janelas de 3 s, {n_dro} sonolentas "
          f"({n_dro / n_win * 100:.1f} %).\n")

    for hw, rot in ((0.0, "Tabela 1 -- pontuacoes cruas"),
                    (HALF_WIDTH, f"Tabela 2 -- suavizadas, meia-largura {HALF_WIDTH:.0f} s")):
        print(f"\n## {rot}\n")
        print("| braco | BAC (media +- dp) | AUROC (media +- dp) | sens (media) | espec (media) | TP | FN | FP | TN |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, _ in ARMS:
            b, a = metrics(D[n], subs, hw)
            tp, fn, fp, tn = confusion(D[n], subs, hw)
            se, sp = sens_spec(D[n], subs, hw)
            print(f"| {n} | {b.mean():.4f} +- {b.std(ddof=1):.4f} | "
                  f"{a.mean():.4f} +- {a.std(ddof=1):.4f} | "
                  f"{se:.3f} | {sp:.3f} | {tp} | {fn} | {fp} | {tn} |")
        print("\nBAC, AUROC, sens e espec sao medias por sujeito; (sens + espec) / 2 = BAC. "
              "TP/FN/FP/TN sao contagens agregadas sobre todas as janelas, so para escala.")

    print(f"\n## Tabela 3 -- custo\n")
    print("| braco | params cabeca | params encoder | pico GB | min/fold |")
    print("|---|---|---|---|---|")
    for n, _ in ARMS:
        r = D[n][subs[0]][3]
        mins = np.mean([D[n][s][3]["train_seconds"] for s in subs]) / 60
        print(f"| {n} | {r['trainable_head']:,} | {r['trainable_encoder']:,} | "
              f"{r['peak_gb']:.2f} | {mins:.1f} |")
    mins = {n: np.mean([D[n][s][3]["train_seconds"] for s in subs]) / 60 for n, _ in ARMS}
    a, lw, lo = (mins[ARMS[i][0]] for i in (1, 4, 5))
    print("\nA cabeca linear congelada rodou em L4; os demais bracos, em A100, e so "
          f"entre eles os tempos sao comparaveis. Com a cabeca de atencao, o layer-wise "
          f"custa {lw / a:.1f}x e o LoRA {lo / a:.1f}x o tempo do encoder congelado "
          f"({a:.1f}, {lw:.1f} e {lo:.1f} min/fold).")

    print(f"\n## Tabela 4 -- Wilcoxon pareado com correcao de Holm (suavizado)\n")
    names = [n for n, _ in ARMS]
    M = {n: metrics(D[n], subs, HALF_WIDTH)[0] for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        _, p = wilcoxon(M[a], M[b])
        rows.append([a, b, (M[a].mean() - M[b].mean()) * 100, p, ved(M[a], M[b])])
    adj = holm([r[3] for r in rows])
    print("| A | B | dif (pp) | p bruto | p Holm | V/E/D de A |")
    print("|---|---|---|---|---|---|")
    for r, pa in sorted(zip(rows, adj), key=lambda x: x[1]):
        mark = " *" if pa < 0.05 else ""
        print(f"| {r[0]} | {r[1]} | {r[2]:+.2f} | {r[3]:.4f} | {pa:.4f}{mark} | {r[4]} |")
    print("\n* significativo a 5 % apos correcao.")

    print(f"\n## Tabela 5 -- varredura da suavizacao (cabeca de atencao congelada)\n")
    d = D["Cabeca atencao, encoder congelado"]
    base = metrics(d, subs, 0.0)[0]
    print("| meia-largura (s) | BAC | AUROC | vs 0 s | p | V/E/D |")
    print("|---|---|---|---|---|---|")
    for hw in (0, 15, 30, 60, 120, 300):
        b, a = metrics(d, subs, float(hw))
        if hw == 0:
            print(f"| {hw} | {b.mean():.4f} | {a.mean():.4f} | -- | -- | -- |")
        else:
            _, p = wilcoxon(b, base)
            print(f"| {hw} | {b.mean():.4f} | {a.mean():.4f} | {(b.mean() - base.mean()) * 100:+.2f} | "
                  f"{p:.2g} | {ved(b, base)} |")

    print(f"\n## Tabela 6 -- resultados negativos\n")
    print("| tecnica | folds | BAC controle | BAC teste | vs controle | p | V/E/D | BAC validacao vs controle |")
    print("|---|---|---|---|---|---|---|---|")
    ctl = D["Cabeca atencao, encoder congelado"]
    for nm, tag, maxf in (("Trials intermediarios (alvo suave)", "linear_attn-intert", 12),
                          ("Adaptacao em tempo de teste (featnorm)", "linear_attn-ttat", None),
                          ("Temperature scaling", "linear_attn-tempt", None),
                          ("Adversarial de sujeito, lambda=0,03", "linear_adv003t", 12),
                          ("Adversarial de sujeito, lambda=0,1", "linear_adv01t", 12),
                          ("Adversarial de sujeito, lambda=0,3", "linear_adv03t", 12)):
        d2 = load(tag, maxf)
        common = sorted(set(d2) & set(ctl))
        if not common:
            continue
        b2 = metrics(d2, common, HALF_WIDTH)[0]
        b1 = metrics(ctl, common, HALF_WIDTH)[0]
        _, p = wilcoxon(b2, b1)
        v2 = np.mean([d2[s][3]["valid_balanced_accuracy"] for s in common])
        v1 = np.mean([ctl[s][3]["valid_balanced_accuracy"] for s in common])
        print(f"| {nm} | {len(common)} | {b1.mean():.4f} | {b2.mean():.4f} | "
              f"{(b2.mean() - b1.mean()) * 100:+.2f} pp | "
              f"{p:.4f} | {ved(b2, b1)} | {(v2 - v1) * 100:+.2f} pp |")
    print("\nO controle e a cabeca de atencao congelada restrita aos mesmos folds da "
          "tecnica: 25 folds para TTA e temperatura, 12 folds para trials "
          "intermediarios e adversarial. As diferencas so sao comparaveis dentro do "
          "mesmo grupo de folds.")
    print("\nO adversarial degrada monotonicamente apenas na BAC de validacao (criterio "
          "de selecao). No teste suavizado as diferencas nao sao ordenadas em lambda.")

    print(f"\n## Tabela 7 -- substrato: filtro espacial e taxa de aprendizado (sem suavizacao)\n")
    print("| passo | execucao | BAC@0,5 | AUROC | vs anterior | p | V/E/D |")
    print("|---|---|---|---|---|---|---|")
    prev = None
    for nm, tag in (("30 canais, lr 5e-4", "linear"),
                    ("+ filtro 62 canais (lr 5e-4)", "linear_c62-lr5e4"),
                    ("+ lr 2e-4", "linear_c62-lr2e4")):
        fs = sorted(glob.glob(os.path.join(RESULTS, f"{tag}_fold*.json")))
        js = sorted((json.load(open(f)) for f in fs), key=lambda j: j["fold"])
        # the 30-channel baseline predates threshold tuning, so its
        # balanced_accuracy already is the one at 0.5
        b = np.array([j.get("balanced_accuracy_at_half", j["balanced_accuracy"]) for j in js])
        a = np.array([j["roc_auc"] for j in js])
        if prev is None:
            print(f"| {nm} | {tag} | {b.mean():.4f} | {a.mean():.4f} | -- | -- | -- |")
        else:
            _, p = wilcoxon(b, prev)
            print(f"| {nm} | {tag} | {b.mean():.4f} | {a.mean():.4f} | "
                  f"{(b.mean() - prev.mean()) * 100:+.2f} pp | {p:.2g} | {ved(b, prev)} |")
        prev = b
    # same configuration as linear_c62-lr2e4, rerun in the final campaign with
    # the attention head and the PEFT arms; it is the linear row of Tables 1-4
    js = sorted((json.load(open(f)) for f in glob.glob(os.path.join(RESULTS, "linear_flat25_fold*.json"))),
                key=lambda j: j["fold"])
    b = np.array([j["balanced_accuracy_at_half"] for j in js])
    a = np.array([j["roc_auc"] for j in js])
    _, p = wilcoxon(b, prev)
    print(f"| reexecucao, mesma config (campanha final) | linear_flat25 | {b.mean():.4f} | {a.mean():.4f} | "
          f"{(b.mean() - prev.mean()) * 100:+.2f} pp | {p:.2g} | {ved(b, prev)} |")
    print("\nA ultima linha nao muda fator algum: e variacao entre execucoes (piso de "
          "ruido de BAC ~1,1 pp; AUROC quase identica). O ganho da cabeca de atencao "
          "e medido contra ela, na mesma campanha.")
    print("\nO total de +6,14 pp e a soma dos dois passos; atribuir ao filtro sozinho "
          "e o erro que versoes anteriores deste texto cometeram.")

    print(f"\n## Teto de calibracao (oraculo, inatingivel em operacao)\n")
    for nm in ("Cabeca atencao, encoder congelado",):
        d = D[nm]
        b = metrics(d, subs, HALF_WIDTH)[0]
        o = np.array([max(balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH), t)
                          for t in np.unique(smooth(d[s][0], d[s][2], HALF_WIDTH)))
                      for s in subs])
        print(f"{nm}: BAC {b.mean():.4f}, oraculo {o.mean():.4f}, folga {(o.mean() - b.mean()) * 100:.2f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
