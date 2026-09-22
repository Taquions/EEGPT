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

ARMS = [
    ("Cabeca linear, encoder congelado", "linear_flat25"),
    ("Cabeca atencao, encoder congelado", "linear_attn25"),
    ("Cabeca linear + layer-wise", "layerwise_pw"),
    ("Cabeca linear + LoRA r=16", "lora_pw"),
    ("Cabeca atencao + layer-wise", "layerwise_lw-attn"),
    ("Cabeca atencao + LoRA r=16", "lora_lora-attn"),
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
        print("| braco | BAC (media +- dp) | AUROC (media +- dp) | sens | espec | TP | FN | FP | TN |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, _ in ARMS:
            b, a = metrics(D[n], subs, hw)
            tp, fn, fp, tn = confusion(D[n], subs, hw)
            print(f"| {n} | {b.mean():.4f} +- {b.std(ddof=1):.4f} | "
                  f"{a.mean():.4f} +- {a.std(ddof=1):.4f} | "
                  f"{tp / (tp + fn):.3f} | {tn / (tn + fp):.3f} | {tp} | {fn} | {fp} | {tn} |")

    print(f"\n## Tabela 3 -- custo\n")
    print("| braco | params cabeca | params encoder | pico GB | min/fold |")
    print("|---|---|---|---|---|")
    for n, _ in ARMS:
        r = D[n][subs[0]][3]
        mins = np.mean([D[n][s][3]["train_seconds"] for s in subs]) / 60
        print(f"| {n} | {r['trainable_head']:,} | {r['trainable_encoder']:,} | "
              f"{r['peak_gb']:.2f} | {mins:.1f} |")
    print("\nOs tempos por fold NAO sao comparaveis entre linhas: os bracos congelados "
          "rodaram em L4 e os demais em A100. Medido na mesma A100, a cabeca de atencao "
          "congelada leva 2,1 min/fold, o layer-wise 3,3 e o LoRA 6,1.")

    print(f"\n## Tabela 4 -- Wilcoxon pareado com correcao de Holm (suavizado)\n")
    names = [n for n, _ in ARMS]
    M = {n: metrics(D[n], subs, HALF_WIDTH)[0] for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        _, p = wilcoxon(M[a], M[b])
        rows.append([a, b, (M[a].mean() - M[b].mean()) * 100, p, int((M[a] > M[b]).sum())])
    adj = holm([r[3] for r in rows])
    print("| A | B | dif (pp) | p bruto | p Holm | vitorias |")
    print("|---|---|---|---|---|---|")
    for r, pa in sorted(zip(rows, adj), key=lambda x: x[1]):
        mark = " *" if pa < 0.05 else ""
        print(f"| {r[0]} | {r[1]} | {r[2]:+.2f} | {r[3]:.4f} | {pa:.4f}{mark} | {r[4]}/{len(subs)} |")
    print("\n* significativo a 5 % apos correcao.")

    print(f"\n## Tabela 5 -- varredura da suavizacao (cabeca de atencao congelada)\n")
    d = D["Cabeca atencao, encoder congelado"]
    base = metrics(d, subs, 0.0)[0]
    print("| meia-largura (s) | BAC | AUROC | vs 0 s | p | vitorias |")
    print("|---|---|---|---|---|---|")
    for hw in (0, 15, 30, 60, 120, 300):
        b, a = metrics(d, subs, float(hw))
        if hw == 0:
            print(f"| {hw} | {b.mean():.4f} | {a.mean():.4f} | -- | -- | -- |")
        else:
            _, p = wilcoxon(b, base)
            print(f"| {hw} | {b.mean():.4f} | {a.mean():.4f} | {(b.mean() - base.mean()) * 100:+.2f} | "
                  f"{p:.2g} | {int((b > base).sum())}/{len(subs)} |")

    print(f"\n## Tabela 6 -- resultados negativos\n")
    print("| tecnica | BAC | vs controle | p | vitorias |")
    print("|---|---|---|---|---|")
    ctl = D["Cabeca atencao, encoder congelado"]
    for nm, tag, maxf in (("Trials intermediarios (alvo suave)", "linear_attn-inter", 12),
                          ("Adaptacao em tempo de teste (featnorm)", "linear_attn-tta", None),
                          ("Temperature scaling", "linear_attn-temp", None),
                          ("Adversarial de sujeito, lambda=0,1", "linear_adv01", 12),
                          ("Adversarial de sujeito, lambda=0,3", "linear_adv03", 12)):
        d2 = load(tag, maxf)
        common = sorted(set(d2) & set(ctl))
        if not common:
            continue
        b2 = metrics(d2, common, HALF_WIDTH)[0]
        b1 = metrics(ctl, common, HALF_WIDTH)[0]
        _, p = wilcoxon(b2, b1)
        print(f"| {nm} | {b2.mean():.4f} | {(b2.mean() - b1.mean()) * 100:+.2f} pp | "
              f"{p:.4f} | {int((b2 > b1).sum())}/{len(common)} |")

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
