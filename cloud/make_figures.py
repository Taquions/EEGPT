"""Figures for the write-up, drawn straight from the result JSONs.

The document has no result figure at all, and four claims in the text are much
easier to see than to read. Each function below draws exactly one of them.

    python cloud/make_figures.py --out <dir>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_predictions import smooth, balanced_accuracy, roc_auc

HALF_WIDTH = 60.0
ARMS = [
    ("Cabeça linear, congelado", "linear_flat25", "#9a9a9a"),
    ("Cabeça atenção, congelado", "linear_attn25", "#1f77b4"),
    ("Cabeça linear + layer-wise", "layerwise_pw", "#ff7f0e"),
    ("Cabeça linear + LoRA", "lora_pw", "#2ca02c"),
    ("Cabeça atenção + layer-wise", "layerwise_lw-attn", "#d62728"),
    ("Cabeça atenção + LoRA", "lora_lora-attn", "#9467bd"),
]


def load(results, tag):
    out = {}
    for f in glob.glob(os.path.join(results, f"{tag}_fold*.json")):
        if f.endswith(".devrun.json"):
            continue
        r = json.load(open(f))
        pw = r.get("per_window")
        if pw:
            out[r["test_subject"]] = (np.asarray(pw["onset"], float),
                                      np.asarray(pw["label"], int),
                                      np.asarray(pw["score"], float))
    return out


def fig_smoothing_sweep(D, subs, out):
    """Why 60 s: the gain has an interior optimum, which pure noise reduction
    would not produce. Both metrics are drawn because they peak at different
    widths -- 60 s for the decision, 120 s for the ranking."""
    d = D["Cabeça atenção, congelado"]
    hws = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300]
    bac = [np.mean([balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], h)) for s in subs]) for h in hws]
    auc = [np.mean([roc_auc(d[s][1], smooth(d[s][0], d[s][2], h)) for s in subs]) for h in hws]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(hws, bac, "o-", color="#1f77b4", label="Acurácia balanceada")
    ax.plot(hws, auc, "s--", color="#d62728", label="AUROC")
    ax.axvline(90, color="grey", ls=":", lw=1)
    ax.text(93, min(bac) + 0.005, "janela de 90 s\nda regra de rotulagem",
            fontsize=7, color="grey", va="bottom")
    ax.set_xlabel("Meia-largura da suavização (s)")
    ax.set_ylabel("Média sobre 25 sujeitos")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "smoothing-sweep.pdf"))
    plt.close(fig)


def fig_per_subject(D, subs, out):
    """The mean hides that some subjects sit at chance for every arm. Sorted by
    the baseline so the spread and the hard subjects are visible at a glance."""
    ref = D["Cabeça linear, congelado"]
    base = np.array([balanced_accuracy(ref[s][1], smooth(ref[s][0], ref[s][2], HALF_WIDTH))
                     for s in subs])
    order = np.argsort(base)
    xs = np.arange(len(subs))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for name, _tag, colour in ARMS:
        d = D[name]
        v = np.array([balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH)) for s in subs])
        ax.plot(xs, v[order], "o-", ms=3, lw=1, color=colour, label=name, alpha=0.85)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.text(len(subs) - 0.4, 0.507, "acaso", fontsize=7, ha="right", color="grey")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(subs[i]) for i in order], fontsize=6)
    ax.set_xlabel("Sujeito (ordenado pela linha de base)")
    ax.set_ylabel("Acurácia balanceada suavizada")
    # A legenda de seis entradas cobria os sujeitos difíceis quando desenhada
    # dentro dos eixos, que são justamente os que a figura existe para mostrar.
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7, frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per-subject.pdf"))
    plt.close(fig)


def fig_roc(D, subs, out):
    """The operating point is a mission parameter, not a hyperparameter: the
    curve is what the model delivers, and the cut on it is a policy choice."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    grid = np.linspace(0, 1, 201)
    for name, _tag, colour in ARMS:
        if "atenção, congelado" not in name and "atenção + layer" not in name:
            continue
        d = D[name]
        curves = []
        for s in subs:
            sm = smooth(d[s][0], d[s][2], HALF_WIDTH)
            l = d[s][1]
            thr = np.unique(sm)
            tpr = np.array([(sm[l == 1] >= t).mean() for t in thr])
            fpr = np.array([(sm[l == 0] >= t).mean() for t in thr])
            o = np.argsort(fpr)
            curves.append(np.interp(grid, fpr[o], tpr[o]))
        mean = np.mean(curves, axis=0)
        ax.plot(grid, mean, color=colour, lw=1.6, label=name)
        # ponto de operação com limiar 0,5
        tp = np.mean([(smooth(d[s][0], d[s][2], HALF_WIDTH)[d[s][1] == 1] >= 0.5).mean() for s in subs])
        fp = np.mean([(smooth(d[s][0], d[s][2], HALF_WIDTH)[d[s][1] == 0] >= 0.5).mean() for s in subs])
        ax.plot(fp, tp, "o", color=colour, ms=7, mec="k", mew=0.6, zorder=5)
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set_xlabel("Taxa de falso alarme (1 − especificidade)")
    ax.set_ylabel("Sensibilidade")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc-operacao.pdf"))
    plt.close(fig)


def fig_oracle(D, subs, out):
    """How much of the shortfall is where the cut falls rather than what the
    model sees: the distance from each point to the diagonal is the subject's
    calibration loss."""
    d = D["Cabeça atenção, congelado"]
    got = np.array([balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH)) for s in subs])
    orc = np.array([max(balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH), t)
                        for t in np.unique(smooth(d[s][0], d[s][2], HALF_WIDTH))) for s in subs])
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(orc, got, s=26, color="#1f77b4", zorder=3)
    lim = [0.45, 1.0]
    ax.plot(lim, lim, "k:", lw=1)
    for s, x, y in zip(subs, orc, got):
        if x - y > 0.1:
            ax.annotate(str(s), (x, y), fontsize=6, xytext=(3, -6),
                        textcoords="offset points", color="grey")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("BAC com o melhor limiar do sujeito (oráculo)")
    ax.set_ylabel("BAC com limiar fixo em 0,5")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "oraculo.pdf"))
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "downstream", "results_sadt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = {n: load(args.results, t) for n, t, _ in ARMS}
    subs = sorted(set.intersection(*[set(d) for d in D.values()]))
    print(f"{len(subs)} sujeitos")
    for fn in (fig_smoothing_sweep, fig_per_subject, fig_roc, fig_oracle):
        fn(D, subs, args.out)
        print(f"  {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
