"""Per-subject smoothed BAC of the two frozen arms, linear head against attention head.

    python cloud/head_paired.py --out <dir>
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_tables import ARMS, HALF_WIDTH, load, ved
from smooth_predictions import balanced_accuracy, smooth


def plot_head_paired(bac_linear, bac_atencao, out="Cap4/head-paired.pdf"):
    a, b = np.asarray(bac_linear), np.asarray(bac_atencao)
    up = b > a
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for y0, y1, u in zip(a, b, up):
        ax.plot([0, 1], [y0, y1], color="#003580" if u else "#c0392b",
                alpha=0.55 if u else 0.9, lw=1.1, marker="o", ms=3.5)
    ax.plot([0, 1], [a.mean(), b.mean()], color="black", lw=2.6, marker="D", ms=7, zorder=5)
    ax.annotate(f"{a.mean():.3f}", (0, a.mean()), xytext=(-12, 0), textcoords="offset points",
                ha="right", va="center", fontsize=10, fontweight="bold")
    ax.annotate(f"{b.mean():.3f}", (1, b.mean()), xytext=(12, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], ["Cabeça linear", "Cabeça com\natenção"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("BAC suavizada por sujeito")
    ax.set_title(f"Codificador congelado: {up.sum()} de {len(a)} sujeitos melhoram", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    leg = ax.legend(handles=[
        Line2D([], [], color="#003580", alpha=0.55, lw=1.1, marker="o", ms=3.5, label="sujeito que melhora"),
        Line2D([], [], color="#c0392b", lw=1.1, marker="o", ms=3.5, label="sujeito que piora"),
        Line2D([], [], color="black", lw=2.6, marker="D", ms=6, label=f"média dos {len(a)}"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    leg.set_in_layout(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tags = dict(ARMS)
    D = {k: load(tags[k]) for k in ("Cabeca linear, encoder congelado", "Cabeca atencao, encoder congelado")}
    lin_d, att_d = D.values()
    subs = sorted(set(lin_d) & set(att_d))
    bac = lambda d: np.array([balanced_accuracy(d[s][1], smooth(d[s][0], d[s][2], HALF_WIDTH)) for s in subs])
    lin, att = bac(lin_d), bac(att_d)
    print(f"{len(subs)} sujeitos: linear {lin.mean():.4f}, atencao {att.mean():.4f}, V/E/D {ved(att, lin)}")
    plot_head_paired(lin, att, os.path.join(args.out, "head-paired.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
