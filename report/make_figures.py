"""Figures for the v1 report, read straight from the committed results (no hand-typed numbers).

Usage: python report/make_figures.py   → report/fig_exp2.pdf, report/fig_exp3.pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", Path(__file__).resolve().parent
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs, light)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False,
})


def fig_exp2():
    r = json.loads((RES / "routing_quality.json").read_text())
    g = lambda k: (r[k]["ram_gb"], 100 * r[k]["gsm8k_acc"])
    task = [g("Qwen3-30B-A3B-4bit|task-K32"), g("Qwen3-30B-A3B-4bit|task-K48"),
            g("Qwen3-30B-A3B-4bit|task-K64|screen50"), g("Qwen3-30B-A3B-4bit|full")]
    glob, dense = g("Qwen3-30B-A3B-4bit|global-K32"), g("Qwen3-8B-4bit|dense")
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot(*zip(*task), color=S1, lw=2, marker="o", ms=6, mec="white", mew=1.5, zorder=3,
            label="30B-A3B, task-predicted resident experts")
    ax.plot(*glob, ls="none", color=S3, marker="s", ms=6, mec="white", mew=1.5, zorder=3,
            label="30B-A3B, task-agnostic resident experts")
    ax.plot(*dense, ls="none", color=S2, marker="D", ms=6, mec="white", mew=1.5, zorder=3, label="8B dense")
    for (x, y), t, off, ha, va in zip(task, ["32/128", "48/128", "64/128 (n=50)", "all 128"],
                                      [(2, 9), (6, -6), (6, -6), (-8, 0)], ["left", "left", "left", "right"],
                                      ["bottom", "top", "top", "center"]):
        ax.annotate(t, (x, y), xytext=off, textcoords="offset points", color=INK2, fontsize=8, ha=ha, va=va)
    ax.annotate("8B dense", dense, xytext=(6, -3), textcoords="offset points", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("resident weights (GB, 4-bit)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xlim(3.5, 18.5)
    ax.set_ylim(-4, 104)
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 0.99), fontsize=7.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp2.pdf", bbox_inches="tight")


def fig_exp3():
    ci = json.loads((RES / "exp3.json").read_text())["paired_ci"]
    sets = [("gsm8k", "GSM8K"), ("humaneval", "HumanEval"), ("notes", "personal notes"), ("drop", "DROP")]
    comps = [("oracle - base", "oracle task adapter − no adapter", S2),
             ("oracle - multi-r8", "oracle task adapter − shared LoRA r8 (equal cost)", S1),
             ("oracle - multi-r24", "oracle task adapter − shared LoRA r24 (equal storage)", S3)]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for j, (key, label, col) in enumerate(comps):
        for i, (s, _) in enumerate(sets):
            m, lo, hi = (100 * v for v in ci[key][s])
            y = i + (j - 1) * 0.24
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", color=col, ms=6, mec="white", mew=1.5, zorder=3, label=label if i == 0 else None)
    ax.set_yticks(range(len(sets)), [t for _, t in sets])
    ax.invert_yaxis()
    ax.set_xlabel("difference in accuracy / exact match (points), mean and 95% paired-bootstrap interval")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=7.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp3.pdf", bbox_inches="tight")


if __name__ == "__main__":
    fig_exp2()
    fig_exp3()
    print("wrote", OUT / "fig_exp2.pdf", OUT / "fig_exp3.pdf")
