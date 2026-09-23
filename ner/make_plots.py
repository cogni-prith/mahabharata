#!/usr/bin/env python3
"""Publication figures for the delta study (memorization vs recognition).

Reads train_delta/delta_results.json, writes PNGs to train_delta/figures/.
Palette: Okabe-Ito (colorblind-safe). B1=grey floor, S=blue (hero), B2=orange.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RES = json.load(open("train_delta/delta_results.json"))
OUT = "train_delta/figures"
os.makedirs(OUT, exist_ok=True)

# --- colors (Okabe-Ito, CVD-safe) ---
C = {"B1": "#949494", "S": "#0072B2", "B2": "#E69F00"}
INK, INK2, GRID = "#222222", "#555555", "#DDDDDD"
LABELS = {"B1": "B1  untrained lg (zero-shot)",
          "S":  "S  trained lg (distilled)",
          "B2": "B2  LLM silver (teacher)"}
SHORT = {"B1": "B1 zero-shot", "S": "S distilled", "B2": "B2 teacher"}

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def barlabels(ax, bars, fmt="{:.1f}", dy=0.6, size=8.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=INK)


# ---------- Fig 1: systems overview (micro / macro / weighted) ----------
def fig_overview():
    metrics = ["micro", "macro", "weighted"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(metrics))
    w = 0.26
    for i, k in enumerate(["B1", "S", "B2"]):
        vals = [RES[m][k] for m in metrics]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w,
                      color=C[k], label=LABELS[k], edgecolor="white", linewidth=1.2)
        barlabels(ax, bars)
    ax.set_xticks(list(x)); ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel("Span-F1 vs human gold")
    ax.set_ylim(0, 100)
    ax.set_title("NER performance against human gold (8 test parvas)", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/fig1_systems_overview.png"); plt.close(fig)


# ---------- Fig 2: the ladder (floor -> student -> teacher) with deltas ----------
def fig_ladder():
    micro = RES["micro"]
    pts = [("B1", micro["B1"]), ("S", micro["S"]), ("B2", micro["B2"])]
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    y = 0
    ax.plot([micro["B1"], micro["B2"]], [y, y], color=GRID, lw=2, zorder=1)
    for k, v in pts:
        ax.scatter([v], [y], s=320, color=C[k], zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(v, y + 0.11, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(v, y - 0.13, SHORT[k], ha="center", fontsize=9.5, color=INK2)
    d = RES["deltas"]

    def bracket(x0, x1, yb, text, col, tx=None):
        ax.annotate("", xy=(x1, yb), xytext=(x0, yb),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
        ax.text(tx if tx is not None else (x0 + x1) / 2, yb + 0.04, text,
                ha="center", fontsize=10, color=col, fontweight="bold")
    bracket(micro["B1"], micro["S"], 0.32, f"Δ2 = +{d['D2_S_minus_B1']:.1f}", C["S"])
    bracket(micro["S"], micro["B2"], 0.32, f"Δ3 = +{d['D3_B2_minus_S']:.1f}", C["B2"])
    bracket(micro["B1"], micro["B2"], -0.36, f"Δ1 = +{d['D1_B2_minus_B1']:.1f}", INK2)
    ax.set_xlim(20, 100); ax.set_ylim(-0.7, 0.62)
    ax.set_yticks([]); ax.set_xlabel("Micro span-F1 vs human gold")
    ax.set_title("From zero-shot floor to distilled student to LLM teacher", fontweight="bold", loc="left")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.14,
             "Δ2 = distillation gain over the zero-shot floor (the real test)   ·   "
             "Δ3 = gap to the LLM teacher (silver ceiling)   ·   Δ1 = teacher vs floor",
             ha="center", fontsize=8.5, color=INK2)
    fig.savefig(f"{OUT}/fig2_delta_ladder.png"); plt.close(fig)


# ---------- Fig 3: per-label span-F1 ----------
def fig_per_label():
    order = sorted(RES["per_label_f1"], key=lambda l: -RES["support"].get(l, 0))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = range(len(order))
    h = 0.26
    for i, k in enumerate(["B2", "S", "B1"]):  # teacher top of each group
        vals = [RES["per_label_f1"][l][k] for l in order]
        off = (1 - i) * h
        bars = ax.barh([yi + off for yi in y], vals, h, color=C[k],
                       label=LABELS[k], edgecolor="white", linewidth=1.0)
        for b, v in zip(bars, vals):
            if v > 1:
                ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                        va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{l}  (n={RES['support'].get(l,0)})" for l in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Span-F1 vs human gold"); ax.set_xlim(0, 105)
    ax.set_title("Per-entity-type performance vs human gold", fontweight="bold", loc="left")
    handles, lab = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lab[::-1], frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/fig3_per_label.png"); plt.close(fig)


# ---------- Fig 4: per-parva micro-F1 ----------
def fig_per_parva():
    parvas = list(RES["per_parva_micro_f1"].keys())
    unreviewed = {"MahaPrasthanika", "Mausala"}
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = range(len(parvas))
    w = 0.26
    for i, k in enumerate(["B1", "S", "B2"]):
        vals = [RES["per_parva_micro_f1"][p][k] for p in parvas]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w, color=C[k],
                      label=LABELS[k], edgecolor="white", linewidth=1.0)
    labels = [(p + " ⚠" if p in unreviewed else p) for p in parvas]
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Micro span-F1 vs gold"); ax.set_ylim(0, 108)
    ax.set_title("Per-parva performance vs human gold   (⚠ = <0.1% human edits, silver≈gold)",
                 fontweight="bold", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{OUT}/fig4_per_parva.png"); plt.close(fig)


# ---------- Fig 5: token accuracy vs span-F1 (the metric gap) ----------
def fig_metric_gap():
    tok = RES["token_agreement_pct"]; span = RES["micro"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    keys = ["B1", "S", "B2"]
    x = range(len(keys)); w = 0.34
    b1 = ax.bar([xi - w/2 for xi in x], [tok[k] for k in keys], w,
                color="#BBBBBB", label="Token-level accuracy", edgecolor="white", linewidth=1.2)
    b2 = ax.bar([xi + w/2 for xi in x], [span[k] for k in keys], w,
                color=[C[k] for k in keys], label="Strict span-F1", edgecolor="white", linewidth=1.2)
    barlabels(ax, b1); barlabels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[k] for k in keys])
    ax.set_ylabel("Score vs human gold (%)"); ax.set_ylim(0, 108)
    ax.set_title("Token accuracy flatters; span-F1 is the honest metric", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{OUT}/fig5_token_vs_span.png"); plt.close(fig)


if __name__ == "__main__":
    fig_overview(); fig_ladder(); fig_per_label(); fig_per_parva(); fig_metric_gap()
    print("wrote:")
    for f in sorted(os.listdir(OUT)):
        print(" ", os.path.join(OUT, f))
