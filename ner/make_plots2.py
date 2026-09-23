#!/usr/bin/env python3
"""Second batch of figures for the delta study.

Reads train_delta/delta_results.json + train_delta/extra_metrics.json,
writes fig6..fig10 to train_delta/figures/. Same Okabe-Ito palette as make_plots.py.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = json.load(open("train_delta/delta_results.json"))
EX = json.load(open("train_delta/extra_metrics.json"))
OUT = "train_delta/figures"
os.makedirs(OUT, exist_ok=True)

C = {"B1": "#949494", "S": "#0072B2", "B2": "#E69F00"}
P_COL, R_COL = "#0072B2", "#009E73"   # precision=blue, recall=green (CVD-safe)
INK, INK2, GRID = "#222222", "#555555", "#DDDDDD"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


# ---------- Fig 6: training curve (two panels, no dual axis) ----------
def fig_training():
    tc = EX["training_curve"]
    s, f, loss = tc["steps"], tc["dev_f"], tc["ner_loss"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(s, f, color=C["S"], lw=2)
    best_i = int(np.argmax(f))
    a1.scatter([s[best_i]], [f[best_i]], s=70, color=C["B2"], zorder=3, edgecolor="white")
    a1.annotate(f"best {f[best_i]:.1f}", (s[best_i], f[best_i]),
                textcoords="offset points", xytext=(6, -14), color=INK2, fontsize=9)
    a1.set_ylabel("Dev span-F1\n(vs silver, internal)")
    a1.set_ylim(0, max(f) * 1.15)
    a1.set_title("Training curve (checkpoint selection on the internal silver dev)",
                 fontweight="bold", loc="left")
    a2.plot(s, loss, color=INK2, lw=1.6)
    a2.set_ylabel("NER loss"); a2.set_xlabel("Training step")
    for a in (a1, a2):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "Dev-F1 here is fidelity to the LLM teacher, not gold accuracy "
             "(the gold result is span-F1 81.1, reported separately).",
             fontsize=8, color=INK2)
    fig.savefig(f"{OUT}/fig6_training_curve.png"); plt.close(fig)


# ---------- Fig 7: student precision vs recall per label ----------
def fig_pr():
    labs = sorted(EX["labels"], key=lambda l: -RES["support"].get(l, 0))
    pr = EX["student_pr"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    y = np.arange(len(labs)); h = 0.36
    pb = ax.barh(y + h/2, [pr[l]["p"] for l in labs], h, color=P_COL,
                 label="Precision", edgecolor="white", linewidth=1.0)
    rb = ax.barh(y - h/2, [pr[l]["r"] for l in labs], h, color=R_COL,
                 label="Recall", edgecolor="white", linewidth=1.0)
    for bars in (pb, rb):
        for b in bars:
            w = b.get_width()
            if w > 1:
                ax.text(w + 1, b.get_y() + b.get_height()/2, f"{w:.0f}",
                        va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{l}  (n={RES['support'].get(l,0)})" for l in labs], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Score vs human gold (%)"); ax.set_xlim(0, 105)
    ax.set_title("Student precision vs recall per entity type", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.03, "WEAPON, NORP, GPE and EVENT are precision-heavy / recall-poor: "
             "the student is cautious and misses them, rather than mislabelling.",
             fontsize=8, color=INK2)
    fig.savefig(f"{OUT}/fig7_student_precision_recall.png"); plt.close(fig)


# ---------- Fig 8: confusion matrix (token-level, row-normalized %) ----------
def fig_confusion():
    order = EX["order"]
    M = np.array(EX["confusion_row_pct"])
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Predicted (student)"); ax.set_ylabel("Gold (human)")
    ax.set_title("Token-level confusion, row-normalized (%)", fontweight="bold", loc="left")
    for i in range(len(order)):
        for j in range(len(order)):
            v = M[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("% of gold-row tokens", fontsize=9)
    ax.grid(False)
    fig.text(0.01, -0.09, "Diagonal = correct. Right column (O) = missed entities; "
             "bottom row (O) = spurious. Off-diagonal cells are type confusions.",
             fontsize=8, color=INK2)
    fig.savefig(f"{OUT}/fig8_confusion_matrix.png"); plt.close(fig)


# ---------- Fig 9: long tail — training support vs student F1 ----------
def fig_longtail():
    labs = EX["labels"]
    xs = [RES["support"].get(l, 1) for l in labs]
    ys = [RES["per_label_f1"][l]["S"] for l in labs]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(xs, ys, s=90, color=C["S"], edgecolor="white", linewidth=1.2, zorder=3)
    for l, x, y in zip(labs, xs, ys):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(7, 3),
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Gold entity count in the test set (log scale)")
    ax.set_ylabel("Student span-F1 vs gold")
    ax.set_ylim(-5, 100)
    ax.set_title("The long tail: rare types are where recognition breaks down",
                 fontweight="bold", loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.savefig(f"{OUT}/fig9_longtail_support_vs_f1.png"); plt.close(fig)


# ---------- Fig 10: recovery decomposition per label (floor + gain + residual) ----------
def fig_recovery():
    labs = sorted(EX["labels"], key=lambda l: -RES["support"].get(l, 0))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y = np.arange(len(labs))
    b1 = [RES["per_label_f1"][l]["B1"] for l in labs]
    gain = [max(0, RES["per_label_f1"][l]["S"] - RES["per_label_f1"][l]["B1"]) for l in labs]
    resid = [max(0, RES["per_label_f1"][l]["B2"] - RES["per_label_f1"][l]["S"]) for l in labs]
    ax.barh(y, b1, color=C["B1"], label="B1 floor (zero-shot)", edgecolor="white", linewidth=1.0)
    ax.barh(y, gain, left=b1, color=C["S"], label="Δ2 distillation gain", edgecolor="white", linewidth=1.0)
    ax.barh(y, resid, left=np.array(b1)+np.array(gain), color=C["B2"], alpha=0.55,
            label="Δ3 residual gap to teacher", edgecolor="white", linewidth=1.0, hatch="//")
    for i, l in enumerate(labs):
        s = RES["per_label_f1"][l]["S"]
        ax.text(min(s, 99) + 1, i, f"S={s:.0f}", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9.5); ax.invert_yaxis()
    ax.set_xlabel("Span-F1 vs human gold  (stacks to the teacher B2)"); ax.set_xlim(0, 105)
    ax.set_title("How much of the teacher each type recovers", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/fig10_recovery_decomposition.png"); plt.close(fig)


if __name__ == "__main__":
    fig_training(); fig_pr(); fig_confusion(); fig_longtail(); fig_recovery()
    print("wrote fig6..fig10 to", OUT)
