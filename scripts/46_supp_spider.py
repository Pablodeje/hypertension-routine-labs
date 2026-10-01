import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as c

# Supplementary figure (candidate): best model of each laboratory set against blood pressure alone

# ---------------- Settings ----------------
THRESHOLD = "Youden"       # "Youden" (fixed in train) or "0.5"
RANGE = (0.4, 0.9)         # same radial axis in all panels
REFERENCE = "BP"
SET_LABELS = {"BP+Lasso": "BP + Lasso", "BP+SHAP": "BP + SHAP", "BP+All": "BP + All"}   # panel order
COLOR, COLOR_REF = "#0072B2", "grey"

# ---------------- Data ----------------
t = pd.read_excel(c.BENCHMARK, sheet_name="Full")
t["F1"] = 2 * t[f"PPV ({THRESHOLD})"] * t[f"Sensitivity ({THRESHOLD})"] / (t[f"PPV ({THRESHOLD})"] + t[f"Sensitivity ({THRESHOLD})"])
axes_def = [("AUROC", "AUROC"), (f"Sensitivity ({THRESHOLD})", "Sensitivity"), (f"Specificity ({THRESHOLD})", "Specificity"),
            ("F1", "F1"), (f"Kappa ({THRESHOLD})", "Kappa")]


def best(set_name):   # best algorithm of the set by CV AUC in train (not by the test set)
    return t[t["Set"] == set_name].sort_values("CV AUC (train)", ascending=False).iloc[0]


def close(row):
    v = [row[col] for col, _ in axes_def]
    return np.append(v, v[0])


ref = best(REFERENCE)

# ---------------- Figure ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(1, len(SET_LABELS), figsize=(7.2, 2.8), subplot_kw={"projection": "polar"})
ang = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False)
ang_c = np.append(ang, ang[0])
levels = np.round(np.arange(RANGE[0] + 0.1, RANGE[1] + 0.001, 0.1), 1)

for ax, (set_name, name), letter in zip(axes, SET_LABELS.items(), "ABC"):
    row = best(set_name)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(*RANGE)

    # Polygonal grid instead of circular
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    for r in levels:
        ax.plot(ang_c, [r] * len(ang_c), color="lightgrey", linewidth=0.6, zorder=0)
    for a in ang:
        ax.plot([a, a], RANGE, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_yticks([])
    if letter == "A":   # scale only in panel A, next to the AUROC vertex (the radius is exact at the vertices)
        for r in levels:
            ax.text(np.deg2rad(-3), r, f"{r:.1f}", color="grey", fontsize=8, ha="right", va="center", zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2))

    ax.plot(ang_c, close(ref), color=COLOR_REF, linewidth=1.0, linestyle="--", marker="o", markersize=2.5,
            label="BP only")
    ax.plot(ang_c, close(row), color=COLOR, linewidth=1.2, marker="o", markersize=3, label="BP + laboratory predictors")
    ax.fill(ang_c, close(row), color=COLOR, alpha=0.15)

    ax.set_xticks(ang)
    ax.set_xticklabels([e for _, e in axes_def])
    ax.tick_params(axis="x", pad=2)
    ax.set_title(name, pad=16)
    ax.text(-0.15, 1.27, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(c.FIGURES / "supp_spider.svg")
print(f"Saved to {c.FIGURES / 'supp_spider.svg'}")
