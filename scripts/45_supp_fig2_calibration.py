import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as c

# Supplementary Figure 2: calibration of the best model of each set on the test set
MODELS = [("BP", "BP"), ("BP_Lasso", "BP+Lasso"), ("BP_SHAP", "BP+SHAP"), ("BP_All", "BP+All")]
N_GROUPS = 10
COLOR = "#0072B2"

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), sharex=True, sharey=True)

for ax, (slug, name), letter in zip(axes.flat, MODELS, "ABCD"):
    p = pd.read_excel(c.BENCHMARK_DIR / f"best_{slug}_predictions.xlsx", sheet_name="Predictions")
    p = p[p["Partition"] == "test"]
    p["group"] = pd.qcut(p["prob"], N_GROUPS, labels=False)
    g = p.groupby("group").agg(pred=("prob", "mean"), obs=("y_true", "mean"), n=("y_true", "size"))
    ci = 1.96 * np.sqrt(g["obs"] * (1 - g["obs"]) / g["n"])

    count, edges = np.histogram(p["prob"], bins=50, range=(0, 1))   # distribution, maximum height 0.2
    ax.bar(edges[:-1], count / count.max() * 0.2, width=0.02, align="edge", color="lightgrey")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.errorbar(g["pred"], g["obs"], yerr=ci, fmt="o", color=COLOR, markersize=4, capsize=2, elinewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(name)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.18, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

for ax in axes[1]:
    ax.set_xlabel("Predicted probability")
for ax in axes[:, 0]:
    ax.set_ylabel("Observed proportion")

fig.tight_layout()
fig.savefig(c.FIGURES / "supp_fig2_calibration.svg")
