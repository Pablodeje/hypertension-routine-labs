import pandas as pd
import matplotlib.pyplot as plt

import config as c

# Figure 3: paired differences in test-set AUROC and Brier score against blood pressure alone
ALGORITHMS = [("LogisticRegression", "Elastic-net logistic\nregression"), ("kNN", "k-nearest\nneighbours"),
              ("MLP", "Multilayer\nperceptron"), ("RandomForest", "Random forest"), ("CatBoost", "CatBoost")]
SETS = [("BP+Lasso", "#E69F00", "o"), ("BP+SHAP", "#0072B2", "s"), ("BP+All", "#009E73", "D")]
PANELS = [("diff AUROC", "ΔAUROC vs. BP (95% CI)", "A"), ("diff Brier", "ΔBrier score vs. BP (95% CI)", "B")]

plt.rcParams.update({"font.family": "Arial", "font.size": 9})

df = pd.read_excel(c.BENCHMARK, sheet_name="Paired gain").set_index(["Algorithm", "Set"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)

for ax, (col, label, letter) in zip(axes, PANELS):
    for g, (algorithm, _) in enumerate(ALGORITHMS):
        for j, (s, color, marker) in enumerate(SETS):
            y = -(g * 4 + j)
            r = df.loc[(algorithm, s)]
            ax.errorbar(r[col], y, xerr=[[r[col] - r[col + " CI95 low"]], [r[col + " CI95 high"] - r[col]]],
                        fmt=marker, color=color, markersize=4, capsize=2, elinewidth=1,
                        label=s if g == 0 else None)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.08 if letter == "B" else -0.45, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

axes[0].set_yticks([-(g * 4 + 1) for g in range(len(ALGORITHMS))])
axes[0].set_yticklabels([name for _, name in ALGORITHMS])
axes[0].tick_params(axis="y", length=0)
axes[1].tick_params(axis="y", length=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0))

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(c.FIGURES / "fig3_forest.svg")
