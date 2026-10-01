import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

import config as c

# Figure 2: cross-validated k curves, adding analytes in SHAP order and in lasso order

# ---------------- Data ----------------
k_opt = int(pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")["Selected"].sum())


def summary(path):
    f = pd.read_excel(path, sheet_name="k curve by fold")
    n = f["fold"].nunique()
    res = f.groupby("k").agg(ll=("logloss", "mean"), ll_sd=("logloss", "std"),
                             auc=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
    res["ll_se"] = res["ll_sd"] / np.sqrt(n)
    res["auc_se"] = res["auc_sd"] / np.sqrt(n)
    return res


series = [(summary(c.SELECTION_CATBOOST), "SHAP order", "#0072B2", -2),     # Okabe-Ito blue
          (summary(c.KCURVE_LASSO), "Lasso order", "#D55E00", 2)]           # Okabe-Ito vermillion
res = series[0][0]

# ---------------- Figure ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                     "mathtext.fontset": "custom", "mathtext.it": "Arial:italic"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
panels = [("ll", "ll_se", "Log-loss (5-fold CV)", "A"), ("auc", "auc_se", "AUROC (5-fold CV)", "B")]

for ax, (col, se, label, letter) in zip(axes, panels):
    base = res.loc[res["k"] == 0, col].item()
    for d, name, color, dx in series:
        # 2-pt horizontal offset so that the error bars do not overlap
        eb = ax.errorbar(d["k"], d[col], yerr=d[se], fmt="o-", color=color, label=name,
                         markersize=3.5, linewidth=1.2, capsize=2, elinewidth=0.8)
        for artist in eb.get_children():
            artist.set_transform(offset_copy(ax.transData, fig=fig, x=dx, y=0, units="points"))
    ax.axhline(base, color="grey", linestyle=":", linewidth=1)
    ax.annotate("BP only", xy=(2, base), xytext=(0, 2), textcoords="offset points",
                color="grey", ha="left", va="bottom")
    ax.axvline(k_opt, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xscale("function", functions=(lambda x: np.sign(x) * np.sqrt(np.abs(x)),
                                         lambda x: np.sign(x) * x ** 2))   # square-root axis
    ax.set_xlim(-1, res["k"].max() + 8)
    ax.set_xticks([0, 5, 10, 20, 30, 50, 70, 102])
    ax.set_xlabel("Laboratory predictors added to blood pressure ($k$)")
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

axes[0].annotate(f"$k$ = {k_opt}", xy=(k_opt, 0.5), xycoords=("data", "axes fraction"),
                 xytext=(-3, 0), textcoords="offset points",
                 color="grey", ha="right", va="center")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(c.FIGURES / "fig2_k_curve.svg")
