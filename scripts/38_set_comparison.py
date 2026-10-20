"""
Comparison between the laboratory predictor sets reported in the text (Results, Comparison between predictor sets):
paired differences in test AUROC between the best models of BP+Lasso, BP+SHAP and BP+All, on the same
bootstrap resamples as the benchmark, and the share of the gain over BP retained by each selected set.
"""
import numpy as np
import pandas as pd

import config as c
from benchmark_common import N_BOOT_CI, SEED, bootstrap_metrics

SETS = {"BP+Lasso": "BP_Lasso", "BP+SHAP": "BP_SHAP", "BP+All": "BP_All"}
PAIRS = [("BP+SHAP", "BP+Lasso"), ("BP+All", "BP+Lasso"), ("BP+All", "BP+SHAP")]

auc, algorithm = {}, {}
y = None
for s, slug in SETS.items():
    path = c.BENCHMARK_DIR / f"best_{slug}_predictions.xlsx"
    p = pd.read_excel(path, sheet_name="Predictions")
    p = p[p["Partition"] == "test"]
    info = pd.read_excel(path, sheet_name="Model").set_index("Field")["Value"]
    if y is None:
        y, ids = p["y_true"].to_numpy(), p["subject_id"].to_numpy()
    elif not (p["subject_id"].to_numpy() == ids).all():
        raise ValueError(f"{path.name}: test set in a different order")
    auc[s] = bootstrap_metrics(y, p["prob"].to_numpy(), float(info["Youden threshold (train)"]),
                               N_BOOT_CI, SEED)["AUROC"]
    algorithm[s] = info["Algorithm"]
    print(f"{s}: best model {algorithm[s]}, test AUROC {float(info['Test AUROC']):.4f}")

print("\nPaired difference in test AUROC (95% CI)")
for a, b in PAIRS:
    full = pd.read_excel(c.BENCHMARK_DIR / f"best_{SETS[a]}_predictions.xlsx", sheet_name="Model")\
        .set_index("Field")["Value"]["Test AUROC"]
    ref = pd.read_excel(c.BENCHMARK_DIR / f"best_{SETS[b]}_predictions.xlsx", sheet_name="Model")\
        .set_index("Field")["Value"]["Test AUROC"]
    lo, hi = np.nanpercentile(auc[a] - auc[b], [2.5, 97.5])
    print(f"{a} - {b}: {float(full) - float(ref):.4f} ({lo:.4f} to {hi:.4f})")

pivot = pd.read_excel(c.BENCHMARK, sheet_name="AUROC pivot").set_index("Algorithm")
print("\nShare of the gain over BP obtained with all laboratory predictors (same algorithm)")
for alg in pivot.index:
    g = pivot.loc[alg]
    print(f"{alg}: BP+Lasso {100 * (g['BP+Lasso'] - g['BP']) / (g['BP+All'] - g['BP']):.1f}% | "
          f"BP+SHAP {100 * (g['BP+SHAP'] - g['BP']) / (g['BP+All'] - g['BP']):.1f}%")
