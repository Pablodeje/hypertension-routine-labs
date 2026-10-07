"""
Minimalist model for the demonstration web page (post hoc, outside the main analysis).

Blood pressure predictors plus the first MINIMAL_K laboratory predictors of the SHAP ranking
(12_catboost_shap.py). With MINIMAL_K = 4 all four come from the basic metabolic panel, so the model
needs no blood count. The value 4 was chosen after seeing the ranking, for that practical reason.

Same procedure as 20_benchmark.py (same split, 5 algorithms, grids, tuning, metrics and bootstrap,
from benchmark_common.py). The best algorithm is chosen by cross-validated AUC in the training set
and saved like the others (predictions + .joblib + .onnx). Its test AUROC is compared with the best
blood-pressure-only model of the benchmark on the same bootstrap resamples (paired difference).
Run after 20_benchmark.py.
"""
import traceback
import warnings

import numpy as np
import pandas as pd

import config as c
from benchmark_common import (BEST_CRITERION, N_BOOT_CI, ORDER, SEED, SET_BP, bootstrap_metrics,
                              fit_evaluate, load_split, log, save_best)

warnings.filterwarnings("ignore", category=FutureWarning)

MINIMAL_K = 4
SET_NAME, SLUG = "BP+Minimal", "BP_Minimal"

shap = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")   # sorted by mean |SHAP|
MINIMAL = shap["Variable"].head(MINIMAL_K).tolist()
PAPER_MINIMAL = ["Urea Nitrogen_max", "Chloride_min", "Glucose_max", "Potassium_min"]   # as in the paper
if MINIMAL != PAPER_MINIMAL:
    log(f"WARNING: the minimal set differs from the paper: {MINIMAL} vs {PAPER_MINIMAL}")
cols = MINIMAL + SET_BP
log(f"{SET_NAME}: {MINIMAL} + {len(SET_BP)} blood pressure predictors")

X, y, X_tr, X_te, y_tr, y_te = load_split()

# ---------------- 5 algorithms ----------------
rows, configs, boots = [], [], {}
best_value = -np.inf
for algorithm in ORDER:
    log(f"  {algorithm} / {SET_NAME} ({len(cols)} predictors) ...")
    try:
        row, point, boot, model, prob_tr, prob, threshold, cvr = \
            fit_evaluate(algorithm, SET_NAME, cols, X_tr, y_tr, X_te, y_te)
        rows.append(row)
        configs.append(cvr.assign(Algorithm=algorithm))
        boots[algorithm] = boot
        if row[BEST_CRITERION] > best_value:
            best_value = row[BEST_CRITERION]
            save_best(SLUG, SET_NAME, algorithm, model, cols, X_tr, y_tr, X_te, y_te,
                      prob_tr, prob, threshold, row)
    except Exception as e:
        log(f"    FAILED: {e}")
        traceback.print_exc()

table = pd.DataFrame(rows)
best = table.loc[table[BEST_CRITERION].idxmax()]

# ---------------- Paired difference against the best blood-pressure-only model ----------------
bp = pd.read_excel(c.BENCHMARK_DIR / "best_BP_predictions.xlsx", sheet_name="Predictions")
bp = bp[bp["Partition"] == "test"]
bp_info = pd.read_excel(c.BENCHMARK_DIR / "best_BP_predictions.xlsx", sheet_name="Model").set_index("Field")["Value"]
if not (bp["subject_id"].to_numpy() == X_te.index.to_numpy()).all():
    raise ValueError("The test set of best_BP_predictions.xlsx does not match the current split")
y_te_arr = y_te.to_numpy()
boot_bp = bootstrap_metrics(y_te_arr, bp["prob"].to_numpy(), float(bp_info["Youden threshold (train)"]),
                            N_BOOT_CI, SEED)
d = boots[best["Algorithm"]]["AUROC"] - boot_bp["AUROC"]
diff = best["AUROC"] - float(bp_info["Test AUROC"])
diff_lo, diff_hi = np.nanpercentile(d, [2.5, 97.5])

with pd.ExcelWriter(c.BENCHMARK_DIR / "results_minimal.xlsx") as w:
    table.sort_values("AUROC", ascending=False).to_excel(w, sheet_name="Best", index=False)
    pd.concat(configs, ignore_index=True).to_excel(w, sheet_name="All configs", index=False)
    pd.DataFrame([{"Minimal": best["Algorithm"], "Reference": f"BP ({bp_info['Algorithm']})",
                   "diff AUROC": diff, "diff AUROC CI95 low": diff_lo, "diff AUROC CI95 high": diff_hi}])\
        .to_excel(w, sheet_name="Paired gain", index=False)

print(f"\nLaboratory predictors: {', '.join(MINIMAL)}")
print(table[["Algorithm", "CV AUC (train)", "AUROC", "AUROC CI95 low", "AUROC CI95 high",
             "Brier", "Calibration slope"]].round(4).to_string(index=False))
print(f"\nBest by {BEST_CRITERION}: {best['Algorithm']} | test AUROC {best['AUROC']:.3f} "
      f"({best['AUROC CI95 low']:.3f}–{best['AUROC CI95 high']:.3f})")
print(f"Difference vs BP alone ({bp_info['Algorithm']}, AUROC {float(bp_info['Test AUROC']):.3f}): "
      f"{diff:.3f} ({diff_lo:.3f} to {diff_hi:.3f})")
