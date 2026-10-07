"""
Full benchmark: 5 algorithms x 4 predictor sets (long run, several hours).
Settings, algorithms, metrics and model export are in benchmark_common.py (shared with 50_minimal_model.py).

Robustness:
  - saves results_<ALGORITHM>.xlsx as soon as each algorithm finishes
  - if an algorithm fails, the error is logged and the run continues with the next one
  - the best model of each set (predictions + .joblib + .onnx) is overwritten whenever a better one appears
"""
import time
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as c
from benchmark_common import (ALGORITHMS, BEST_CRITERION, ORDER, SET_BP, TEST_SIZE, SPLIT_SEED,
                              N_FOLDS, N_BOOT_CI, SEED, log, load_split, fit_evaluate, save_best)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Predictor sets ----------------
# Selected analytes, in the order of each selection output (lasso: frequency then |coefficient|; SHAP: rank)
lasso = pd.read_excel(c.SELECTION_LASSO)
SET_LASSO = lasso.loc[lasso["Selected"], "Variable"].tolist() + SET_BP
shap = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")
SET_SHAP = shap.loc[shap["Selected"], "Variable"].tolist() + SET_BP

ANALYTES = [
    "Anion GAP", "Bicarbonate", "Chloride", "Creatinine", "Glucose", "Hematocrit",
    "Hemoglobin", "Potassium", "Sodium", "Urea Nitrogen", "MCH", "MCHC", "MCV",
    "Platelet Count", "RDW", "Red Blood Cells", "White Blood Cells",
]
STATISTICS = ["mean", "sd", "min", "max", "q1", "q3"]   # without _variance, as in the selection stage
SET_ALL = [f"{a}_{s}" for a in ANALYTES for s in STATISTICS] + SET_BP

# Sets reported in the paper (Supplementary Table 7), to warn if a rerun selects different ones
PAPER_LASSO = [
    "Glucose_mean", "Creatinine_min", "Urea Nitrogen_q1", "Potassium_min", "RDW_min",
    "Chloride_min", "Potassium_max", "Anion GAP_min", "Urea Nitrogen_max", "White Blood Cells_min",
    "Urea Nitrogen_min", "Bicarbonate_max", "Anion GAP_sd", "Glucose_sd", "Chloride_q1",
    "Sodium_sd", "Potassium_sd", "Sodium_min", "Anion GAP_q3", "Bicarbonate_sd",
    "Hemoglobin_sd", "Chloride_max", "Sodium_max", "Sodium_q3", "Potassium_q1",
    "Anion GAP_max", "Bicarbonate_min", "Platelet Count_sd", "Potassium_q3", "RDW_max",
    "MCV_min", "MCV_sd", "Chloride_sd", "MCH_sd",
]
PAPER_SHAP = [
    "Urea Nitrogen_max 0.248697", "Chloride_min 0.172505", "Glucose_max 0.153176", "Potassium_min 0.132754", "White Blood Cells_min 0.125205",
    "RDW_min 0.123383", "Glucose_q1 0.113458", "Creatinine_max 0.110627", "Urea Nitrogen_q3 0.095103", "Glucose_mean 0.087899",
    "Urea Nitrogen_mean 0.086254", "Creatinine_q1 0.079192", "Glucose_q3 0.079166", "White Blood Cells_sd 0.078551", "Potassium_max 0.078396",
    "Sodium_max 0.071133", "Urea Nitrogen_q1 0.068512", "Platelet Count_sd 0.062185", "Chloride_q1 0.060016", "Creatinine_mean 0.059544",
    "Creatinine_q3 0.055980", "Chloride_max 0.048828", "Bicarbonate_max 0.048591", "Platelet Count_min 0.046860", "Urea Nitrogen_sd 0.041643",
    "Anion GAP_q3 0.041306", "Glucose_sd 0.038037", "Potassium_sd 0.037040", "Potassium_q1 0.034841", "Anion GAP_sd 0.034539",
    "Anion GAP_max 0.034279", "Chloride_sd 0.031564", "Chloride_mean 0.030731", "Bicarbonate_sd 0.030083", "Creatinine_sd 0.029518",
    "Creatinine_min 0.026832", "Sodium_sd 0.026202", "Potassium_mean 0.025975", "Platelet Count_q3 0.024865", "Platelet Count_q1 0.024029",
    "Anion GAP_min 0.022762", "RDW_q1 0.022376", "MCV_sd 0.021283", "Bicarbonate_min 0.020991", "Red Blood Cells_min 0.020430",
    "MCHC_sd 0.020213", "RDW_sd 0.019906", "Platelet Count_max 0.018691", "Red Blood Cells_q1 0.017965", "Hemoglobin_sd 0.017433",
    "Glucose_min 0.017429", "Sodium_q3 0.017329", "Hematocrit_q1 0.016795", "White Blood Cells_q1 0.016690", "Urea Nitrogen_min 0.015570",
    "RDW_mean 0.015353", "Hemoglobin_q1 0.014962", "Platelet Count_mean 0.014718", "White Blood Cells_q3 0.014446", "Hemoglobin_min 0.014344",
    "MCHC_q1 0.013946", "Anion GAP_mean 0.012709", "Chloride_q3 0.011995", "White Blood Cells_mean 0.011464", "Bicarbonate_mean 0.011415",
    "Sodium_mean 0.011191", "Hemoglobin_mean 0.010946", "Potassium_q3 0.010822", "RDW_max 0.010716", "MCHC_min 0.010684",
]

# The first set is the reference for the gain over blood pressure
FEATURE_SETS = {
    "BP": SET_BP,
    "BP+Lasso": SET_LASSO,
    "BP+SHAP": SET_SHAP,
    "BP+All": SET_ALL,
}
SLUG = {"BP": "BP", "BP+Lasso": "BP_Lasso", "BP+SHAP": "BP_SHAP", "BP+All": "BP_All"}

# ---------------- Load ----------------
T0 = time.time()
X, y, X_tr, X_te, y_tr, y_te = load_split()

for n, cols in FEATURE_SETS.items():
    missing = [col for col in cols if col not in X.columns]
    if missing:
        raise KeyError(f"Variables not found in '{n}': {missing}")
for n, read, paper in (("BP+Lasso", SET_LASSO[:-len(SET_BP)], PAPER_LASSO),
                       ("BP+SHAP", SET_SHAP[:-len(SET_BP)], PAPER_SHAP)):
    if read != paper:
        log(f"WARNING: '{n}' differs from the paper: {len(read)} vs {len(paper)} analytes | "
            f"only here: {sorted(set(read) - set(paper))} | only in the paper: {sorted(set(paper) - set(read))}"
            f"{' | same variables, different order' if set(read) == set(paper) else ''}")

for n, cols in FEATURE_SETS.items():
    log(f"  {n}: {len(cols)} predictors")

total_combos = sum(int(np.prod([len(v) for v in a["grid"].values()]))
                   for a in ALGORITHMS.values())
total_fits = sum(min(a["n_iter"], int(np.prod([len(v) for v in a["grid"].values()])))
                 for a in ALGORITHMS.values()) * len(FEATURE_SETS) * N_FOLDS
log(f"Total grid: {total_combos} combinations")
log(f"Planned fits: {total_fits} ({len(ALGORITHMS)} algorithms x "
    f"{len(FEATURE_SETS)} sets x {N_FOLDS} folds)\n")

# ---------------- Main loop ----------------
everything, failures = [], []
models = {}       # (algorithm, set) -> (point metrics, bootstrap distribution)
best_value = {}   # set -> criterion value of the best saved model

for algorithm in ORDER:
    cfg = ALGORITHMS[algorithm]
    n_combos = int(np.prod([len(v) for v in cfg["grid"].values()]))
    n_iter = min(cfg["n_iter"], n_combos)
    t_alg = time.time()
    log("=" * 64)
    log(f"{algorithm}: {n_combos} combinations, {n_iter} sampled per set")
    log("=" * 64)

    rows, configs = [], []
    for set_name, cols in FEATURE_SETS.items():
        log(f"  {algorithm} / {set_name} ({len(cols)} predictors) ...")
        try:
            row, point, boot, model, prob_tr, prob, threshold, cvr = \
                fit_evaluate(algorithm, set_name, cols, X_tr, y_tr, X_te, y_te)
            rows.append(row)
            models[(algorithm, set_name)] = (point, boot)
            configs.append(cvr)

            if row[BEST_CRITERION] > best_value.get(set_name, -np.inf):
                best_value[set_name] = row[BEST_CRITERION]
                save_best(SLUG[set_name], set_name, algorithm, model, cols, X_tr, y_tr, X_te, y_te,
                          prob_tr, prob, threshold, row)
        except Exception as e:
            log(f"    FAILED: {e}")
            traceback.print_exc()
            failures.append({"Algorithm": algorithm, "Set": set_name, "Error": str(e)})

    # ---- Incremental save per algorithm ----
    if rows:
        out = c.BENCHMARK_DIR / f"results_{algorithm}.xlsx"
        meta = pd.DataFrame([
            {"Field": "Algorithm", "Value": algorithm},
            {"Field": "Target", "Value": c.TARGET},
            {"Field": "Data file", "Value": str(c.COHORT_CLEAN)},
            {"Field": "Filter", "Value": "Admission == YES; no _variance columns"},
            {"Field": "Total rows", "Value": len(X)},
            {"Field": "Train / Test", "Value": f"{len(X_tr)} / {len(X_te)}"},
            {"Field": "test_size / random_state", "Value": f"{TEST_SIZE} / {SPLIT_SEED}"},
            {"Field": "CV folds", "Value": N_FOLDS},
            {"Field": "Selection metric", "Value": "roc_auc"},
            {"Field": "Grid combinations", "Value": n_combos},
            {"Field": "Sampled configurations", "Value": n_iter},
            {"Field": "Bootstrap CI", "Value": f"{N_BOOT_CI} stratified resamples, "
                                                f"95% percentile, identical for all models"},
            {"Field": "CatBoost iterations", "Value": "fixed at 2000, no early stopping"},
            {"Field": "Threshold", "Value": "Youden, derived in train by CV"},
            {"Field": "Estimator seed", "Value": SEED},
            {"Field": "Algorithm minutes", "Value": round((time.time()-t_alg)/60, 1)},
            {"Field": "Date", "Value": time.strftime("%Y-%m-%d %H:%M")},
        ])
        with pd.ExcelWriter(out) as w:
            pd.DataFrame(rows).sort_values("AUROC", ascending=False)\
                .to_excel(w, sheet_name="Best", index=False)
            pd.concat(configs, ignore_index=True)\
                .to_excel(w, sheet_name="All configs", index=False)
            meta.to_excel(w, sheet_name="Run config", index=False)
        everything.extend(rows)
        log(f"  Saved {out}  ({(time.time()-t_alg)/60:.1f} min)")

    elapsed = time.time() - T0
    log(f"  Cumulative: {elapsed/3600:.2f} h  "
        f"(estimated end {datetime.now() + timedelta(seconds=elapsed * (len(ORDER)/(ORDER.index(algorithm)+1) - 1)):%H:%M})\n")

# ---------------- Consolidated ----------------
if everything:
    table = pd.DataFrame(everything)
    set_order = list(FEATURE_SETS)
    table["Set"] = pd.Categorical(table["Set"], categories=set_order, ordered=True)
    table = table.sort_values(["Set", "AUROC"], ascending=[True, False])

    pivot = table.pivot_table(index="Algorithm", columns="Set",
                              values="AUROC", observed=True).round(4)
    gain = pivot.sub(pivot[set_order[0]], axis=0).round(4).drop(columns=set_order[0])

    best = table.loc[table.groupby("Set", observed=True)[BEST_CRITERION].idxmax(),
                     ["Set", "Algorithm", "CV AUC (train)", "AUROC",
                      "AUROC CI95 low", "AUROC CI95 high", "Best parameters"]]

    # Paired differences against BP, same algorithm, same resamples
    paired = []
    ref = set_order[0]
    for (alg, s), (p1, b1) in models.items():
        if s == ref or (alg, ref) not in models:
            continue
        p0, b0 = models[(alg, ref)]
        d = b1 - b0
        f = {"Algorithm": alg, "Set": s, "Reference": ref}
        for k in d.columns:
            f[f"diff {k}"] = p1[k] - p0[k]
            f[f"diff {k} CI95 low"], f[f"diff {k} CI95 high"] = np.nanpercentile(d[k], [2.5, 97.5])
        paired.append(f)
    paired = pd.DataFrame(paired)

    with pd.ExcelWriter(c.BENCHMARK) as w:
        table.to_excel(w, sheet_name="Full", index=False)
        pivot.to_excel(w, sheet_name="AUROC pivot")
        gain.to_excel(w, sheet_name="Gain over BP")
        paired.to_excel(w, sheet_name="Paired gain", index=False)
        best.to_excel(w, sheet_name="Best by set", index=False)
        if failures:
            pd.DataFrame(failures).to_excel(w, sheet_name="Failures", index=False)

    log("=" * 64)
    log(f"FINISHED in {(time.time()-T0)/3600:.2f} h")
    log("=" * 64)
    print(pivot.to_string())
    print(f"\n--- Gain over '{set_order[0]}' ---")
    print(gain.to_string())
    print(f"\n--- Best model by set (criterion: {BEST_CRITERION}) ---")
    print(best.to_string(index=False))
    print(f"\nSaved to {c.BENCHMARK}")

if failures:
    log(f"\n{len(failures)} combinations failed:")
    for f in failures:
        print(f"  {f['Algorithm']} / {f['Set']}: {f['Error']}")
