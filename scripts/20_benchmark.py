"""
Full benchmark: 5 algorithms x 4 predictor sets (long run, several hours).

Robustness:
  - saves results_<ALGORITHM>.xlsx as soon as each algorithm finishes
  - if an algorithm fails, the error is logged and the run continues with the next one
  - the predictions of the best model of each set are overwritten whenever a better one appears
"""
import time
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (RandomizedSearchCV, train_test_split,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

import config as c

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Settings ----------------
OTHER_LABELS = ["Hypertension A", "Hypertension B"]

TEST_SIZE, SPLIT_SEED = 0.25, 0    # IDENTICAL to the selection stage: the test set was not used for selection
N_FOLDS = 5
N_BOOT_CI = 2000
SEED = 42

# Criterion to choose the best model of each set (the one whose predictions are saved)
# "CV AUC (train)" -> does not use the test set | "AUROC" -> test AUROC
BEST_CRITERION = "CV AUC (train)"

ORDER = ["LogisticRegression", "kNN", "MLP", "RandomForest", "CatBoost"]

# ---------------- Predictor sets ----------------
SET_BP = [
    "bp_s_mean", "bp_s_sd", "bp_s_q1", "bp_s_q3", "bp_s_min", "bp_s_max",
    "bp_d_mean", "bp_d_sd", "bp_d_q1", "bp_d_q3", "bp_d_min", "bp_d_max",
]

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
    "Creatinine_min", "Glucose_mean", "Urea Nitrogen_q1", "Potassium_min", "RDW_min",
    "Potassium_max", "Chloride_min", "Urea Nitrogen_min", "Anion GAP_sd", "White Blood Cells_min",
    "Bicarbonate_max", "Anion GAP_min", "Urea Nitrogen_max", "Chloride_q1", "Sodium_sd",
    "Anion GAP_q3", "Sodium_min", "Glucose_sd", "Sodium_max", "Potassium_sd",
    "Hemoglobin_sd", "Anion GAP_max", "Bicarbonate_sd", "Chloride_max", "Platelet Count_sd",
    "RDW_max", "Sodium_q3", "Potassium_q3", "Potassium_q1", "Bicarbonate_min",
]
PAPER_SHAP = [
    "Urea Nitrogen_max", "Glucose_max", "Chloride_min", "Potassium_min", "RDW_min",
    "White Blood Cells_min", "Urea Nitrogen_mean", "Urea Nitrogen_q3", "Glucose_q1", "Glucose_mean",
    "Creatinine_max", "White Blood Cells_sd", "Sodium_max", "Potassium_max", "Creatinine_mean",
    "Platelet Count_sd", "Creatinine_q1", "Glucose_q3", "Bicarbonate_max", "Chloride_q1",
    "Urea Nitrogen_q1", "Chloride_max", "Urea Nitrogen_sd", "Creatinine_q3", "Glucose_sd",
    "Hemoglobin_sd", "Anion GAP_q3", "Chloride_mean", "Sodium_sd", "Platelet Count_min",
    "Anion GAP_sd", "Creatinine_sd", "Creatinine_min", "Bicarbonate_sd", "Potassium_sd",
    "Chloride_sd", "Potassium_mean", "Potassium_q1", "Platelet Count_q3", "RDW_sd",
    "RDW_q1", "White Blood Cells_q3", "Red Blood Cells_min", "Anion GAP_max", "Platelet Count_q1",
    "RDW_mean", "MCV_sd", "MCHC_sd", "Platelet Count_mean", "Anion GAP_min",
    "Anion GAP_mean", "Platelet Count_max", "Sodium_mean", "Chloride_q3", "Glucose_min",
    "Hematocrit_q1", "MCHC_q1", "Sodium_min", "MCHC_mean", "MCHC_min",
    "Hemoglobin_q1", "Red Blood Cells_q1", "RDW_q3", "Sodium_q3", "White Blood Cells_q1",
    "Hemoglobin_min", "Urea Nitrogen_min", "Bicarbonate_mean", "Anion GAP_q1", "Hemoglobin_mean",
]

# The first set is the reference for the gain over blood pressure
FEATURE_SETS = {
    "BP": SET_BP,
    "BP+Lasso": SET_LASSO,
    "BP+SHAP": SET_SHAP,
    "BP+All": SET_ALL,
}
SLUG = {"BP": "BP", "BP+Lasso": "BP_Lasso", "BP+SHAP": "BP_SHAP", "BP+All": "BP_All"}

# ---------------- Algorithms and grids ----------------
ALGORITHMS = {
    "CatBoost": dict(
        # fixed iterations: without eval_set, early stopping does not act inside RandomizedSearchCV
        estimator=CatBoostClassifier(loss_function="Logloss", verbose=False,
                                     random_state=SEED, iterations=2000, thread_count=1),
        grid={
            "est__depth": [4, 5, 6, 8, 10],
            "est__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "est__l2_leaf_reg": [1, 3, 5, 7],
            "est__subsample": [0.6, 0.8, 1.0],
            "est__colsample_bylevel": [0.6, 0.8, 1.0],
            "est__border_count": [32, 128, 254],
        },
        scale=False,
        n_iter=100,
    ),
    "RandomForest": dict(
        estimator=RandomForestClassifier(random_state=SEED, n_jobs=1),
        grid={
            "est__n_estimators": [200, 400, 600, 800],
            "est__max_depth": [10, 15, 20, 30, None],
            "est__max_features": ["sqrt", "log2", 0.5],
            "est__min_samples_leaf": [1, 2, 4, 10],
            "est__min_samples_split": [2, 5, 10],
            "est__criterion": ["gini", "entropy"],
            "est__max_samples": [None, 0.8],
        },
        scale=False,
        n_iter=100,
    ),
    "MLP": dict(
        estimator=MLPClassifier(max_iter=1000, random_state=SEED, early_stopping=True),
        grid={
            "est__hidden_layer_sizes": [(50,), (100,), (200,), (100, 50),
                                        (150, 100, 50), (200, 100)],
            "est__activation": ["relu", "tanh"],
            "est__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "est__learning_rate_init": [1e-4, 5e-4, 1e-3, 1e-2],
            "est__learning_rate": ["constant", "adaptive"],
        },
        scale=True,
        n_iter=150,
    ),
    "kNN": dict(
        estimator=KNeighborsClassifier(n_jobs=1),
        grid={
            "est__n_neighbors": [5, 11, 25, 51, 75, 101, 151, 201, 301],
            "est__weights": ["uniform", "distance"],
            "est__metric": ["euclidean", "manhattan", "chebyshev"],
        },
        scale=True,
        n_iter=54,
    ),
    "LogisticRegression": dict(
        # penalty="elasticnet" is needed for l1_ratio to have an effect
        estimator=LogisticRegression(penalty="elasticnet", max_iter=5000, solver="saga",
                                     random_state=SEED),
        grid={
            "est__C": np.logspace(-4, 3, 40),
            "est__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        },
        scale=True,
        n_iter=200,
    ),
}

# ---------------- Metrics ----------------
def divide(a, b):
    return a / b if b else np.nan


def calibration_slope(y, prob, eps=1e-6, n_iter=50):
    """Unpenalized logistic regression of y on logit(prob) (Newton-Raphson)."""
    p = np.clip(prob, eps, 1 - eps)
    A = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
    b = np.array([0.0, 1.0])
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-A @ b))
        H = A.T @ (A * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, A.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return float(b[1])


def threshold_metrics(y, prob, threshold):
    pred = prob >= threshold
    pos = y == 1
    tp = np.sum(pred & pos); fp = np.sum(pred & ~pos)
    fn = np.sum(~pred & pos); tn = y.size - tp - fp - fn
    n = y.size
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return {
        "Accuracy": po,
        "Kappa": divide(po - pe, 1 - pe),
        "Sensitivity": divide(tp, tp + fn),
        "Specificity": divide(tn, tn + fp),
        "PPV": divide(tp, tp + fp),
        "NPV": divide(tn, tn + fn),
    }


def metrics(y, prob, threshold):
    m = {
        "AUROC": roc_auc_score(y, prob),
        "AUPRC": average_precision_score(y, prob),
        "Brier": float(np.mean((prob - y) ** 2)),
        "Calibration slope": calibration_slope(y, prob),
        "Calibration-in-the-large": float(np.mean(y) - np.mean(prob)),
    }
    for label, t in (("0.5", 0.5), ("Youden", threshold)):
        m.update({f"{k} ({label})": v for k, v in threshold_metrics(y, prob, t).items()})
    return m


def bootstrap_metrics(y, prob, threshold, n_boot, seed):
    """Stratified bootstrap distribution of all metrics.
    Same seed + same y -> same resamples for every model, so differences between models are paired."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for _ in range(n_boot):
        s = np.concatenate([rng.choice(ip, ip.size, replace=True),
                            rng.choice(ineg, ineg.size, replace=True)])
        rows.append(metrics(y[s], prob[s], threshold))
    return pd.DataFrame(rows)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ---------------- Save the predictions of the best model ----------------
def save_best(set_name, algorithm, cols, prob_tr, prob_te, threshold, row):
    path = c.BENCHMARK_DIR / f"best_{SLUG[set_name]}_predictions.xlsx"

    pred = pd.concat([
        pd.DataFrame({"subject_id": X_tr.index, "Partition": "train (OOF)",
                      "y_true": y_tr.values, "prob": prob_tr}),
        pd.DataFrame({"subject_id": X_te.index, "Partition": "test",
                      "y_true": y_te.values, "prob": prob_te}),
    ], ignore_index=True)
    pred["pred_0.5"] = (pred["prob"] >= 0.5).astype(int)
    pred["pred_Youden"] = (pred["prob"] >= threshold).astype(int)

    info = pd.DataFrame([
        {"Field": "Set", "Value": set_name},
        {"Field": "Algorithm", "Value": algorithm},
        {"Field": "Choice criterion", "Value": BEST_CRITERION},
        {"Field": "CV AUC (train)", "Value": row["CV AUC (train)"]},
        {"Field": "Test AUROC", "Value": row["AUROC"]},
        {"Field": "Youden threshold (train)", "Value": threshold},
        {"Field": "Best parameters", "Value": row["Best parameters"]},
    ])

    with pd.ExcelWriter(path) as w:
        pred.to_excel(w, sheet_name="Predictions", index=False)
        info.to_excel(w, sheet_name="Model", index=False)
        pd.DataFrame({"Input order": range(len(cols)), "Variable": cols})\
            .to_excel(w, sheet_name="Predictors", index=False)
    log(f"    New best for '{set_name}' -> {path}")

# ---------------- Load ----------------
# Same order of steps as in the selection stage, to reproduce the split exactly
T0 = time.time()
log(f"Loading {c.COHORT_CLEAN}")
df = pd.read_excel(c.COHORT_CLEAN, engine="openpyxl")
n_total = len(df)
df = df[df["Admission"].astype(str).str.strip() == "YES"].drop(columns="Admission")
log(f"Admission == YES: {len(df)} of {n_total} subjects")
df = df.set_index("subject_id")
df = df.drop(columns=[col for col in df.columns if str(col).endswith("_variance")])

y = df.pop(c.TARGET).astype(int)
X = df.drop(columns=[col for col in OTHER_LABELS if col in df.columns]).astype(float)
mask = X.isna().any(axis=1) | y.isna()
X, y = X[~mask], y[~mask]

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

if not X.index.is_unique:
    log(f"WARNING: {len(X) - X.index.nunique()} duplicated subject_id")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          stratify=y, random_state=SPLIT_SEED)
y_te_arr = y_te.to_numpy()
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
log(f"Train {len(X_tr)} | Test {len(X_te)} | prevalence {y.mean():.4f}")
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
        t0 = time.time()
        log(f"  {algorithm} / {set_name} ({len(cols)} predictors) ...")
        try:
            steps = ([("sc", StandardScaler())] if cfg["scale"] else []) \
                    + [("est", cfg["estimator"])]
            rs = RandomizedSearchCV(Pipeline(steps), cfg["grid"], n_iter=n_iter,
                                    cv=cv, scoring="roc_auc", n_jobs=-1,
                                    random_state=0, refit=True)
            rs.fit(X_tr[cols], y_tr)

            prob_tr = cross_val_predict(rs.best_estimator_, X_tr[cols], y_tr, cv=cv,
                                        method="predict_proba", n_jobs=-1)[:, 1]
            fpr, tpr, thr = roc_curve(y_tr, prob_tr)
            threshold = float(thr[int(np.argmax(tpr - fpr))])

            prob = rs.best_estimator_.predict_proba(X_te[cols])[:, 1]
            point = metrics(y_te_arr, prob, threshold)
            boot = bootstrap_metrics(y_te_arr, prob, threshold, N_BOOT_CI, SEED)

            row = {"Algorithm": algorithm, "Set": set_name, "N predictors": len(cols)}
            for k, v in point.items():
                row[k] = v
                row[f"{k} CI95 low"], row[f"{k} CI95 high"] = np.nanpercentile(boot[k], [2.5, 97.5])
            row.update({
                "Youden threshold (train)": round(threshold, 4),
                "CV AUC (train)": rs.best_score_,
                "Best parameters": str(rs.best_params_),
                "Minutes": round((time.time() - t0) / 60, 1),
            })
            rows.append(row)
            models[(algorithm, set_name)] = (point, boot)

            cvr = pd.DataFrame(rs.cv_results_)[["params", "mean_test_score",
                                                "std_test_score", "rank_test_score"]]
            cvr.insert(0, "Set", set_name)
            cvr["params"] = cvr["params"].astype(str)
            configs.append(cvr)

            log(f"    AUROC {row['AUROC']:.4f} [{row['AUROC CI95 low']:.4f}-"
                f"{row['AUROC CI95 high']:.4f}] | Brier {row['Brier']:.4f} | "
                f"slope {row['Calibration slope']:.3f} | "
                f"{(time.time()-t0)/60:.1f} min")
            log(f"    {rs.best_params_}")

            if row[BEST_CRITERION] > best_value.get(set_name, -np.inf):
                best_value[set_name] = row[BEST_CRITERION]
                save_best(set_name, algorithm, cols, prob_tr, prob, threshold, row)
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
