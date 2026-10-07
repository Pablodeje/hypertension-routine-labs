"""
Shared settings and functions of the benchmark (20_benchmark.py) and of the minimalist model (50_minimal_model.py),
so that both use exactly the same split, algorithms, grids, metrics, bootstrap and model export.
"""
import time
from datetime import datetime

import joblib
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

# ---------------- Data and split ----------------
def load_split():
    """Same order of steps as in the selection stage, to reproduce the split exactly."""
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
    if not X.index.is_unique:
        log(f"WARNING: {len(X) - X.index.nunique()} duplicated subject_id")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              stratify=y, random_state=SPLIT_SEED)
    log(f"Train {len(X_tr)} | Test {len(X_te)} | prevalence {y.mean():.4f}")
    return X, y, X_tr, X_te, y_tr, y_te


CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)


# ---------------- Tuning and evaluation of one algorithm on one predictor set ----------------
def fit_evaluate(algorithm, set_name, cols, X_tr, y_tr, X_te, y_te):
    cfg = ALGORITHMS[algorithm]
    n_combos = int(np.prod([len(v) for v in cfg["grid"].values()]))
    n_iter = min(cfg["n_iter"], n_combos)
    t0 = time.time()
    y_te_arr = y_te.to_numpy()

    steps = ([("sc", StandardScaler())] if cfg["scale"] else []) \
            + [("est", cfg["estimator"])]
    rs = RandomizedSearchCV(Pipeline(steps), cfg["grid"], n_iter=n_iter,
                            cv=CV, scoring="roc_auc", n_jobs=-1,
                            random_state=0, refit=True)
    rs.fit(X_tr[cols], y_tr)

    prob_tr = cross_val_predict(rs.best_estimator_, X_tr[cols], y_tr, cv=CV,
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

    cvr = pd.DataFrame(rs.cv_results_)[["params", "mean_test_score",
                                        "std_test_score", "rank_test_score"]]
    cvr.insert(0, "Set", set_name)
    cvr["params"] = cvr["params"].astype(str)

    log(f"    AUROC {row['AUROC']:.4f} [{row['AUROC CI95 low']:.4f}-"
        f"{row['AUROC CI95 high']:.4f}] | Brier {row['Brier']:.4f} | "
        f"slope {row['Calibration slope']:.3f} | "
        f"{(time.time()-t0)/60:.1f} min")
    log(f"    {rs.best_params_}")
    return row, point, boot, rs.best_estimator_, prob_tr, prob, threshold, cvr


# ---------------- Saving the best model of a set: predictions, .joblib and .onnx ----------------
def export_onnx(model, algorithm, n_vars, path):
    if algorithm == "CatBoost":
        model.named_steps["est"].save_model(str(path), format="onnx")
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(model,
                              initial_types=[("input", FloatTensorType([None, n_vars]))],
                              options={id(model.named_steps["est"]): {"zipmap": False}})
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())


def onnx_difference(path, X, prob_ref):
    """Max |prob ONNX - prob Python| on the test set (ONNX works in float32)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    out = sess.run(None, {sess.get_inputs()[0].name: X.to_numpy(np.float32)})[1]
    p = np.array([o[1] for o in out]) if isinstance(out, list) else np.asarray(out)[:, 1]
    return float(np.max(np.abs(p - prob_ref)))


def save_best(slug, set_name, algorithm, model, cols, X_tr, y_tr, X_te, y_te,
              prob_tr, prob_te, threshold, row):
    base = c.BENCHMARK_DIR / f"best_{slug}"

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

    # Model: the previous one is deleted first, so that no .onnx of another algorithm is left behind
    for ext in (".joblib", ".onnx"):
        base.with_suffix(ext).unlink(missing_ok=True)
    joblib.dump(model, base.with_suffix(".joblib"))
    try:
        export_onnx(model, algorithm, len(cols), base.with_suffix(".onnx"))
        dif = onnx_difference(base.with_suffix(".onnx"), X_te[cols], prob_te)
        info.loc[len(info)] = {"Field": "Max diff. prob ONNX vs Python (test)", "Value": dif}
        log(f"    ONNX saved (max diff. prob {dif:.2e})")
    except Exception as e:
        info.loc[len(info)] = {"Field": "ONNX", "Value": f"FAILED: {e}"}
        log(f"    ONNX FAILED: {e}  (the .joblib is kept)")

    path = c.BENCHMARK_DIR / f"best_{slug}_predictions.xlsx"
    with pd.ExcelWriter(path) as w:
        pred.to_excel(w, sheet_name="Predictions", index=False)
        info.to_excel(w, sheet_name="Model", index=False)
        pd.DataFrame({"Input order": range(len(cols)), "Variable": cols})\
            .to_excel(w, sheet_name="Predictors", index=False)
    log(f"    New best for '{set_name}' -> {path}")
