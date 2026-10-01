import itertools
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

import config as c

# BP+SHAP: per-fold SHAP ranking of analytes (CatBoost) and cross-validated curve over k

# ---------------- Settings ----------------
TEST_SIZE, SPLIT_SEED = 0.25, 0
N_FOLDS = 5
ES_SIZE = 0.15   # fraction of each fold's training data held out for early stopping (and for SHAP)

GRID = {"depth": [6, 8], "learning_rate": [0.05]}
FIXED = dict(iterations=2000, l2_leaf_reg=3, eval_metric="Logloss", early_stopping_rounds=100,
             verbose=0, task_type="CPU", random_seed=0)

K_VALUES = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]   # analytes added to blood pressure

# ---------------- Load ----------------
df = pd.read_excel(c.COHORT_CLEAN).set_index("subject_id")
df = df[df["Admission"].astype(str).str.strip() == "YES"]
df = df.drop(columns=[col for col in df.columns if col.endswith("_variance")])

X = df.drop(columns=[c.TARGET] + c.NON_PREDICTORS).astype(float)
mask = X.isna().any(axis=1) | df[c.TARGET].isna()
X, y = X[~mask], df.loc[~mask, c.TARGET].astype(int)

X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X_tr, y_tr))

bp = [col for col in X.columns if col.startswith("bp_")]
analytes = [col for col in X.columns if col not in bp]
print(f"Train {len(X_tr)} | BP {len(bp)} | analytes {len(analytes)} | prevalence {y_tr.mean():.4f}")

# ---------------- Helpers ----------------
def fit(cols, params, tr):
    # Early stopping on an internal split of the fold's training data: the validation fold is not touched
    Xa, Xe, ya, ye = train_test_split(X_tr.iloc[tr][cols], y_tr.iloc[tr], test_size=ES_SIZE,
                                      stratify=y_tr.iloc[tr], random_state=0)
    m = CatBoostClassifier(**params, **FIXED)
    m.fit(Xa, ya, eval_set=(Xe, ye), use_best_model=True)
    return m, Xe, ye


def cv_metrics(cols_by_fold, params):
    ll, auc = [], []
    for (tr, va), cols in zip(folds, cols_by_fold):
        m = fit(cols, params, tr)[0]
        p = m.predict_proba(X_tr.iloc[va][cols])[:, 1]
        ll.append(log_loss(y_tr.iloc[va], p))
        auc.append(roc_auc_score(y_tr.iloc[va], p))
    return np.array(ll), np.array(auc)

# ---------------- 1. Small grid (all variables, training set only) ----------------
grid = []
for g in [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]:
    ll, auc = cv_metrics([list(X.columns)] * N_FOLDS, g)
    grid.append({**g, "logloss_cv": ll.mean(), "auc_cv": auc.mean()})
    print(f"  {g} -> CV log-loss {ll.mean():.4f} | CV AUC {auc.mean():.4f}")

grid_df = pd.DataFrame(grid).sort_values("logloss_cv")
best = {"depth": int(grid_df.iloc[0]["depth"]), "learning_rate": float(grid_df.iloc[0]["learning_rate"])}
print(f"Chosen configuration: {best}")

# ---------------- 2. Per-fold SHAP ranking of analytes (on the internal split, not used for training) ----------------
shap, rankings = [], []
for i, (tr, va) in enumerate(folds):
    m, Xe, ye = fit(list(X.columns), best, tr)
    sv = m.get_feature_importance(data=Pool(Xe, ye), type="ShapValues")[:, :-1]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    shap.append(imp)
    rankings.append(imp[analytes].sort_values(ascending=False).index.tolist())
    print(f"  SHAP fold {i + 1}/{N_FOLDS}")

# Reference ranking (mean across folds), for reporting and for the final selection
ranking = pd.DataFrame({"Variable": X.columns, "SHAP": pd.concat(shap, axis=1).mean(axis=1).values})
ranking = ranking[ranking["Variable"].isin(analytes)].sort_values("SHAP", ascending=False).reset_index(drop=True)

# ---------------- 3. Log-loss curve over k (BP + k analytes, each fold's own ranking) ----------------
ks = sorted(set([k for k in K_VALUES if k <= len(analytes)] + [len(analytes)]))
ll_folds, auc_folds = {}, {}
for k in ks:
    ll_folds[k], auc_folds[k] = cv_metrics([bp + rk[:k] for rk in rankings], best)
    print(f"  BP + {k} analytes -> CV log-loss {ll_folds[k].mean():.4f} | CV AUC {auc_folds[k].mean():.4f}")

# Paired 1-SE on log-loss: per-fold differences against the k with the lowest mean log-loss
k_best = min(ks, key=lambda k: ll_folds[k].mean())
curve = []
for k in ks:
    diff = ll_folds[k] - ll_folds[k_best]
    curve.append({"k": k, "logloss_cv": ll_folds[k].mean(), "auc_cv": auc_folds[k].mean(),
                  "mean_diff": diff.mean(), "se_diff": diff.std(ddof=1) / np.sqrt(N_FOLDS)})

curve_df = pd.DataFrame(curve)
# Per-fold values (for the error bars in the figure)
by_fold = pd.DataFrame([{"k": k, "fold": i + 1, "logloss": ll_folds[k][i], "auc": auc_folds[k][i]}
                        for k in ks for i in range(N_FOLDS)])
k_opt = int(curve_df.loc[curve_df["mean_diff"] <= curve_df["se_diff"], "k"].min())

# ---------------- Save ----------------
ranking["Selected"] = ranking.index < k_opt
# Stability: in how many folds each analyte is in the top k_opt of that fold's own ranking
ranking["Folds in top k"] = ranking["Variable"].map(
    lambda v: sum(v in rk[:k_opt] for rk in rankings))
with pd.ExcelWriter(c.SELECTION_CATBOOST) as w:
    ranking.to_excel(w, sheet_name="SHAP ranking", index=False)
    curve_df.to_excel(w, sheet_name="k curve", index=False)
    by_fold.to_excel(w, sheet_name="k curve by fold", index=False)
    grid_df.to_excel(w, sheet_name="Grid", index=False)

print(f"\nCurve:\n{curve_df.to_string(index=False)}")
print(f"\nChosen k: {k_opt} analytes + {len(bp)} blood pressure predictors")
print(ranking.head(k_opt).to_string(index=False))
