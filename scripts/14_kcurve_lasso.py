import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

import config as c

# k curve with the same CatBoost models as 12_catboost_shap.py, adding analytes in each fold's lasso order

# ---------------- Settings (as in 12_catboost_shap.py) ----------------
TEST_SIZE, SPLIT_SEED = 0.25, 0
N_FOLDS = 5
ES_SIZE = 0.15
FIXED = dict(iterations=2000, l2_leaf_reg=3, eval_metric="Logloss", early_stopping_rounds=100,
             verbose=0, task_type="CPU", random_seed=0)
K_VALUES = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]

# ---------------- Load (as in 12_catboost_shap.py) ----------------
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

# Same configuration chosen by the CatBoost grid (first row of the Grid sheet)
g = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="Grid").iloc[0]
best = {"depth": int(g["depth"]), "learning_rate": float(g["learning_rate"])}
print(f"CatBoost configuration: {best}")

# Each fold's own lasso ranking
r = pd.read_excel(c.LASSO_RANKING_FOLDS)
rankings = [r[r["fold"] == f + 1].sort_values("rank")["Variable"].tolist() for f in range(N_FOLDS)]

# ---------------- Helpers (as in 12_catboost_shap.py) ----------------
def fit(cols, params, tr):
    Xa, Xe, ya, ye = train_test_split(X_tr.iloc[tr][cols], y_tr.iloc[tr], test_size=ES_SIZE,
                                      stratify=y_tr.iloc[tr], random_state=0)
    m = CatBoostClassifier(**params, **FIXED)
    m.fit(Xa, ya, eval_set=(Xe, ye), use_best_model=True)
    return m


def cv_metrics(cols_by_fold, params):
    ll, auc = [], []
    for (tr, va), cols in zip(folds, cols_by_fold):
        p = fit(cols, params, tr).predict_proba(X_tr.iloc[va][cols])[:, 1]
        ll.append(log_loss(y_tr.iloc[va], p))
        auc.append(roc_auc_score(y_tr.iloc[va], p))
    return np.array(ll), np.array(auc)

# ---------------- k curve in lasso order ----------------
ks = sorted(set([k for k in K_VALUES if k <= len(analytes)] + [len(analytes)]))
rows = []
for k in ks:
    ll, auc = cv_metrics([bp + rk[:k] for rk in rankings], best)
    rows += [{"k": k, "fold": i + 1, "logloss": ll[i], "auc": auc[i]} for i in range(N_FOLDS)]
    print(f"  BP + {k} analytes -> CV log-loss {ll.mean():.4f} | CV AUC {auc.mean():.4f}")

pd.DataFrame(rows).to_excel(c.KCURVE_LASSO, sheet_name="k curve by fold", index=False)
print(f"Saved to {c.KCURVE_LASSO}")
