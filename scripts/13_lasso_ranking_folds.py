import numpy as np
import pandas as pd

from glum import GeneralizedLinearRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import config as c

# Lasso ranking of analytes within each CatBoost cross-validation fold (full Bolasso on each fold's training data),
# so that the lasso k curve uses rankings that never saw the validation fold, as the SHAP curve does

# ---------------- Settings ----------------
TEST_SIZE, SPLIT_SEED = 0.25, 0
N_FOLDS = 5
ALPHAS = np.logspace(-6, 0, 25)
N_BOOTSTRAP = 200
MAX_ITER = 1000

# ---------------- Load (as in 10_bolasso.py and 12_catboost_shap.py) ----------------
df = pd.read_excel(c.COHORT_CLEAN).set_index("subject_id")
df = df[df["Admission"].astype(str).str.strip() == "YES"]
df = df.drop(columns=[col for col in df.columns if col.endswith("_variance")])

X = df.drop(columns=[c.TARGET] + c.NON_PREDICTORS).astype(float)
mask = X.isna().any(axis=1) | df[c.TARGET].isna()
X, y = X[~mask], df.loc[~mask, c.TARGET].astype(int)

X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X_tr, y_tr))   # same folds as CatBoost

is_bp = X.columns.str.startswith("bp_")
P1 = np.where(is_bp, 0.0, 1.0)


def model(alpha):
    return Pipeline([("sc", StandardScaler()),
                     ("est", GeneralizedLinearRegressor(family="binomial", alpha=alpha, l1_ratio=1.0,
                                                        P1=P1, max_iter=MAX_ITER))])


# ---------------- Full Bolasso within each fold's training data ----------------
rows = []
for f, (tr, _) in enumerate(folds):
    Xf, yf = X_tr.iloc[tr], y_tr.iloc[tr]

    # alpha by the 1-SE rule with an inner CV on the fold's training data
    ll = np.zeros((N_FOLDS, len(ALPHAS)))
    for i, (a_tr, a_va) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(Xf, yf)):
        for j, a in enumerate(ALPHAS):
            m = model(a).fit(Xf.iloc[a_tr], yf.iloc[a_tr])
            ll[i, j] = log_loss(yf.iloc[a_va], m.predict(Xf.iloc[a_va]))
    mean = ll.mean(axis=0)
    se = ll.std(axis=0, ddof=1) / np.sqrt(N_FOLDS)
    j_min = int(np.argmin(mean))
    alpha_1se = ALPHAS[mean <= mean[j_min] + se[j_min]].max()

    pipe = model(alpha_1se)
    coef = pipe.fit(Xf, yf).named_steps["est"].coef_.ravel().copy()

    count = np.zeros(X.shape[1])
    not_converged = 0
    for b in range(N_BOOTSTRAP):
        Xb, yb = resample(Xf, yf, stratify=yf, random_state=b)
        m = pipe.fit(Xb, yb).named_steps["est"]
        count += m.coef_.ravel() != 0
        not_converged += m.n_iter_ >= MAX_ITER

    # Same order as 10_bolasso.py: frequency, then |coefficient|; frequency 0 at the end
    r = pd.DataFrame({"Variable": X.columns, "Selection frequency": count / N_BOOTSTRAP,
                      "abs_coef": np.abs(coef)})[~is_bp]
    r = r.sort_values(["Selection frequency", "abs_coef"], ascending=False, kind="stable").reset_index(drop=True)
    r.insert(0, "fold", f + 1)
    r.insert(1, "rank", r.index + 1)
    rows.append(r)
    print(f"Fold {f + 1}/{N_FOLDS} | 1-SE alpha {alpha_1se:.2e} | frequency >= 0.95: "
          f"{(r['Selection frequency'] >= 0.95).sum()} | bootstraps not converged: {not_converged}")

pd.concat(rows).to_excel(c.LASSO_RANKING_FOLDS, index=False)
print(f"Saved to {c.LASSO_RANKING_FOLDS}")
