import numpy as np
import pandas as pd

from glum import GeneralizedLinearRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import config as c

# Sensitivity analysis: same procedure as 10_bolasso.py with an elastic-net penalty

# ---------------- Settings ----------------
TEST_SIZE, SPLIT_SEED = 0.25, 0
N_FOLDS = 5
ALPHAS = np.logspace(-6, 0, 25)
L1_RATIOS = [0.1, 0.5, 0.9]
N_BOOTSTRAP = 200
FREQUENCY_THRESHOLD = 0.95
MAX_ITER = 1000

# ---------------- Load ----------------
df = pd.read_excel(c.COHORT_CLEAN).set_index("subject_id")
df = df[df["Admission"].astype(str).str.strip() == "YES"]
df = df.drop(columns=[col for col in df.columns if col.endswith("_variance")])

X = df.drop(columns=[c.TARGET] + c.NON_PREDICTORS).astype(float)
mask = X.isna().any(axis=1) | df[c.TARGET].isna()
X, y = X[~mask], df.loc[~mask, c.TARGET].astype(int)

X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)

is_bp = X.columns.str.startswith("bp_")
P = np.where(is_bp, 0.0, 1.0)   # blood pressure unpenalized (factor 0 in L1 and L2); only analytes are penalized
print(f"Train {len(X_tr)} | BP {is_bp.sum()} | analytes {(~is_bp).sum()} | prevalence {y_tr.mean():.4f}")


def model(alpha, l1):
    return Pipeline([("sc", StandardScaler()),
                     ("est", GeneralizedLinearRegressor(family="binomial", alpha=alpha, l1_ratio=l1,
                                                        P1=P, P2=P, max_iter=MAX_ITER))])

# ---------------- l1_ratio at the CV minimum and alpha by the 1-SE rule (training set only) ----------------
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0).split(X_tr, y_tr)
ll = np.zeros((N_FOLDS, len(L1_RATIOS), len(ALPHAS)))
for i, (tr, va) in enumerate(folds):
    for r, l1 in enumerate(L1_RATIOS):
        for j, a in enumerate(ALPHAS):
            m = model(a, l1).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            ll[i, r, j] = log_loss(y_tr.iloc[va], m.predict(X_tr.iloc[va]))
    print(f"  CV fold {i + 1}/{N_FOLDS}")

mean = ll.mean(axis=0)
se = ll.std(axis=0, ddof=1) / np.sqrt(N_FOLDS)
r_min, j_min = np.unravel_index(np.argmin(mean), mean.shape)
l1_min, alpha_min = L1_RATIOS[r_min], ALPHAS[j_min]

# 1-SE on alpha only, with l1_ratio fixed at the optimum (glmnet practice)
alpha_1se = ALPHAS[mean[r_min] <= mean[r_min, j_min] + se[r_min, j_min]].max()
l1_1se = l1_min
print(f"CV minimum: alpha={alpha_min:.2e}, l1_ratio={l1_min} | 1-SE: alpha={alpha_1se:.2e}, l1_ratio={l1_1se}")
if j_min in (0, len(ALPHAS) - 1):
    print("  WARNING: the CV minimum is at the edge of the alpha grid")

# Convergence at the two configurations that drive the selection (fit on the whole training set)
for name, a, l1 in (("CV minimum", alpha_min, l1_min), ("1-SE", alpha_1se, l1_1se)):
    it = model(a, l1).fit(X_tr, y_tr).named_steps["est"].n_iter_
    print(f"  {name}: {it} iterations {'-> DID NOT CONVERGE' if it >= MAX_ITER else '(converged)'}")

pipe = model(alpha_1se, l1_1se)
coef = pipe.fit(X_tr, y_tr).named_steps["est"].coef_.ravel().copy()
print(f"Analytes with non-zero coefficient in the 1-SE fit: {int((coef[~is_bp] != 0).sum())} of {(~is_bp).sum()}")

# ---------------- Bootstrap selection frequency (training set only) ----------------
count = np.zeros(X_tr.shape[1])
not_converged = 0
for i in range(N_BOOTSTRAP):
    Xb, yb = resample(X_tr, y_tr, stratify=y_tr, random_state=i)
    m = pipe.fit(Xb, yb).named_steps["est"]
    count += m.coef_.ravel() != 0
    not_converged += m.n_iter_ >= MAX_ITER
    print(f"  bootstrap {i + 1}/{N_BOOTSTRAP}")
frequency = count / N_BOOTSTRAP
print(f"Bootstraps not converged: {not_converged} of {N_BOOTSTRAP}")

# ---------------- Results (analytes only; blood pressure always enters) ----------------
res = pd.DataFrame({"Variable": X.columns, "Coefficient": coef,
                    "abs_coef": np.abs(coef), "Selection frequency": frequency})[~is_bp]
res["Selected"] = res["Selection frequency"] >= FREQUENCY_THRESHOLD
res = res[res["Selection frequency"] > 0].sort_values(["Selection frequency", "abs_coef"], ascending=False)
res.to_excel(c.SELECTION_ELASTICNET, index=False)

print(f"\nSelected (frequency >= {FREQUENCY_THRESHOLD}): {res['Selected'].sum()}")
print(res[res["Selected"]].to_string(index=False))
