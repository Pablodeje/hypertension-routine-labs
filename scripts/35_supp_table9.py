import pandas as pd
import config as c

# Supplementary Table 9: test-set performance and calibration of all models (tab-separated, to paste into Word).
# One row with the point estimates and one with the 95% CIs per model
SETS = ["BP", "BP+Lasso", "BP+SHAP", "BP+All"]
ALGORITHMS = [("LogisticRegression", "Elastic-net logistic regression"), ("kNN", "k-nearest neighbours"),
              ("MLP", "Multilayer perceptron"), ("RandomForest", "Random forest"), ("CatBoost", "CatBoost")]
# metric, decimals, CI separator
METRICS = [("AUROC", 3, "–"), ("AUPRC", 3, "–"), ("Brier", 3, "–"),
           ("Calibration slope", 3, "–"), ("Calibration-in-the-large", 4, " to ")]

df = pd.read_excel(c.BENCHMARK, sheet_name="Full").set_index(["Algorithm", "Set"])


def f(x, d):
    return f"{x:.{d}f}".replace("-", "−")   # typographic minus


print("Predictor set\tAUROC\tAUPRC\tBrier score\tCalibration slope\tCalibration-in-the-large")
for algorithm, name in ALGORITHMS:
    print(name)
    for s in SETS:
        r = df.loc[(algorithm, s)]
        print(s + "\t" + "\t".join(f(r[m], d) for m, d, _ in METRICS))
        print("\t" + "\t".join(f"({f(r[m + ' CI95 low'], d)}{sep}{f(r[m + ' CI95 high'], d)})" for m, d, sep in METRICS))
