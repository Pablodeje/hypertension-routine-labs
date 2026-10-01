import pandas as pd
import config as c

# Supplementary Table 11: paired differences against blood pressure alone in secondary performance and
# calibration metrics (tab-separated, to paste into Word). One row with the differences and one with the 95% CIs
SETS = ["BP+Lasso", "BP+SHAP", "BP+All"]
ALGORITHMS = [("LogisticRegression", "Elastic-net logistic regression"), ("kNN", "k-nearest neighbours"),
              ("MLP", "Multilayer perceptron"), ("RandomForest", "Random forest"), ("CatBoost", "CatBoost")]
METRICS = [("AUPRC", 3), ("Calibration slope", 3), ("Calibration-in-the-large", 4)]

df = pd.read_excel(c.BENCHMARK, sheet_name="Paired gain").set_index(["Algorithm", "Set"])


def f(x, d):
    return f"{x:.{d}f}".replace("-", "−")   # typographic minus


print("Predictor set\tΔAUPRC\tΔCalibration slope\tΔCalibration-in-the-large")
for algorithm, name in ALGORITHMS:
    print(name)
    for s in SETS:
        r = df.loc[(algorithm, s)]
        print(s + "\t" + "\t".join(f(r[f"diff {m}"], d) for m, d in METRICS))
        print("\t" + "\t".join(f"({f(r[f'diff {m} CI95 low'], d)} to {f(r[f'diff {m} CI95 high'], d)})" for m, d in METRICS))
