import pandas as pd
import config as c

# Supplementary Table 10: test-set performance of all models at the 0.5 and Youden thresholds
# (tab-separated, to paste into Word)
SETS = ["BP", "BP+Lasso", "BP+SHAP", "BP+All"]
ALGORITHMS = [("LogisticRegression", "Elastic-net logistic regression"), ("kNN", "k-nearest neighbours"),
              ("MLP", "Multilayer perceptron"), ("RandomForest", "Random forest"), ("CatBoost", "CatBoost")]
METRICS = ["Accuracy", "Kappa", "Sensitivity", "Specificity", "PPV", "NPV"]

df = pd.read_excel(c.BENCHMARK, sheet_name="Full").set_index(["Algorithm", "Set"])


def f(x):
    return f"{x:.3f}".replace("-", "−")   # typographic minus


print("Predictor set and threshold\t" + "\t".join(METRICS))
for algorithm, name in ALGORITHMS:
    print(name)
    for s in SETS:
        r = df.loc[(algorithm, s)]
        for t, label in (("0.5", f"{s}, 0.5"), ("Youden", f"{s}, Youden, {f(r['Youden threshold (train)'])}")):
            cells = [f"{f(r[f'{m} ({t})'])} ({f(r[f'{m} ({t}) CI95 low'])}–{f(r[f'{m} ({t}) CI95 high'])})"
                     for m in METRICS]
            print(label + "\t" + "\t".join(cells))
