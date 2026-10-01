import pandas as pd
import config as c

# Table 2: test-set AUROC (95% CI) by algorithm and predictor set (tab-separated, to paste into Word)
SETS = ["BP", "BP+Lasso", "BP+SHAP", "BP+All"]
ALGORITHMS = [("LogisticRegression", "Elastic-net logistic regression"), ("kNN", "k-nearest neighbours"),
              ("MLP", "Multilayer perceptron"), ("RandomForest", "Random forest"), ("CatBoost", "CatBoost")]

df = pd.read_excel(c.BENCHMARK, sheet_name="Full").set_index(["Algorithm", "Set"])

print("Algorithm\t" + "\t".join(SETS))
for algorithm, name in ALGORITHMS:
    cells = [f"{df.loc[(algorithm, s), 'AUROC']:.3f} ({df.loc[(algorithm, s), 'AUROC CI95 low']:.3f}–"
             f"{df.loc[(algorithm, s), 'AUROC CI95 high']:.3f})" for s in SETS]
    print(name + "\t" + "\t".join(cells))
