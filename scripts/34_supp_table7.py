import pandas as pd
import config as c

# Supplementary Table 7: selection frequencies, lasso coefficients and SHAP ranking of the 102 laboratory
# predictors, ordered by SHAP rank (tab-separated, to paste into Word)
ANALYTES = {
    "Sodium": "Sodium", "Potassium": "Potassium", "Chloride": "Chloride", "Bicarbonate": "Bicarbonate",
    "Anion GAP": "Anion gap", "Urea Nitrogen": "Urea nitrogen", "Creatinine": "Creatinine", "Glucose": "Glucose",
    "Hemoglobin": "Haemoglobin", "Hematocrit": "Haematocrit", "Red Blood Cells": "Red cell count",
    "MCV": "Mean corpuscular volume", "MCH": "Mean corpuscular haemoglobin",
    "MCHC": "Mean corpuscular haemoglobin concentration", "RDW": "Red cell distribution width",
    "White Blood Cells": "White cell count", "Platelet Count": "Platelet count",
}
STATISTICS = {"mean": "mean", "sd": "SD", "min": "minimum", "max": "maximum",
              "q1": "first quartile", "q3": "third quartile"}

shap = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")
lasso = pd.read_excel(c.SELECTION_LASSO).set_index("Variable")
enet = pd.read_excel(c.SELECTION_ELASTICNET).set_index("Variable")


def number(x, d):
    # typographic minus; a coefficient of exactly 0 is shown as "0"
    return "0" if x == 0 else f"{x:.{d}f}".replace("-", "−")


print("Predictor\tLasso frequency (%)\tLasso coefficient\tElastic net frequency (%)\tMean |SHAP|\tSHAP rank\tFolds in top k (n)")
for rank, r in enumerate(shap.itertuples(index=False), start=1):
    analyte, stat = r.Variable.rsplit("_", 1)
    # predictors never selected in any resample are not in the lasso/elastic-net outputs: frequency and coefficient 0
    l_freq = lasso["Selection frequency"].get(r.Variable, 0) * 100
    l_coef = lasso["Coefficient"].get(r.Variable, 0)
    e_freq = enet["Selection frequency"].get(r.Variable, 0) * 100
    print(f"{ANALYTES[analyte]}, {STATISTICS[stat]}\t{l_freq:.1f}\t{number(l_coef, 3)}\t{e_freq:.1f}\t"
          f"{r.SHAP:.4f}\t{rank}\t{r[3]}")
