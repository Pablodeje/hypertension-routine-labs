"""
Contribution of each analyte to the SHAP selection, as reported in the text (Results, Contribution of individual
analytes; Discussion): the values of Figure 4, by analyte and by group.
"""
import pandas as pd

import config as c

GROUPS = {
    "Electrolytes": ["Sodium", "Potassium", "Chloride", "Bicarbonate", "Anion GAP"],
    "Renal function (urea nitrogen and creatinine)": ["Urea Nitrogen", "Creatinine"],
    "Glucose": ["Glucose"],
    "WBC and RDW": ["White Blood Cells", "RDW"],
    "White cells and platelets": ["White Blood Cells", "Platelet Count"],
    "Red cells (haemoglobin, haematocrit, RBC, MCV, MCH, MCHC, RDW)":
        ["Hemoglobin", "Hematocrit", "Red Blood Cells", "MCV", "MCH", "MCHC", "RDW"],
    "Haemoglobin, haematocrit, RBC and red cell indices (MCV, MCH, MCHC)":
        ["Hemoglobin", "Hematocrit", "Red Blood Cells", "MCV", "MCH", "MCHC"],
}

r = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")
r["analyte"] = r["Variable"].str.rsplit("_", n=1).str[0]
pct = (r.groupby("analyte")["SHAP"].sum() / r["SHAP"].sum() * 100).sort_values(ascending=False)

print("Analyte\t% of total laboratory mean |SHAP|\tCumulative %")
for a, v in pct.items():
    print(f"{a}\t{v:.1f}\t{pct.loc[:a].sum():.1f}")
print()
for g, members in GROUPS.items():
    print(f"{g}\t{pct[members].sum():.1f}")
