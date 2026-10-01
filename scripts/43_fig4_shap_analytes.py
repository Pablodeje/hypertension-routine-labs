import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config as c

# Figure 4: contribution of each analyte to the SHAP selection (share of the total laboratory mean |SHAP|)

# analyte in the data: (label, group)
ANALYTES = {
    "Sodium": ("Sodium", "Electrolytes"), "Potassium": ("Potassium", "Electrolytes"),
    "Chloride": ("Chloride", "Electrolytes"), "Bicarbonate": ("Bicarbonate", "Electrolytes"),
    "Anion GAP": ("Anion gap", "Electrolytes"),
    "Urea Nitrogen": ("Urea nitrogen", "Renal function"), "Creatinine": ("Creatinine", "Renal function"),
    "Glucose": ("Glucose", "Glucose"),
    "White Blood Cells": ("WBC", "White cells and platelets"),
    "Platelet Count": ("Platelet count", "White cells and platelets"),
    "Hemoglobin": ("Haemoglobin", "Red cells"), "Hematocrit": ("Haematocrit", "Red cells"),
    "Red Blood Cells": ("RBC", "Red cells"), "MCV": ("MCV", "Red cells"),
    "MCH": ("MCH", "Red cells"),
    "MCHC": ("MCHC", "Red cells"),
    "RDW": ("RDW", "Red cells"),
}
COLORS = {"Electrolytes": "#0072B2", "Renal function": "#D55E00", "Glucose": "#E69F00",
           "White cells and platelets": "#009E73", "Red cells": "#CC79A7"}

plt.rcParams.update({"font.family": "Arial", "font.size": 9})

r = pd.read_excel(c.SELECTION_CATBOOST, sheet_name="SHAP ranking")
r["analyte"] = r["Variable"].str.rsplit("_", n=1).str[0]
r = r[r["analyte"].isin(ANALYTES)]

pct = (r.groupby("analyte")["SHAP"].sum() / r["SHAP"].sum() * 100).sort_values()

fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.barh([ANALYTES[a][0] for a in pct.index], pct.values,
        color=[COLORS[ANALYTES[a][1]] for a in pct.index], height=0.7)
ax.set_xlabel("Share of total laboratory mean |SHAP| (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=col, label=g) for g, col in COLORS.items()], frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig(c.FIGURES / "fig4_shap_analytes.svg")