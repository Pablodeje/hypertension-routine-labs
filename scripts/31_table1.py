import numpy as np
import pandas as pd
import config as c

# Table 1: characteristics of the analysed cohort by hypertension status (tab-separated, to paste into Word)
# label, column, factor to SI, unit, decimals
ROWS = [
    ("Age (anchor age)", "anchor_age", 1, "years", 0),
    ("Blood pressure readings", "bp_s_count", 1, "n", 0),
    ("Metabolic panel determinations", "Sodium_count", 1, "n", 0),
    ("Blood count determinations", "Hemoglobin_count", 1, "n", 0),
    ("Systolic blood pressure", "bp_s_mean", 1, "mmHg", 0),
    ("Diastolic blood pressure", "bp_d_mean", 1, "mmHg", 0),
    ("Sodium", "Sodium_mean", 1, "mmol/L", 1),
    ("Potassium", "Potassium_mean", 1, "mmol/L", 2),
    ("Chloride", "Chloride_mean", 1, "mmol/L", 1),
    ("Bicarbonate", "Bicarbonate_mean", 1, "mmol/L", 1),
    ("Anion gap", "Anion GAP_mean", 1, "mmol/L", 1),
    ("Urea nitrogen", "Urea Nitrogen_mean", 0.357, "mmol/L", 1),
    ("Creatinine", "Creatinine_mean", 88.4, "µmol/L", 0),
    ("Glucose", "Glucose_mean", 0.0555, "mmol/L", 1),
    ("Haemoglobin", "Hemoglobin_mean", 10, "g/L", 0),
    ("Haematocrit", "Hematocrit_mean", 0.01, "L/L", 2),
    ("Red cell count", "Red Blood Cells_mean", 1, "×10¹²/L", 2),
    ("Mean corpuscular volume", "MCV_mean", 1, "fL", 1),
    ("Mean corpuscular haemoglobin", "MCH_mean", 1, "pg", 1),
    ("Mean corpuscular haemoglobin concentration", "MCHC_mean", 10, "g/L", 0),
    ("Red cell distribution width", "RDW_mean", 1, "%", 1),
    ("White cell count", "White Blood Cells_mean", 1, "×10⁹/L", 1),
    ("Platelet count", "Platelet Count_mean", 1, "×10⁹/L", 0),
]

df = (pd.read_excel(c.COHORT_CLEAN)
      .merge(pd.read_parquet(c.TABLE1_DATA), on="subject_id", how="left"))
df = df[df["Admission"] == "YES"].drop_duplicates("subject_id")
assert df[["gender", "anchor_age", "bp_s_count", "Sodium_count", "Hemoglobin_count"]].notna().all().all()

cases = df[df[c.TARGET] == 1]
controls = df[df[c.TARGET] == 0]


def med_iqr(x, d):
    q1, m, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{m:,.{d}f} ({q1:,.{d}f}–{q3:,.{d}f})"


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


lines = [f"Variable\tUnit\tControls (n = {len(controls):,})\tHypertension (n = {len(cases):,})\tSMD"]

for label, col, factor, unit, d in ROWS:
    a = cases[col] * factor
    b = controls[col] * factor
    lines.append(f"{label}\t{unit}\t{med_iqr(b, d)}\t{med_iqr(a, d)}\t{smd(a, b):.2f}")
    if col == "anchor_age":
        fa = (cases["gender"] == "F").mean()
        fb = (controls["gender"] == "F").mean()
        s = (fa - fb) / np.sqrt((fa * (1 - fa) + fb * (1 - fb)) / 2)
        lines.append(f"Female sex\tn (%)\t{(controls['gender'] == 'F').sum():,} ({100 * fb:.1f})\t"
                     f"{(cases['gender'] == 'F').sum():,} ({100 * fa:.1f})\t{s:.2f}")

print("\n".join(lines))
