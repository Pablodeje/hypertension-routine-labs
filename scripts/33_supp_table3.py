import pandas as pd
import config as c

# Supplementary Table 3: plausibility limits and patients excluded per variable (tab-separated, to paste into Word).
# Same cohort, steps and limits as 04_exclusions.py
# column: label, unit (as recorded in MIMIC-IV), source (H = Harutyunyan, R = ricu, A = author-defined)
ROWS = {
    "bp_s": ("Systolic blood pressure", "mmHg", "H"),
    "bp_d": ("Diastolic blood pressure", "mmHg", "H"),
    "Sodium": ("Sodium", "mEq/L", "H"),
    "Potassium": ("Potassium", "mEq/L", "H"),
    "Chloride": ("Chloride", "mEq/L", "H"),
    "Bicarbonate": ("Bicarbonate", "mEq/L", "H"),
    "Anion GAP": ("Anion gap", "mEq/L", "H"),
    "Urea Nitrogen": ("Urea nitrogen", "mg/dL", "H"),
    "Creatinine": ("Creatinine", "mg/dL", "H"),
    "Glucose": ("Glucose", "mg/dL", "H"),
    "Hemoglobin": ("Haemoglobin", "g/dL", "H"),
    "Hematocrit": ("Haematocrit", "%", "H"),
    "Red Blood Cells": ("Red cell count", "×10⁶/µL", "R"),
    "MCV": ("Mean corpuscular volume", "fL", "R"),
    "MCH": ("Mean corpuscular haemoglobin", "pg", "A"),
    "MCHC": ("Mean corpuscular haemoglobin concentration", "g/dL", "R"),
    "RDW": ("Red cell distribution width", "%", "R"),
    "White Blood Cells": ("White cell count", "×10³/µL", "H"),
    "Platelet Count": ("Platelet count", "×10³/µL", "H"),
}

df = pd.read_parquet(c.COHORT)
df = df[df["Admission"].astype(str).str.strip() == "YES"]
df = df.drop(columns=[col for col in df.columns if col.endswith("_variance")])
X = df.drop(columns=[c.TARGET] + c.NON_PREDICTORS).astype(float)
X = X[~(X.isna().any(axis=1) | df[c.TARGET].isna())]

flags = {}
for col in ROWS:
    lower, upper = c.LIMITS[col]
    flags[col] = (X[f"{col}_min"] < lower) | (X[f"{col}_max"] > upper)
    if col not in c.NO_ZERO_RULE:
        flags[col] |= X[f"{col}_min"] == 0
flags = pd.DataFrame(flags)
alone = flags.sum(axis=1) == 1

print("Variable\tUnit\tLower limit\tUpper limit\tSource\tExcluded, n\tExcluded by this variable alone, n")
for col, (label, unit, source) in ROWS.items():
    lower, upper = c.LIMITS[col]
    print(f"{label}\t{unit}\t{lower:g}\t{upper:g}\t{source}\t{flags[col].sum():,}\t{(flags[col] & alone).sum():,}")
print(f"Any variable\t\t\t\t\t{flags.any(axis=1).sum():,} of {len(X):,}\t")
