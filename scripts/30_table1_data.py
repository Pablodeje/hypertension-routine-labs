import pandas as pd
import config as c

# Table 1 data not in the cohort file: sex and age (patients) and number of determinations per patient
ids = pd.read_excel(c.COHORT_CLEAN, usecols=["subject_id"])

patients = pd.read_csv(c.PATIENTS, usecols=["subject_id", "gender", "anchor_age"])

labs = pd.read_parquet(c.LABS_PARQUET)
labs = labs[labs["analyte"].isin(["Sodium", "Hemoglobin"])]
n_lab = labs.groupby(["subject_id", "analyte"], observed=True).size().unstack(fill_value=0)
n_lab.columns = [f"{a}_count" for a in n_lab.columns]

n_bp = pd.read_parquet(c.BP_PARQUET).groupby("subject_id").size().rename("bp_s_count")

data = (ids.merge(patients, on="subject_id", how="left")
           .merge(n_lab, left_on="subject_id", right_index=True, how="left")
           .merge(n_bp, left_on="subject_id", right_index=True, how="left"))
data.to_parquet(c.TABLE1_DATA, index=False)
print(f"Saved: {len(data):,} patients | with missing data: {data.isna().any(axis=1).sum()}")
