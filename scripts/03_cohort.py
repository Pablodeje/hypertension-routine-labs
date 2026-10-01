import numpy as np
import pandas as pd
import config as c


def aggregate(g, ddof=1):
    # g: series groupby -> one column per statistic (quartiles with linear interpolation)
    return pd.DataFrame({"mean": g.mean(), "variance": g.var(ddof=ddof), "sd": g.std(ddof=ddof),
                         "q1": g.quantile(0.25), "q3": g.quantile(0.75), "min": g.min(), "max": g.max()})


labs = pd.read_parquet(c.LABS_PARQUET)
bp = pd.read_parquet(c.BP_PARQUET)

# ---------------- Eligibility: >= 3 determinations of every analyte and of blood pressure ----------------
n_lab = labs.groupby(["subject_id", "analyte"], observed=True).size().unstack(fill_value=0)
n_lab = n_lab.reindex(columns=list(c.ITEMIDS), fill_value=0)
ok_lab = (n_lab >= c.MIN_DETERMINATIONS).all(axis=1)
ok_bp = bp.groupby("subject_id").size() >= c.MIN_DETERMINATIONS
eligible = ok_lab[ok_lab].index.intersection(ok_bp[ok_bp].index)

print(f"Patients with >=3 of each of the 17 analytes: {ok_lab.sum():,}")
print(f"Patients with >=3 blood pressure readings:    {ok_bp.sum():,}")
print(f"Eligible (both):                              {len(eligible):,}")

# ---------------- Aggregates ----------------
labs = labs[labs["subject_id"].isin(eligible)]
bp = bp[bp["subject_id"].isin(eligible)]

blocks = []
for analyte, sub in labs.groupby("analyte", observed=True):
    a = aggregate(sub.groupby("subject_id")["value"])
    a.columns = [f"{analyte}_{e}" for e in a.columns]
    blocks.append(a)
for side in ("s", "d"):
    a = aggregate(bp.groupby("subject_id")[side], ddof=0)   # blood pressure used ddof=0; analytes, ddof=1
    a.columns = [f"bp_{side}_{e}" for e in a.columns]
    blocks.append(a)
cohort = pd.concat(blocks, axis=1)

# ---------------- Outcome ----------------
dx = pd.read_csv(c.DIAGNOSES, usecols=["subject_id", "icd_code", "icd_version"], dtype={"icd_code": str})
dx["icd_code"] = dx["icd_code"].str.strip()


def patients(icd9, icd10):
    # exact code match within its ICD version
    m = ((dx["icd_version"] == 9) & dx["icd_code"].isin(icd9)) | ((dx["icd_version"] == 10) & dx["icd_code"].isin(icd10))
    return set(dx.loc[m, "subject_id"])


a = patients(c.ICD9_A, c.ICD10_A)
b = patients(c.ICD9_B, c.ICD10_B)
cohort["Hypertension A"] = cohort.index.isin(a).astype(int)
cohort["Hypertension B"] = cohort.index.isin(b - a).astype(int)   # B codes without A
cohort[c.TARGET] = cohort.index.isin(a | b).astype(int)

# ---------------- Admission flag (flags only; nobody is excluded here) ----------------
adm = pd.read_csv(c.ADMISSIONS, usecols=["subject_id"])
cohort["Admission"] = np.where(cohort.index.isin(adm["subject_id"]), "YES", "NO")

# ---------------- Column order ----------------
cohort = cohort[c.COLUMNS]
cohort.index.name = "subject_id"
cohort.to_parquet(c.COHORT)
print(f"\nCohort: {len(cohort):,} patients x {cohort.shape[1]} columns | "
      f"with admission {(cohort['Admission'] == 'YES').sum():,} | cases {cohort[c.TARGET].sum():,}")
