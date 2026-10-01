import pandas as pd
import config as c

# Availability of each analyte among all patients with laboratory data (input to Supplementary Figure 1):
# True if the patient has >= 3 determinations of that analyte
labs = pd.read_parquet(c.LABS_PARQUET)

n = labs.groupby(["subject_id", "analyte"], observed=True).size().unstack(fill_value=0)
n = n.reindex(columns=list(c.ITEMIDS), fill_value=0)
available = n >= c.MIN_DETERMINATIONS

available.to_parquet(c.AVAILABILITY)
print(f"Patients with laboratory data: {len(available):,}")
print(f"With >=3 determinations of all 17 analytes: {available.all(axis=1).sum():,}")
print((available.mean() * 100).round(1).to_string())
