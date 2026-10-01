import pandas as pd
import config as c

# Plausibility exclusions. Limits are evaluated among patients with an admission record;
# excluded patients are removed from the full cohort.

# ---------------- Load ----------------
full = pd.read_parquet(c.COHORT).reset_index()
df = full.set_index("subject_id")
df = df[df["Admission"].astype(str).str.strip() == "YES"]
df = df.drop(columns=[col for col in df.columns if col.endswith("_variance")])

X = df.drop(columns=[c.TARGET] + c.NON_PREDICTORS).astype(float)
mask = X.isna().any(axis=1) | df[c.TARGET].isna()
X = X[~mask]
print(f"Patients (complete cases): {len(X)}\n")

# ---------------- Count ----------------
rows = []
flags = {}
for v, (lower, upper) in c.LIMITS.items():
    vmin, vmax = X[f"{v}_min"], X[f"{v}_max"]
    below = vmin < lower if lower is not None else pd.Series(False, index=X.index)
    above = vmax > upper if upper is not None else pd.Series(False, index=X.index)
    zero = (vmin == 0) if v not in c.NO_ZERO_RULE else pd.Series(False, index=X.index)
    any_flag = below | above | zero
    flags[v] = any_flag
    rows.append({"Variable": v, "Lower limit": lower, "Upper limit": upper,
                 "Below": int(below.sum()), "Above": int(above.sum()),
                 "Zero": int(zero.sum()), "Total": int(any_flag.sum())})

flags = pd.DataFrame(flags)
n_flags = flags.sum(axis=1)
excluded = n_flags > 0

table = pd.DataFrame(rows)
table["This variable only"] = [int((flags[v] & (n_flags == 1)).sum()) for v in table["Variable"]]
table = table.sort_values("Total", ascending=False)

pd.set_option("display.width", 200)
print(table.to_string(index=False))

n_exc = int(excluded.sum())
print(f"\nPatients excluded (any limit): {n_exc} of {len(X)} ({100 * n_exc / len(X):.2f}%)")
print(f"Resulting cohort: {len(X) - n_exc}")

# ---------------- Save ----------------
excluded_ids = excluded[excluded].index
clean = full[~full["subject_id"].isin(excluded_ids)]
clean.to_excel(c.COHORT_CLEAN, index=False)
print(f"Saved: {len(clean)} rows")
