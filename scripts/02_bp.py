import pandas as pd
import config as c

# Blood pressure from omr: every row whose result_name starts with "Blood Pressure" (position-specific included);
# result_value is "systolic/diastolic"; duplicates are not removed
omr = pd.read_csv(c.OMR, usecols=["subject_id", "result_name", "result_value"])
omr = omr[omr["result_name"].str.startswith("Blood Pressure", na=False)]

parts = omr["result_value"].astype(str).str.split("/", n=1, expand=True)
bp = pd.DataFrame({"subject_id": omr["subject_id"].astype("int32"),
                   "s": pd.to_numeric(parts[0], errors="coerce"),
                   "d": pd.to_numeric(parts[1], errors="coerce")})
invalid = bp[["s", "d"]].isna().any(axis=1)
bp = bp[~invalid]

bp.to_parquet(c.BP_PARQUET)
print(f"'Blood Pressure*' rows: {len(omr):,} | not parseable: {invalid.sum():,}")
print(f"Readings saved: {len(bp):,} | patients: {bp['subject_id'].nunique():,}")
