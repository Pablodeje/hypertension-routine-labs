import pandas as pd
import config as c

# Extract our item IDs with a numeric value from labevents, reading in chunks
analyte_of = {i: a for a, ids in c.ITEMIDS.items() for i in ids}

parts = []
n_read = 0
for chunk in pd.read_csv(c.LABEVENTS, usecols=["subject_id", "itemid", "valuenum"],
                         dtype={"subject_id": "int32", "itemid": "int32", "valuenum": "float64"},
                         chunksize=10_000_000):
    n_read += len(chunk)
    b = chunk[chunk["itemid"].isin(analyte_of.keys()) & chunk["valuenum"].notna()]
    parts.append(pd.DataFrame({"subject_id": b["subject_id"],
                               "analyte": b["itemid"].map(analyte_of),
                               "value": b["valuenum"]}))
    print(f"  rows read: {n_read:,}")

labs = pd.concat(parts, ignore_index=True)
labs["analyte"] = labs["analyte"].astype("category")
labs.to_parquet(c.LABS_PARQUET)
print(f"\nDeterminations saved: {len(labs):,} | patients: {labs['subject_id'].nunique():,}")
print(labs["analyte"].value_counts().to_string())
