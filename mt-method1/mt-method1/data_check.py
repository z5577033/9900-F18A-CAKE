# Databricks notebook source
import joblib
import polars as pl
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append(r"/Workspace/9900-f18a-cake/working_branch/src")


CSV_PATH = "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked/MValue_concat.csv"
# CSV_PATH = "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked_subset_leukaemia_subsampled/MValue_polaris_pivot_0.csv"
# CSV_PATH = "/Volumes/cb_prod/comp9300-9900-f18b-cake/9900-f18b-cake/data/mvalue_outputs_masked/MValue_concat.csv"
JOBLIB_PATH = "/Workspace/9900-f18a-cake/working_branch/data/freeze0525/diseaseTree_mapped.joblib"
OUTPUT_PATH = "/Workspace/9900-f18a-cake/working_branch/data/freeze0525/biosample_alignment.csv"

ID_PATTERN = re.compile(r".*_T_.*_M$")  # 2J0D2U4J_T_XJAFP6HU_M

try:
    df_csv = pl.read_csv(CSV_PATH, columns=["biosample_id"], ignore_errors=True)
except Exception:
    df_csv = pl.read_csv(CSV_PATH, ignore_errors=True)
    if "biosample_id" not in df_csv.columns:
        for c in df_csv.columns:
            if c.lower() == "biosample_id":
                df_csv = df_csv.rename({c: "biosample_id"})
                break

csv_ids = [str(x).strip() for x in df_csv["biosample_id"].to_list() if x and str(x).strip()]
print(f"read csv done {len(csv_ids)} samples total")
print(csv_ids[:10])

def deep_scan_ids(obj, ids: set, _depth=0, _maxdepth=8):
    if obj is None or _depth > _maxdepth:
        return
    if isinstance(obj, str):
        if ID_PATTERN.match(obj):
            ids.add(obj)
        return
    if isinstance(obj, (list, tuple, set)):
        for it in obj:
            deep_scan_ids(it, ids, _depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and ID_PATTERN.match(k):
                ids.add(k)
            deep_scan_ids(v, ids, _depth + 1)
        return
    if hasattr(obj, "__dict__"):
        deep_scan_ids(vars(obj), ids, _depth + 1)

obj = joblib.load(JOBLIB_PATH)
joblib_ids = set()
deep_scan_ids(obj, joblib_ids)
joblib_ids = sorted(joblib_ids)

from pprint import pprint

print(type(obj))
pprint(obj)

print(f"read Joblib done {len(joblib_ids)} samples total")
print(joblib_ids[:10])


csv_set = set(csv_ids)
joblib_set = set(joblib_ids)

intersection = csv_set & joblib_set
csv_only = csv_set - joblib_set
joblib_only = joblib_set - csv_set

print(f"🔹 CSV count: {len(csv_set)}")
print(f"🔹 Joblib count: {len(joblib_set)}")
print(f"🔹 joint count: {len(intersection)}")
print(f"🔹 CSV only: {len(csv_only)}")
print(f"🔹 Joblib only: {len(joblib_only)}")



plt.figure(figsize=(6,5))
plt.bar(["CSV only", "Joblib only", "Align"], 
        [len(csv_only), len(joblib_only), len(intersection)],
        color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.ylabel("Sample cnt")
plt.title("Biosample ID joint")
for i, v in enumerate([len(csv_only), len(joblib_only), len(intersection)]):
    plt.text(i, v + 2, str(v), ha="center", fontweight="bold")
plt.show()

alignment_df = pd.DataFrame({
    "biosample_id_csv": list(csv_set),
    "in_joblib": [x in joblib_set for x in csv_set]
})

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
alignment_df.to_csv(OUTPUT_PATH, index=False)
print(f"Align output: {OUTPUT_PATH}")
alignment_df.head(10)


# COMMAND ----------

from pprint import pprint
import joblib
import polars as pl
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append(r"/Workspace/9900-f18a-cake/working_branch/src")

obj = joblib.load(JOBLIB_PATH)

print(type(obj))
pprint(obj)