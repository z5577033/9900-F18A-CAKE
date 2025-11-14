# Databricks notebook source
import os,sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

print("Thread limits set.")

# COMMAND ----------

raise_on_error = True   # --raise-on-error    
disable_dm = True     #  --disable-dm     
rf_n_jobs = 1           #  --rf-n-jobs 1   
cv_n_jobs = 1           #  --cv-n-jobs 1     
only_node = "Haematological malignancy"        #  --only-node "Haematological malignancy"   
prefilter_topk = 200    #  --prefilter-topk 50 
prefilter_scan_max = 5000   # --prefilter-scan-max 3000 
prefilter_chunk_size = 1000     # --prefilter-chunk-size 500

if disable_dm:
    os.environ["MCH_DISABLE_DM"] = "1"

os.environ["RF_N_JOBS"] = str(rf_n_jobs)
os.environ["CV_N_JOBS"] = str(cv_n_jobs)
os.environ["MCH_PREFILTER_TOPK"] = str(prefilter_topk)
os.environ["MCH_PREFILTER_SCAN_MAX"] = str(prefilter_scan_max)
os.environ["MCH_PREFILTER_CHUNK_SIZE"] = str(prefilter_chunk_size)
if only_node:
    os.environ["MCH_ONLY_NODE"] = only_node

print("Environment variables set.")

# COMMAND ----------

import polars as pl
test = pl.read_csv(
    "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data/mvalue_outputs_masked_subset_leukaemia_subsampled/MValue_polaris_pivot_0.csv"
)
print(test.height, test.width)
print(test.head(2))


# COMMAND ----------

# MAGIC %md
# MAGIC # Guys use the old interactive compute, I have set it up to autoamtically install all the dependencies needed. You shouldn't even need to run the sudo install at the top. I have a init script running. It should be all ready for you to use. You can go on compute to change the node type if you need a bigger machine. Im going to destroy this single user one we made- James
# MAGIC

# COMMAND ----------

sys.path.append(r"/Workspace/9900-f18a-cake/working_branch/src")
from mch.models.training import BatchModelTrainer

print("Starting training...")
trainer = BatchModelTrainer()
stats = trainer.train_all_models(raise_on_error=raise_on_error)

print("Training finished!")
display(stats) 