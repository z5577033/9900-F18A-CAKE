# README
This branch focuses on minimal modifications required to make the MCH pipeline runnable end-to-end.
It resolves conflicts, fixes broken paths/imports, adds missing resources, and introduces a lightweight feature-selection mode for constrained environments.

## Branch info
The full workflow was running on databricks, don't commit any changes on this branch (or other branch) since all modify should be done on databricks and be automaticly updated to github.
To change work directory, go src/mch/config/setting.py

The corresponding Databricks workspace directory is:

/Workspace/9900-F18A-CAKE/working_branch

## Running instruction
how to run: 

python /app/src/mch/models/run_training.py 

--raise-on-error                                                                         "don't jump out when some node have error"

--disable-dm                                                                             "if something wrong with R enviroment or memory is not adequate (less than 64 GB with 20GB dataset)"

--rf-n-jobs 1 --cv-n-jobs 1 --only-node "Haematological malignancy"                      "similair, if memory is not adequate"

--prefilter-topk 50 --prefilter-scan-max 3000 --prefilter-chunk-size 500                 "if dm was disabled, then using this to do the feature filter"

Or, it may can run in /app/src/mch/models/Training_model.ipynb
