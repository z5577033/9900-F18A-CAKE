# Databricks notebook source
from pathlib import Path
import rpy2.robjects as ro
ro.r('options(rpy2_quiet=TRUE)')
print(ro.r('R.version.string'))  # rpy status check
ro.r('library(limma); library(arrow)')  
ro.r['source'](str(Path('/app/src/mch/models/differentialMethylation.R')))
print(ro.r('exists("runDM")'))  # load R library check: TRUE
print(ro.r('Sys.getenv("R_HOME")'))

# COMMAND ----------

from mch.models.training import BatchModelTrainer
from mch.config.modelTrainingParameters import resultsDirectory, parameter_grid

# COMMAND ----------

models = BatchModelTrainer()
stats = models.train_all_models(raise_on_error=True)
stats