"""
Model generation utilities for methylation classification models.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import time
import json
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- Internal imports
from mch.models.differentialMethylationClassifier import DifferentialMethylation
from mch.data_processing.dataset_filtering import filter_problem_probes
from mch.utils.logging_utils import load_config  # <-- keep only load_config
from mch.config.settings import (
    FREEZE_NUMBER, 
    FREEZE,
    FREEZE_DIR,
    main_tree, 
    model_directory,
    tree_directory
)

# ---- Logging (avoid name collisions with stdlib logging)
import logging as _logging
logger = _logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(_logging.INFO)
    _log_dir = Path("/app/data/logs/model_generation")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _fh = _logging.FileHandler(_log_dir / f"model_generation_{_ts}.log")
    _fh.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_fh)


# -------------------------------------------------------------------
# Small utility transformers to keep DM happy & preserve column names
# -------------------------------------------------------------------

class EnsureDataFrame(BaseEstimator, TransformerMixin):
    """Ensure X is a pandas DataFrame; preserve column names when possible."""
    def fit(self, X, y=None):
        self._cols = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self._cols)

class ReplaceInf(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN while preserving DataFrame type and columns."""
    def fit(self, X, y=None):
        self._cols = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.replace([np.inf, -np.inf], np.nan)
        X = np.asarray(X)
        X[np.isinf(X)] = np.nan
        return pd.DataFrame(X, columns=self._cols) if self._cols else X


# ------------------------------
# Config helpers & model factory
# ------------------------------

def load_model_config() -> dict:
    """Load model configuration from YAML file (via project loader)."""
    return load_config("model_training_config.yaml")

def get_model_config(model_type: str) -> Tuple[object, Dict]:
    """Return an estimator instance and its param grid from config."""
    cfg = load_model_config().get("model_configs", {})
    if model_type not in cfg:
        supported = list(cfg.keys())
        msg = f"Unsupported model type: {model_type}. Supported types: {supported}"
        logger.error(msg)
        raise ValueError(msg)

    model_cfg = cfg[model_type]
    params = model_cfg.get("parameters", {}) or {}

    if model_type == "svm":
        model = svm.SVC(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Model type {model_type} is in config but not implemented")

    return model, model_cfg.get("parameter_grid", {})


# ---------------------------
# Dataset construction logic
# ---------------------------

def make_dataset(filtered_data: pd.DataFrame, disease_tree) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build a dataset and design frame for a given disease tree node.

    Returns (filtered_data, design) or None if checks fail.
    """
    logger.info("Constructing dataset for %s", getattr(disease_tree, "name", "<unknown>"))

    # Drop problematic probes first
    filtered_data = filter_problem_probes(filtered_data)

    # Build design (labels)
    truth_values = pd.Series(["otherCancerType"] * len(filtered_data["sample_id"]))
    design = pd.DataFrame({"sample_id": filtered_data.sample_id, "cancerType": truth_values.values})

    for cancer in disease_tree.get_child_names():
        cancer_tree = disease_tree.find_node_by_name(cancer)
        samples = cancer_tree.get_samples_recursive()
        if len(samples) >= 3:
            mask = design["sample_id"].isin(samples)
            design.loc[mask, "cancerType"] = cancer

    # Keep only rows used in this cohort
    keep_mask = design["cancerType"] != "otherCancerType"
    filtered_data = filtered_data[keep_mask]
    design = design[keep_mask]

    # Basic quality checks
    if len(filtered_data["sample_id"]) < 10:
        logger.warning("Skipping, %s has fewer than 10 samples", disease_tree.name)
        return None

    # remove fully-null columns
    filtered_data = filtered_data.dropna(axis="columns")

    if design["cancerType"].nunique() < 2:
        logger.warning("Skipping, there is only one subgroup of %s", disease_tree.name)
        return None

    # Return features + labels (design only needs the y column)
    return filtered_data, pd.DataFrame({"cancerType": design["cancerType"]})


# -------------------------
# Training / search helpers
# -------------------------

def run_grid_search(
    filtered_data: pd.DataFrame,
    design: pd.DataFrame,
    model_type: str,
    custom_param_grid: Optional[dict] = None,
    name: str = "",
) -> Tuple[Pipeline, pd.Series]:
    """
    Train a model with (optional) grid search:

    - Ensures DM receives a DataFrame
    - Replaces infs
    - Scales with with_mean=False (safe for wide data / sparsity)
    - Performs GridSearchCV only if param grid is non-empty; otherwise .fit directly
    """
    logger.info("Starting training for model type: %s", model_type)
    logger.info("Initial training dataset shape: %s", filtered_data.shape)

    df = filtered_data.dropna(axis="columns")
    df.columns = df.columns.astype(str)
    logger.info("Actual training dataset shape after dropna: %s", df.shape)

    logger.info("Splitting data for training/validation")
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        design["cancerType"],
        test_size=0.2,
        random_state=42,
        stratify=design["cancerType"],
    )

    model, default_param_grid = get_model_config(model_type)
    param_grid = custom_param_grid if custom_param_grid is not None else default_param_grid

    # pipeline: keep DataFrame -> replace inf -> DM -> scale -> model
    dm = DifferentialMethylation()

    pipeline = Pipeline(
        steps=[
            ("ensure_df", EnsureDataFrame()),
            ("replace_inf", ReplaceInf()),
            ("differentialMethylation", dm),
            ("scaling", StandardScaler(with_mean=False)),
            ("modelGeneration", model),
        ]
    )

    cfg = load_model_config()
    gs_cfg = cfg.get("grid_search_config", {})
    n_splits = int(gs_cfg.get("n_splits", 3))
    scoring = gs_cfg.get("scoring", "accuracy")
    verbose = int(gs_cfg.get("verbose", 0))
    n_jobs = int(gs_cfg.get("n_jobs", 1))
    error_score = gs_cfg.get("error_score", "raise")

    # Fit with/without grid search
    if isinstance(param_grid, dict) and len(param_grid) > 0:
        logger.info("Running GridSearchCV with param grid (%d keys)", len(param_grid))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            error_score=error_score,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        logger.info("Best params: %s", search.best_params_)
        logger.info("Best CV score: %.4f", search.best_score_)
    else:
        logger.info("No/empty grid provided. Fitting pipeline directly.")
        best_model = pipeline.fit(X_train, y_train)

    return best_model, y_test


# -------------------------
# One-level training runner
# -------------------------

def run_level(all_data: pd.DataFrame, tree, results_dir: Optional[Path] = None) -> None:
    """
    Train for one tree node and persist:
      - model (joblib)
      - disease tree with validation_samples
    """
    logger.info("Running model generation for %s", tree.name)

    if results_dir is None:
        results_dir = Path(load_model_config().get("resultsDirectory"))

    ds = make_dataset(all_data, tree)
    if ds is None:
        logger.warning("Skipping; no dataset constructed for %s", tree.name)
        return

    dataset, design = ds
    uniq = design["cancerType"].unique()
    logger.info("Dataset shape: %s; categories in %s: %s", dataset.shape, tree.name, uniq)

    cfg = load_model_config()
    model_type = cfg.get("default_model_type", "svm")

    model, val_y = run_grid_search(
        filtered_data=dataset,
        design=design,
        model_type=model_type,
        name=tree.name,
    )

    # capture validation samples for downstream use
    tree.validation_samples = val_y

    out_dir = Path(results_dir) / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"model-{model_type}-{tree.name}.joblib"
    tree_path = out_dir / f"diseaseTree-{model_type}-{tree.name}.joblib"

    logger.info("Saving %s model to %s", model_type, model_path)
    joblib.dump(model, model_path)

    logger.info("Saving tree (with validation samples) to %s", tree_path)
    joblib.dump(tree, tree_path)

    logger.info("Completed model generation for %s", tree.name)
