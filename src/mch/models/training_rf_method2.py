import os, time, math, logging, heapq, sys
from pathlib import Path
from typing import Dict, Optional, Any

import polars as pl
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import FunctionTransformer
import joblib

# ---------------------------------------------------------------------
# Make sure mt-method2 repo is on sys.path
# ---------------------------------------------------------------------
REPO_SRC = "/Workspace/9900-f18a-cake/mt-method2"
if f"{REPO_SRC}/src" not in sys.path:
    sys.path.append(f"{REPO_SRC}/src")

from mch.models.differentialMethylationClassifier import DifferentialMethylation
from mch.config.settings import mvalue_df, main_tree, DATA_DIR
from mch.config.modelTrainingParameters import parameter_grid, resultsDirectory

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger("mch.training")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(DATA_DIR / "logs" / "training.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------------------------------------------------------------
# Helper: clean feature matrix to avoid inf / 超大值
# ---------------------------------------------------------------------
def _sanitize_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is no inf / -inf and no values outside float32 range.
    This mimics the old behaviour where extreme values were clipped.
    """
    # 转成 float64 处理
    arr = np.asarray(df, dtype=np.float64)

    # 把 inf / -inf 变成 NaN
    bad_mask = ~np.isfinite(arr)
    if bad_mask.any():
        arr[bad_mask] = np.nan

    # 裁剪到 float32 能表示的范围内，避免 overflow
    float32_max = np.finfo(np.float32).max
    arr = np.clip(arr, -float32_max, float32_max)

    # 重新包成 DataFrame，保持 index / columns 一致
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


class BatchModelTrainer:
    """
    Legacy-style trainer: for each node in the disease tree, build a RandomForest
    model on the full feature set, with an optional DifferentialMethylation step.

    Key features:
    - No variance prefiltering (does not filter top-k features)
    - Does not discard rare classes (does not remove classes with <2 samples)
    - Uses train_test_split WITHOUT stratify (matches older RF version)
    - Process: All CpG features → DifferentialMethylation (optional) → RandomForest
    """

    def __init__(self, tree=main_tree):
        self.tree = tree
        self.models: Dict[str, RandomForestClassifier] = {}
        self.training_stats: Dict[str, Dict[str, Any]] = {}

        self.dataDirectory = DATA_DIR
        self.resultsDirectory = resultsDirectory
        self.filteredMValueFile = mvalue_df

        # Environment variables can override RF hyperparameters and CV parallelism
        self.disable_dm = (os.getenv("MCH_DISABLE_DM", "0") == "1")
        self.rf_n_jobs = int(os.getenv("RF_N_JOBS", "1"))
        self.cv_n_jobs = int(os.getenv("CV_N_JOBS", "1"))
        self.only_node = os.getenv("MCH_ONLY_NODE")

        self.rf_params = {
            "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "50")),
            "max_depth": int(os.getenv("RF_MAX_DEPTH", "10")),
        }

    # -----------------------------------------------------------------
    # Train all models
    # -----------------------------------------------------------------
    def train_all_models(
        self,
        save_dir: Optional[Path] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trains RandomForest models for all nodes in the disease tree and
        returns a dictionary of stats for each node.
        """
        nodes = self.tree.get_child_names()
        if self.only_node:
            nodes = [n for n in nodes if n == self.only_node]
            logger.info("Only training specified node: %s", self.only_node)

        for node in nodes:
            try:
                logger.info("Training model for node: %s", node)
                prepared = self._prepare_node_data(node)
                if prepared is None:
                    # Insufficient data, skip this node
                    continue

                # nodeData_pd: All features + biosample_id (pandas, already cleaned)
                # design_pd: biosample_id + cancerType (pandas)
                nodeData_pd, design_pd = prepared

                # X: Remove id column, keep only features
                X_all = nodeData_pd.drop(columns=["biosample_id"])
                y_all = design_pd["cancerType"]

                if y_all.nunique() < 2:
                    logger.warning(
                        "Skip %s: fewer than 2 classes (legacy RF mode)",
                        node,
                    )
                    continue

                # === train/test split (NO stratify, to match old RF version) ===
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all,
                    y_all,
                    test_size=0.2,
                    random_state=42,
                )
                logger.info(
                    "Train/Test shapes: X_train=%s, X_test=%s",
                    X_train.shape,
                    X_test.shape,
                )

                # ---------------- Model & pipeline ----------------
                rf = RandomForestClassifier(
                    n_estimators=self.rf_params["n_estimators"],
                    max_depth=self.rf_params["max_depth"],
                    random_state=42,
                    n_jobs=self.rf_n_jobs,
                )
                logger.info(
                    "Using RandomForestClassifier(n_estimators=%d, max_depth=%s)",
                    self.rf_params["n_estimators"],
                    str(self.rf_params["max_depth"]),
                )

                if self.disable_dm:
                    logger.info("DM disabled (legacy RF)")
                    pipeline = Pipeline([("modelGeneration", rf)])
                else:
                    logger.info("DM enabled (legacy RF)")
                    pipeline = Pipeline(
                        [
                            ("differentialMethylation", DifferentialMethylation()),
                            ("modelGeneration", rf),
                        ]
                    )
                logger.info("Pipeline steps: %s", list(pipeline.named_steps.keys()))

                # ---------------- CV settings (if parameter_grid is provided) ----------------
                if not parameter_grid:
                    stratified_cv = None
                else:
                    min_cls_train = y_train.value_counts().min()
                    n_splits = max(2, min(3, int(min_cls_train))) if min_cls_train >= 2 else 2
                    stratified_cv = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=42,
                    )

                # ---------------- Fitting ----------------
                if not parameter_grid:
                    logger.warning(
                        "parameter_grid is None/empty, fit pipeline without grid search (legacy RF)"
                    )
                    pipeline.fit(X_train, y_train)
                    est = pipeline
                    self.models[node] = rf
                else:
                    search = GridSearchCV(
                        pipeline,
                        param_grid=parameter_grid,
                        scoring="accuracy",
                        cv=stratified_cv,
                        verbose=2,
                        n_jobs=self.cv_n_jobs,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train)
                    est = search.best_estimator_
                    # Extract the best RF from the pipeline
                    self.models[node] = est.named_steps["modelGeneration"]

                # ---------------- Metrics ----------------
                metrics: Dict[str, float] = {}
                if X_test is not None:
                    y_pred = est.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    report = classification_report(
                        y_test,
                        y_pred,
                        output_dict=True,
                        zero_division=0,
                    )
                    metrics = {
                        "accuracy": float(acc),
                        "macro_f1": float(
                            report.get("macro avg", {}).get("f1-score", float("nan"))
                        ),
                        "weighted_f1": float(
                            report.get("weighted avg", {}).get("f1-score", float("nan"))
                        ),
                    }
                    if parameter_grid:
                        logger.info(
                            "[GS-legacy] Test accuracy: %.4f, macro_f1: %.4f, weighted_f1: %.4f",
                            metrics["accuracy"],
                            metrics["macro_f1"],
                            metrics["weighted_f1"],
                        )
                    else:
                        logger.info(
                            "Test accuracy (legacy RF): %.4f, macro_f1: %.4f, weighted_f1: %.4f",
                            metrics["accuracy"],
                            metrics["macro_f1"],
                            metrics["weighted_f1"],
                        )

                # ---------------- Feature importance (top 10) ----------------
                top_features = None
                try:
                    rf_model = est.named_steps.get("modelGeneration", None)
                    if hasattr(rf_model, "feature_importances_"):
                        importances = rf_model.feature_importances_
                        feature_names = list(X_all.columns)
                        idx = np.argsort(importances)[::-1][:10]
                        top_features = [
                            {
                                "feature": feature_names[i],
                                "importance": float(importances[i]),
                            }
                            for i in idx
                        ]
                except Exception:
                    top_features = None

                # ---------------- Record stats ----------------
                stats_entry: Dict[str, Any] = {
                    "n_samples_total": int(len(y_all)),
                    "n_samples_train": int(len(y_train)),
                    "n_samples_test": int(0 if X_test is None else len(y_test)),
                    "n_features": int(X_all.shape[1]),
                    "classes": sorted(map(str, y_all.unique())),
                    "metrics": metrics,
                    "top_features": top_features,
                    # For child notebook to log model with mlflow
                    "estimator": est,
                }
                if parameter_grid:
                    stats_entry["best_params"] = getattr(
                        locals().get("search", None),
                        "best_params_",
                        None,
                    )

                self.training_stats[node] = stats_entry

                # ---------------- Optional model saving ----------------
                if save_dir is not None:
                    out = Path(save_dir) / node
                    out.mkdir(parents=True, exist_ok=True)
                    self._save_model(node, est, out)
                    logger.info("Model trained for %s, save_dir=%s", node, save_dir)
                else:
                    logger.info("Model trained for %s, not saved", node)

            except Exception:
                logger.exception("Error training model for %s", node)
                if raise_on_error:
                    raise

        return self.training_stats

    # -----------------------------------------------------------------
    # Prepare data for a specific node (based on biosample_id)
    # -----------------------------------------------------------------
    def _prepare_node_data(self, node: str):
        """
        Prepare training data for a specific node.
        Uses the current mt-method2's mvalue_df (polars, containing biosample_id),
        but the process is kept as close as possible to the previous RandomForest version.

        Key points to match old RF code:
        - Require at least 10 samples per node
        - Require at least two distinct cancerType values
        - Drop columns with missing values (axis="columns"), not rows
        """
        print(f"{node} origin data count:", mvalue_df.height)
        print(f"{node} mvalue_df columns:", mvalue_df.columns[:5])

        # Initially set all to otherCancerType
        truthValues = ["otherCancerType"] * mvalue_df.height
        design = pl.DataFrame(
            {
                "biosample_id": mvalue_df["biosample_id"],
                "cancerType": truthValues,
            }
        )

        # Find the corresponding node in the tree
        diseaseTree = main_tree.find_node_by_name(node)
        if diseaseTree is None:
            logger.warning("%s: node not found in disease tree, skip", node)
            return None

        diseaseSamples = diseaseTree.get_samples_recursive()
        print(f"{node} diseaseTree count:", len(diseaseSamples))

        # For each child node, label samples with the corresponding cancerType (>=3 samples required)
        for cancer in diseaseTree.get_child_names():
            cancerTree = diseaseTree.find_node_by_name(cancer)
            if cancerTree is None:
                logger.warning("%s: child not found: %s, skip", node, cancer)
                continue
            try:
                samples = cancerTree.get_samples_recursive()
            except Exception as ce:
                logger.warning(
                    "%s: get_samples_recursive failed for child %s: %s",
                    node,
                    cancer,
                    ce,
                )
                continue

            if len(samples) >= 3:
                mask = design["biosample_id"].is_in(samples)
                design = design.with_columns(
                    pl.when(mask)
                    .then(pl.lit(cancer))
                    .otherwise(pl.col("cancerType"))
                    .alias("cancerType")
                )

        # Keep only samples that are in both mvalue_df and diseaseSamples
        filteredData = mvalue_df.filter(pl.col("biosample_id").is_in(diseaseSamples))
        design = design.filter(pl.col("biosample_id").is_in(filteredData["biosample_id"].to_list()))

        print(f"{node} Data after filter:", filteredData.height)
        print(f"{node} Data after design filter:", design.height)
        print(node, filteredData.height, design.height)

        # At least 10 samples required
        if filteredData["biosample_id"].len() < 10:
            print(f"Skipping, {node} has fewer than 10 samples")
            return None

        # At least two different cancerTypes required
        if len(design["cancerType"].unique()) < 2:
            print(f"Skipping, there is only one subgroup of {diseaseTree.name}")
            return None

        # === Match old RF behavior ===
        nodeData_pd = filteredData.to_pandas()
        nodeData_pd = _sanitize_features_df(nodeData_pd)
        nodeData_pd = nodeData_pd.dropna(axis="columns")

        design_pd = design.to_pandas()

        return nodeData_pd, design_pd

    # -----------------------------------------------------------------
    # Save and summarize
    # -----------------------------------------------------------------
    def _save_model(self, node: str, model: Any, save_dir: Path):
        """Save a trained model or pipeline to disk."""
        safe = node.replace("/", "_").replace(" ", "_")
        model_path = save_dir / f"{safe}_model.joblib"
        joblib.dump(model, model_path)

    def get_training_summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing all model training results."""
        return pd.DataFrame.from_dict(self.training_stats, orient="index")