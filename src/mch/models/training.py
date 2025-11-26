import os
import time
import math
import logging
import heapq
import sys
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import FunctionTransformer

# Ensure imports work correctly
sys.path.append("/Workspace/9900-f18a-cake/mt-method2/src")

from mch.models.differentialMethylationClassifier import DifferentialMethylation
from mch.config.settings import mvalue_df, main_tree, DATA_DIR
from mch.config.modelTrainingParameters import parameter_grid, resultsDirectory

# --- Logger Setup ---
logger = logging.getLogger("mch.training")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # Add StreamHandler to ensure logs appear in Databricks stdout (Notebook output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def _replace_inf(X):
    """Replace infinite values with 0 to prevent sklearn errors."""
    X = np.asarray(X, dtype=np.float64)
    mask = ~np.isfinite(X)
    X[mask] = 0.0 
    return X

def _prefilter_polars_chunked(df_pl: pl.DataFrame, topk: int, id_col: str = "biosample_id"):
    """
    Select top-k features based on variance.
    Simplified to scan only the first 5000 features to prevent OOM on large datasets.
    """
    feat_cols = [c for c in df_pl.columns if c != id_col]
    
    # If dataset is small or topk is not set, return all features
    if topk <= 0 or len(feat_cols) <= topk:
        return df_pl.select([id_col, *feat_cols]), feat_cols
    
    # Safety Limit: Only scan first 5000 columns to save memory and time
    scan_cols = feat_cols[:5000] 
    
    # Calculate variance
    var_df = df_pl.select([pl.col(c).var().alias(c) for c in scan_cols])
    vars_dict = var_df.to_dicts()[0]
    
    # Sort and keep top K
    sorted_cols = sorted(vars_dict.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    kept_cols = [k for k, v in sorted_cols[:topk]]
    
    return df_pl.select([id_col, *kept_cols]), kept_cols

class BatchModelTrainer:
    def __init__(self, tree=main_tree):
        self.tree = tree
        self.models = {}
        self.training_stats = {}
        self.filteredMValueFile = mvalue_df
        self.rf_n_jobs = int(os.getenv("RF_N_JOBS", "1"))
        self.rf_params = {
            "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "50")),
            "max_depth": int(os.getenv("RF_MAX_DEPTH", "10"))
        }
        # Default prefilter count (can be overridden by env vars)
        self.prefilter_topk = int(os.getenv("MCH_PREFILTER_TOPK", "200"))

    def train_all_models(self, save_dir: Optional[Path] = None, raise_on_error: bool = False) -> Dict:
        """
        Main loop to train models for each node in the tree.
        """
        target_node = os.getenv("MCH_ONLY_NODE") 
        nodes = [target_node] if target_node else self.tree.get_child_names()

        for node in nodes:
            try:
                print(f"\n [DEBUG] Starting training for node: {node}")
                
                # 1. Prepare Data
                result = self._prepare_node_data(node)
                if result is None: 
                    print(f" [DEBUG] Skipping {node} (Data preparation returned None).")
                    continue
                    
                nodeData, design = result
                print(f" [DEBUG] Data preparation complete. Samples: {len(nodeData)}, Features: {len(nodeData.columns)}")

                # 2. Prefilter Features
                print(f" [DEBUG] Prefiltering features (Top {self.prefilter_topk})...")
                nodeData_pf, kept_cols = _prefilter_polars_chunked(nodeData, topk=self.prefilter_topk)
                
                # 3. Convert to Pandas/Numpy for Sklearn
                # Using fillna(0) to handle any remaining NaNs safely
                X = nodeData_pf.select(kept_cols).to_pandas().fillna(0).values
                y = design["cancerType"].to_pandas().values
                
                # ==========================================================
                #  CRITICAL FIX: Drop classes with fewer than 2 samples
                # This prevents 'train_test_split' ValueError
                # ==========================================================
                unique_classes, class_counts = np.unique(y, return_counts=True)
                rare_classes = unique_classes[class_counts < 2]
                
                if len(rare_classes) > 0:
                    print(f" [DEBUG] Found rare classes (<2 samples), dropping: {rare_classes}")
                    # Create mask to keep only valid classes
                    valid_mask = ~np.isin(y, rare_classes)
                    X = X[valid_mask]
                    y = y[valid_mask]
                    print(f" [DEBUG] Samples remaining after cleanup: {len(y)}")

                # Check if we still have at least 2 classes for classification
                if len(np.unique(y)) < 2:
                    print(" [DEBUG] Skipped: Only 1 class remaining after cleanup. Cannot train classifier.")
                    continue
                # ==========================================================

                # 4. Train Test Split
                print(f" [DEBUG] Fitting Random Forest (Classes: {len(np.unique(y))})...")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # 5. Pipeline Construction
                rf = RandomForestClassifier(
                    n_estimators=self.rf_params["n_estimators"], 
                    max_depth=self.rf_params["max_depth"], 
                    n_jobs=self.rf_n_jobs, 
                    random_state=42
                )
                
                pipeline = Pipeline([
                    ("cleaner", FunctionTransformer(_replace_inf, validate=False)),
                    ("model", rf)
                ])
                
                pipeline.fit(X_train, y_train)
                
                # 6. Evaluation (Added F1 Metrics)
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                # Generate Full Classification Report to extract F1 Scores
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                metrics = {
                    "accuracy": float(acc),
                    "macro_f1": float(report["macro avg"]["f1-score"]),
                    "weighted_f1": float(report["weighted avg"]["f1-score"])
                }

                print(f" [RESULT] {node} Accuracy: {acc:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
                
                self.models[node] = pipeline
                self.training_stats[node] = {
                    "metrics": metrics, 
                    "estimator": pipeline,
                    "classes": list(np.unique(y))
                }

            except Exception as e:
                print(f" [ERROR] Error training {node}: {e}")
                import traceback
                traceback.print_exc()
                if raise_on_error: raise

        return self.training_stats

    def _prepare_node_data(self, node: str):
        """
        Filters the global mvalue_df to find samples belonging to the specific node
        and assigns labels based on its children (subtypes).
        """
        mvalue_df = self.filteredMValueFile
        diseaseTree = self.tree.find_node_by_name(node)
        if not diseaseTree: 
            print(f" [DEBUG] Node {node} not found in DiseaseTree.")
            return None
            
        # 1. Identify all samples belonging to this node (and its children)
        # Note: get_samples_recursive() must return a list or set of strings
        all_valid_samples = set(diseaseTree.get_samples_recursive())
        print(f" [DEBUG] {node} has {len(all_valid_samples)} total samples in tree definition.")
        
        # 2. Filter data in memory
        filteredData = mvalue_df.filter(pl.col("biosample_id").is_in(all_valid_samples))
        current_count = filteredData.height
        print(f" [DEBUG] {node} matched {current_count} samples in current memory dataset.")
        
        if current_count < 5: 
            print(f" [DEBUG] Too few samples (<5), skipping.")
            return None

        # 3. Labeling Logic (Create Design Matrix)
        # Default label is 'other'
        design = pl.DataFrame({
            "biosample_id": filteredData["biosample_id"], 
            "cancerType": ["other"] * filteredData.height
        })
        
        child_names = diseaseTree.get_child_names()
        print(f" [DEBUG] Attempting to split into subtypes: {child_names}")
        
        for child_name in child_names:
            child_node = diseaseTree.find_node_by_name(child_name)
            if not child_node: continue
            c_samples = child_node.get_samples_recursive()
            
            # If a sample belongs to this child, label it
            # We use a low threshold (>0) to allow any matching samples to be labeled
            if len(c_samples) > 0:
                mask = design["biosample_id"].is_in(c_samples)
                design = design.with_columns(
                    pl.when(mask)
                      .then(pl.lit(child_name))
                      .otherwise(pl.col("cancerType"))
                      .alias("cancerType")
                )
        
        # 4. Filter out 'other' (samples that belong to the parent but no known child)
        # We assume we only want to classify known subtypes
        design = design.filter(pl.col("cancerType") != "other")
        filteredData = filteredData.filter(pl.col("biosample_id").is_in(design["biosample_id"]))
        
        # Debugging the distribution
        final_counts = design["cancerType"].value_counts()
        print(f" [DEBUG] Final Class Distribution:\n{final_counts}")
        
        # Need at least 2 subgroups to classify
        if design["cancerType"].n_unique() < 2:
             print(" [DEBUG] Fewer than 2 subgroups found. Cannot classify.")
             return None
             
        # Fill NaNs to prevent issues downstream (Polars fill_null)
        return filteredData.fill_null(0.0), design