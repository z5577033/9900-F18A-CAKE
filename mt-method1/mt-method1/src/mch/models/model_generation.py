"""
Model generation utilities for methylation classification models.
(mt-method1 / mch/models/model_generation.py)

变更要点：
- 统一参数网格前缀：无论 YAML 写的是 modelRefinement__/裸键，都会映射成 modelGeneration__
- Pipeline: DifferentialMethylation -> SimpleImputer -> (StandardScaler for SVM | passthrough for RF/XGB) -> modelGeneration
- 自适应CV：最小类样本数 < n_splits 时自动使用 StratifiedShuffleSplit
- 使用 sample_weight 提升少数类召回
- 加入 Guard/Debug：如仍出现 modelRefinement__ 或网格键与 Pipeline 不匹配，直接报错
"""

from __future__ import annotations

import os
import time
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

# === XGBoost（如未安装：%pip install xgboost） ===
from xgboost import XGBClassifier

from mch.models.differentialMethylationClassifier import DifferentialMethylation
from mch.data_processing.dataset_filtering import filter_problem_probes
from mch.utils.logging_utils import load_config as load_yaml_config

# settings 里只引入真实用到的符号（避免 FREEZE_NUMBER 导入报错）
from mch.config.settings import (
    main_tree,
    model_directory,
    tree_directory
)

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path('/app/data/logs/model_generation')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'model_generation_{timestamp}.log'

_file_handler = logging.FileHandler(log_file)
_file_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_file_handler.setFormatter(_formatter)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file) for h in logger.handlers):
    logger.addHandler(_file_handler)

# -----------------------------
# Helpers
# -----------------------------
def _pick_cv(y, n_splits_default: int = 3, seed: int = 42):
    """若最小类样本数 < n_splits，则改用 StratifiedShuffleSplit，避免某折缺类。"""
    vc = pd.Series(y).value_counts()
    min_count = int(vc.min()) if len(vc) else 0
    if min_count < n_splits_default:
        return StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    return StratifiedKFold(n_splits=n_splits_default, shuffle=True, random_state=seed)

def _normalize_param_grid(param_grid: Optional[Dict], step_name: str = "modelGeneration") -> Dict:
    """
    将 YAML 中以其他前缀（如 modelRefinement__）或裸参数写的网格键名
    统一映射为 `modelGeneration__...`，以匹配当前 Pipeline 步骤名。
    """
    if not param_grid:
        return {}
    fixed = {}
    for k, v in param_grid.items():
        if k.startswith("modelRefinement__"):
            suf = k.split("__", 1)[1]
            fixed[f"{step_name}__{suf}"] = v
        elif "__" in k:
            # 任何其他前缀 -> 统一替换为 modelGeneration__
            suf = k.split("__", 1)[1]
            fixed[f"{step_name}__{suf}"] = v
        else:
            # 裸键
            fixed[f"{step_name}__{k}"] = v
    return fixed

def load_model_config() -> dict:
    """Load model configuration (model_training_config.yaml)."""
    return load_yaml_config("model_training_config.yaml")

# -----------------------------
# Dataset construction
# -----------------------------
def make_dataset(filtered_data: pd.DataFrame, disease_tree) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    - 移除问题探针
    - 仅保留在当前节点子树中（且样本数≥3）的样本
    - ≤9 样本或只有 1 个子类 -> 返回 None
    """
    logger.info(f"Constructing dataset for {disease_tree.name}")

    filtered_data = filter_problem_probes(filtered_data)

    truth_values = pd.Series(["otherCancerType"] * len(filtered_data["sample_id"]))
    design = pd.DataFrame({"sample_id": filtered_data.sample_id, "cancerType": truth_values.values})

    for cancer in disease_tree.get_child_names():
        cancer_tree = disease_tree.find_node_by_name(cancer)
        samples = cancer_tree.get_samples_recursive()
        if len(samples) >= 3:
            design.loc[design['sample_id'].isin(samples), 'cancerType'] = cancer

    # 剔除不在本层次子类中的样本
    filtered_data = filtered_data[design.cancerType != "otherCancerType"]
    design = design[design.cancerType != "otherCancerType"]

    if len(filtered_data["sample_id"]) < 10:
        logger.warning(f"Skipping, {disease_tree.name} has fewer than 10 samples")
        return None
    
    # 去除含缺失的列
    filtered_data = filtered_data.dropna(axis="columns")

    if len(design["cancerType"].unique()) < 2:
        logger.warning(f"Skipping, there is only one subgroup of {disease_tree.name}")
        return None

    return filtered_data, pd.DataFrame({"cancerType": design.cancerType})

# -----------------------------
# Model config & factory
# -----------------------------
def get_model_config(model_type: str) -> Tuple[BaseEstimator, dict]:
    """Get model and its parameter grid based on model type."""
    configs = load_model_config()['model_configs']
    if model_type not in configs:
        msg = f"Unsupported model type: {model_type}. Supported types are: {list(configs.keys())}"
        logger.error(msg)
        raise ValueError(msg)
    
    model_config = configs[model_type]
    
    if model_type == 'svm':
        model = svm.SVC(**model_config['parameters'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_config['parameters'])
        # 若 YAML 未指定 class_weight，默认对不均衡更稳
        if 'class_weight' not in model_config['parameters']:
            model.set_params(class_weight='balanced_subsample')
    elif model_type == 'xgboost':
        model = XGBClassifier(**model_config['parameters'])
    else:
        msg = f"Model type {model_type} is in config but not implemented"
        logger.error(msg)
        raise ValueError(msg)
    
    return model, model_config['parameter_grid']

# -----------------------------
# Orchestration per tree level
# -----------------------------
def run_level(all_data: pd.DataFrame, tree, results_dir: Optional[Path] = None) -> None:
    """Run model generation for a specific level in the disease tree."""
    logger.info(f"Running model generation for {tree.name}")
    
    # Get results directory from config if not provided
    if results_dir is None:
        config = load_model_config()
        results_dir = Path(config['resultsDirectory'])
    
    # Create dataset for this level
    data_construction = make_dataset(all_data, tree)
    if data_construction is None:
        logger.warning(f"Skipping as no dataset was constructed for {tree.name}")
        return
    
    dataset, design = data_construction
    unique_types = design.cancerType.unique()
    logger.info(f"Dataset created: {dataset.shape}")
    logger.info(f"There are {len(unique_types)} sub categories in {tree.name} that will be used")
    logger.info(f"Categories: {unique_types}")
    
    # Get model configuration
    config = load_model_config()
    model_type = config.get('default_model_type', 'svm')
    
    # Run grid search
    model, validation_samples = run_grid_search(
        filtered_data=dataset,
        design=design,
        model_type=model_type,
        name=tree.name
    )
    
    # Store validation samples in tree
    tree.validation_samples = validation_samples
    
    # Create output directories if they don't exist
    model_dir = results_dir / 'trees'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tree with validation samples, including model type in filename
    model_path = model_dir / f"model-{model_type}-{tree.name}.joblib"
    tree_path = model_dir / f"diseaseTree-{model_type}-{tree.name}.joblib"
    
    logger.info(f"Saving {model_type} model to {model_path}")
    joblib.dump(model, model_path)
    
    logger.info(f"Saving tree with validation samples to {tree_path}")
    joblib.dump(tree, tree_path)
    
    logger.info(f"Completed model generation for {tree.name}")

# -----------------------------
# Grid search per node
# -----------------------------
def run_grid_search(
    filtered_data: pd.DataFrame,
    design: pd.DataFrame,
    model_type: str,
    custom_param_grid: Optional[dict] = None,
    name: str = ""
) -> Tuple[Pipeline, pd.Series]:
    """
    - 丢弃含缺失的列、统一列名为字符串
    - 分层划分 train/test（test_size=0.1，给训练集多留少数类）
    - Pipeline: DifferentialMethylation -> Imputer -> (Scaler?)-> Model
        * 仅 SVM 使用 StandardScaler，RF/XGB 设为 'passthrough'
    - GridSearchCV:
        * CV 用 _pick_cv 自适应
        * scoring 若是 accuracy，自动切到 f1_macro
        * 参数网格统一映射为 modelGeneration__ 前缀
        * 使用 sample_weight 提升少数类召回
    """
    logger.info(f"Starting grid search for model type: {model_type}")
    logger.info(f"Initial training dataset shape: {filtered_data.shape}")
    
    df = filtered_data.dropna(axis="columns")
    df.columns = df.columns.astype(str)
    logger.info(f"Actual training dataset shape: {df.shape}")

    # 更小的测试集比例，给训练集多保留少数类样本
    logger.info("Splitting data for training/validation")
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        design["cancerType"],
        test_size=0.1,
        random_state=42,
        stratify=design["cancerType"]
    )
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # 类型规整
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    logger.info(f"Setting up {model_type} model")
    model, default_param_grid = get_model_config(model_type)
    raw_grid = custom_param_grid if custom_param_grid is not None else default_param_grid

    logger.info("Defining pipeline components")
    differential_methylation = DifferentialMethylation()

    # GridSearch 配置
    cfg = load_model_config()
    grid_search_params = cfg['grid_search_config']

    # 自适应 CV
    stratified_cv = _pick_cv(y_train, n_splits_default=grid_search_params['n_splits'], seed=42)

    # 评分指标：accuracy -> 自动切换到 f1_macro
    scoring = grid_search_params.get('scoring', 'f1_macro')
    if scoring == 'accuracy':
        logger.warning("GridSearch scoring='accuracy' -> 自动切换为 'f1_macro' 以适应不均衡数据")
        scoring = 'f1_macro'

    # 仅 SVM 使用 scaler，其它模型不需要
    scaler = StandardScaler() if model_type == 'svm' else 'passthrough'

    logger.info("Defining pipeline")
    pipeline = Pipeline([
        ("differentialMethylation", differential_methylation),
        ("imp", SimpleImputer(strategy="median")),   # 防 NaN/Inf
        ("scaling", scaler),
        ("modelGeneration", model)                   # 步骤名与 YAML 前缀保持一致
    ])

    # 统一参数网格前缀（把 modelRefinement__/裸键 映射为 modelGeneration__）
    param_grid = _normalize_param_grid(raw_grid, step_name="modelGeneration")

    # === Guard & Debug ===
    steps_names = [name for name, _ in pipeline.steps]
    pipe_keys = set(pipeline.get_params().keys())
    grid_keys = set(param_grid.keys())
    unknown = [k for k in grid_keys if k not in pipe_keys]

    print("[DEBUG] pipeline steps:", steps_names)
    print("[DEBUG] CV type:", type(stratified_cv).__name__)
    print("[DEBUG] grid keys sample:", list(grid_keys)[:8])
    print("[DEBUG] unknown grid keys:", unknown)

    if any(k.startswith("modelRefinement__") for k in grid_keys):
        raise RuntimeError("[GUARD] param_grid 仍包含 'modelRefinement__'，映射未生效。请检查导入的 run_grid_search 是否为最新版。")
    if unknown:
        raise RuntimeError(f"[GUARD] param_grid 含未知键：{unknown}（与 pipeline 参数不匹配）")

    # 样本权重：比 class_weight 更细（多分类不均衡时更稳）
    sample_weight = compute_sample_weight('balanced', y_train)

    # GridSearch
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,                     # 确保是映射后的网格
        scoring=scoring,
        cv=stratified_cv,
        verbose=grid_search_params.get('verbose', 2),
        n_jobs=grid_search_params.get('n_jobs', -1),
        error_score='raise'                        # 强制抛错，便于定位
    )
    
    logger.info(f"Fitting {model_type} models (cv={type(stratified_cv).__name__}, scoring={scoring})")
    # 将样本权重传入最终模型步骤
    search.fit(X_train, y_train, **{'modelGeneration__sample_weight': sample_weight})

    best_model = search.best_estimator_
    logger.info(f"Best model parameters: {search.best_params_}")
    logger.info(f"Best cross-validation ({scoring}) score: {search.best_score_:.4f}")

    return best_model, y_test
