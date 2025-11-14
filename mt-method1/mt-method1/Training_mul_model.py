# Databricks notebook source
import os, sys
from pathlib import Path

os.environ["MCH_DATA_DIR"] = "/Volumes/cb_prod/comp9300-9900-f18a-cake/9900-f18a-cake/data"
os.environ["MCH_FREEZE"] = "freeze0525"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SRC_PATH = "/Workspace/9900-f18a-cake/mt-method1/src"
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
print("[sys.path ok]:", SRC_PATH in sys.path)

# COMMAND ----------

import logging

for name in ("py4j", "py4j.clientserver"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
    try:
        lg.handlers.clear()
    except Exception:
        pass

logging.getLogger().setLevel(logging.WARNING)

spark.sparkContext.setLogLevel("ERROR")
jlog = spark._jvm.org.apache.log4j
jlog.LogManager.getLogger("py4j").setLevel(jlog.Level.OFF)
jlog.LogManager.getLogger("py4j.clientserver").setLevel(jlog.Level.OFF)
jlog.LogManager.getRootLogger().setLevel(jlog.Level.WARN)

for n in ["org.apache.spark", "org.sparkproject", "com.databricks", "akka", "stdout", "stderr"]:
    jlog.LogManager.getLogger(n).setLevel(jlog.Level.WARN)

# COMMAND ----------

import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight  # ✅ 新增

from mch.config.settings import mvalue_df, main_tree

print("[IMPORT] All libraries loaded")

# COMMAND ----------

NODE_NAME = main_tree.name
TOPK_FEATURES = 200
TEST_SIZE = 0.2
TOPK_FEATURES = 500
RANDOM_STATE = 42
MODEL_LIST = ["random_forest", "svm", "xgboost"]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"/app/data/run_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[PARAM] Node = {NODE_NAME}")
print(f"[PARAM] Models = {MODEL_LIST}")
print(f"[PARAM] Output = {OUTPUT_DIR}")

# COMMAND ----------

import math, heapq

def prefilter_topk_by_variance_pl(df_pl, topk, id_col="biosample_id",
                                  scan_max=20000, chunk_size=5000):
    feat_cols = [c for c in df_pl.columns if c != id_col]
    feat_cols = feat_cols[:min(scan_max, len(feat_cols))]
    heap = []
    for i in range(0, len(feat_cols), chunk_size):
        cols = feat_cols[i:i+chunk_size]
        var_row = df_pl.select([pl.col(c).cast(pl.Float64).var().alias(c) for c in cols]).to_dicts()[0]
        for c, v in var_row.items():
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            if len(heap) < topk:
                heapq.heappush(heap, (v, c))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, c))
    kept_cols = [c for _, c in sorted(heap, key=lambda x: x[0], reverse=True)]
    return df_pl.select([id_col, *kept_cols]), kept_cols

print("[FUNC] Feature selection ready")

# COMMAND ----------

def build_dataset_for_node(node_name, topk=200):
    disease_tree = main_tree.find_node_by_name(node_name)
    disease_samples = set(disease_tree.get_samples_recursive())
    design = pl.DataFrame({
        "biosample_id": mvalue_df["biosample_id"],
        "cancerType": ["otherCancerType"] * mvalue_df.height
    })
    for cancer in disease_tree.get_child_names():
        child = disease_tree.find_node_by_name(cancer)
        if child is None: continue
        samples = set(child.get_samples_recursive())
        if len(samples) >= 3:
            mask = pl.col("biosample_id").is_in(list(samples))
            design = design.with_columns(
                pl.when(mask).then(pl.lit(cancer)).otherwise(pl.col("cancerType")).alias("cancerType")
            )

    filtered = mvalue_df.filter(pl.col("biosample_id").is_in(list(disease_samples)))
    design = design.filter(pl.col("biosample_id").is_in(filtered["biosample_id"].to_list()))
    filtered_pf, kept_cols = prefilter_topk_by_variance_pl(filtered, topk=topk, id_col="biosample_id")

    y_all = design["cancerType"].to_pandas()
    X_all = filtered_pf.select(kept_cols).to_pandas().astype("float32")

    cls_counts = y_all.value_counts()
    rare = cls_counts[cls_counts < 2].index.tolist()
    if rare:
        mask = ~y_all.isin(rare)
        X_all, y_all = X_all[mask], y_all[mask]

    X_all.index = filtered_pf["biosample_id"].to_pandas()[:len(X_all)]
    classes = sorted(list(y_all.unique()))
    return X_all, y_all, classes

print("[FUNC] Dataset builder ready")

# COMMAND ----------

def get_model_and_grid(model_type: str):
    if model_type == "random_forest":
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        grid = {
            "modelGeneration__n_estimators": [100, 200],
            "modelGeneration__max_depth": [None, 10, 20]
        }
        return model, grid, False
    
    if model_type == "svm":
        model = svm.SVC(
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        grid = {
            "modelGeneration__C": [0.1, 1, 10],
            "modelGeneration__kernel": ["linear", "rbf"]
        }
        return model, grid, True
    
    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist"
        )
        grid = {
            "modelGeneration__max_depth": [4, 6, 8],
            "modelGeneration__learning_rate": [0.05, 0.1]
        }
        return model, grid, False
    
    raise ValueError(f"Unknown model: {model_type}")

print("[FUNC] Model config ready")

# COMMAND ----------

def _pick_cv(y, n_splits_default=3, seed=42):
    vc = pd.Series(y).value_counts()
    min_count = int(vc.min()) if len(vc) else 0
    print(f"[CV Selection] Min class count: {min_count}")
    
    if min_count < n_splits_default:
        print(f"[CV] Using StratifiedShuffleSplit (少数类 < {n_splits_default})")
        return StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    else:
        print(f"[CV] Using StratifiedKFold(n_splits={n_splits_default})")
        return StratifiedKFold(n_splits=n_splits_default, shuffle=True, random_state=seed)

print("[FUNC] Adaptive CV ready")

# COMMAND ----------

print("[DATA] Loading dataset...")
X_all, y_all, classes = build_dataset_for_node(NODE_NAME, topk=TOPK_FEATURES)

print("[SPLIT] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)

print(f"[DATA] X={X_all.shape}, y={y_all.shape}, Classes={classes}")
print(f"[SPLIT] Train={X_train.shape}, Test={X_test.shape}")
print(f"[CLASSES] {', '.join(classes)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Guys use the old interactive compute, I have set it up to autoamtically install all the dependencies needed. You shouldn't even need to run the sudo install at the top. I have a init script running. It should be all ready for you to use. You can go on compute to change the node type if you need a bigger machine. Im going to destroy this single user one we made- James
# MAGIC

# COMMAND ----------

estimator, grid, need_scaler = get_model_and_grid('svm')
print(f"   Grid keys: {list(grid.keys())}")
print("   ✅ 参数名正确！\n")

steps = []
if need_scaler:
    steps.append(("scaling", StandardScaler()))
steps.append(("modelGeneration", estimator))
test_pipe = Pipeline(steps)
pipe_params = list(test_pipe.get_params().keys())
if any('modelGeneration__' in k for k in pipe_params):
    print("   ✅ Pipeline 步骤名正确！\n")

sample_weight = compute_sample_weight('balanced', y_train)
print(f"   Sample weight shape: {sample_weight.shape}")
print(f"   ✅ 样本权重计算正确！\n")

cv = _pick_cv(y_train, n_splits_default=3, seed=42)
print(f"   CV type: {type(cv).__name__}")
print("   ✅ CV 选择正确！\n")



# COMMAND ----------

from IPython.display import display
from sklearn.preprocessing import LabelEncoder
import time  # ⭐ 新增

log_file = OUTPUT_DIR / f"training_log_{TIMESTAMP}.txt"
analysis_summary_file = OUTPUT_DIR / f"analysis_summary_{TIMESTAMP}.txt"
timing_file = OUTPUT_DIR / f"training_timing_{TIMESTAMP}.txt"  # ⭐ 新增


sample_weight = compute_sample_weight('balanced', y_train)
print(f"[INFO] Sample weight shape: {sample_weight.shape}")
print(f"[INFO] Sample weight examples: {sample_weight[:5]}")


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
print(f"[INFO] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

results_summary = []
analysis_pack_paths = {}
timing_results = []


cv = _pick_cv(y_train, n_splits_default=3, seed=RANDOM_STATE)


log_content = []
log_content.append("=" * 80)
log_content.append(f"Training Log - {TIMESTAMP}")
log_content.append("=" * 80)
log_content.append(f"\nNode: {NODE_NAME}")
log_content.append(f"Features: {TOPK_FEATURES}")
log_content.append(f"Models: {MODEL_LIST}")
log_content.append(f"Train size: {X_train.shape}")
log_content.append(f"Test size: {X_test.shape}")
log_content.append(f"Classes: {classes}")
log_content.append(f"\n" + "=" * 80 + "\n")


training_start_time = time.time()

for model_type in MODEL_LIST:
    print(f"\n===== 🚀 Training {model_type} =====")
    log_content.append(f"\n===== Training {model_type} =====\n")

    model_start_time = time.time()
    
    estimator, grid, need_scaler = get_model_and_grid(model_type)

    steps = []
    if need_scaler:
        steps.append(("scaling", StandardScaler()))
    steps.append(("modelGeneration", estimator))
    pipe = Pipeline(steps)

    gridcv = GridSearchCV(
        pipe,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=2
    )

    gs_start_time = time.time()

    print("[INFO] Fitting with sample weights...")

    if model_type == "xgboost":
        try:
            gridcv.fit(X_train, y_train_encoded, 
                       **{'modelGeneration__sample_weight': sample_weight})
        except Exception as e:
            print(f"[ERROR] XGBoost failed with encoded labels: {e}")
            log_content.append(f"[ERROR] XGBoost failed: {str(e)}\n")
            print("[INFO] Retrying XGBoost without sample_weight...")
            gridcv = GridSearchCV(
                pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, verbose=2
            )
            gridcv.fit(X_train, y_train_encoded)
    else:
        gridcv.fit(X_train, y_train, 
                   **{'modelGeneration__sample_weight': sample_weight})

    gs_end_time = time.time()
    gs_duration = gs_end_time - gs_start_time
    
    best_model = gridcv.best_estimator_


    eval_start_time = time.time()
    

    print("[INFO] Evaluating on test set...")
    

    if model_type == "xgboost":
        y_pred_encoded = best_model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        y_test_eval = y_test_encoded
    else:
        y_pred = best_model.predict(X_test)
        y_test_eval = y_test
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, labels=classes, 
                                   output_dict=True, zero_division=0)

    try:
        y_proba = best_model.predict_proba(X_test)
        auc_macro_ovr = roc_auc_score(pd.get_dummies(y_test), y_proba, 
                                      multi_class="ovr", average="macro")
    except Exception as e:
        print(f"[WARNING] Could not compute AUC: {e}")
        auc_macro_ovr = None

    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time

    row = {
        "model": model_type,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_macro_ovr": auc_macro_ovr,
        "best_params": gridcv.best_params_
    }
    results_summary.append(row)

    print(f"\n[{model_type.upper()}] Results")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f} | Weighted-F1: {f1_weighted:.4f}")
    if auc_macro_ovr:
        print(f"AUC (OvR): {auc_macro_ovr:.4f}")

    log_content.append(f"\nBest Parameters: {gridcv.best_params_}")
    log_content.append(f"Accuracy: {acc:.4f}")
    log_content.append(f"Macro-F1: {f1_macro:.4f}")
    log_content.append(f"Weighted-F1: {f1_weighted:.4f}")
    if auc_macro_ovr:
        log_content.append(f"AUC (OvR): {auc_macro_ovr:.4f}")
    
    print("\n[Confusion Matrix]")
    display(pd.DataFrame(cm, index=classes, columns=classes))
    
    print("\n[Classification Report]")
    report_str = classification_report(y_test, y_pred, labels=classes, 
                                       zero_division=0)
    print(report_str)
    log_content.append(f"\nConfusion Matrix:\n{pd.DataFrame(cm, index=classes, columns=classes).to_string()}")
    log_content.append(f"\nClassification Report:\n{report_str}")

    # ⭐ 保存开始时间
    save_start_time = time.time()
    
    # 保存模型和分析
    model_path = OUTPUT_DIR / f"{NODE_NAME.replace(' ', '_')}_{model_type}_best.joblib"
    analysis_path = OUTPUT_DIR / f"{NODE_NAME.replace(' ', '_')}_{model_type}_analysis.joblib"
    
    joblib.dump(best_model, model_path)
    joblib.dump({
        "node": NODE_NAME,
        "model": model_type,
        "metrics": row,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "classes": classes
    }, analysis_path)

    save_end_time = time.time()
    save_duration = save_end_time - save_start_time

    analysis_pack_paths[model_type] = analysis_path
    print(f"[SAVE] Model -> {model_path}")
    print(f"[SAVE] Analysis -> {analysis_path}")
    
    log_content.append(f"\n[SAVE] Model -> {model_path}")
    log_content.append(f"[SAVE] Analysis -> {analysis_path}")
    log_content.append("\n" + "=" * 80 + "\n")

    model_end_time = time.time()
    total_duration = model_end_time - model_start_time

    timing_info = {
        "model": model_type,
        "gridcv_time": gs_duration,
        "evaluation_time": eval_duration,
        "saving_time": save_duration,
        "total_time": total_duration
    }
    timing_results.append(timing_info)
    
    # ⭐ 打印计时信息
    print(f"\n⏱️ [{model_type.upper()}] Timing:")
    print(f"   GridSearchCV:  {gs_duration:.2f}s")
    print(f"   Evaluation:    {eval_duration:.2f}s")
    print(f"   Saving:        {save_duration:.2f}s")
    print(f"   Total:         {total_duration:.2f}s")

print("\n[INFO] Training completed!")
log_content.append("\n[INFO] Training completed!")

training_end_time = time.time()
total_training_time = training_end_time - training_start_time

with open(log_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_content))
print(f"\n[LOG] Training log saved -> {log_file}")

# ⭐ 保存分析摘要
with open(analysis_summary_file, 'w', encoding='utf-8') as f:
    f.write(f"Training Summary - {TIMESTAMP}\n")
    f.write("=" * 80 + "\n\n")
    for row in results_summary:
        f.write(f"Model: {row['model']}\n")
        f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
        f.write(f"  Macro-F1: {row['f1_macro']:.4f}\n")
        f.write(f"  Weighted-F1: {row['f1_weighted']:.4f}\n")
        f.write(f"  AUC: {row['auc_macro_ovr']}\n")
        f.write(f"  Best Params: {row['best_params']}\n\n")
print(f"[SUMMARY] Analysis summary saved -> {analysis_summary_file}")

with open(timing_file, 'w', encoding='utf-8') as f:
    f.write(f"Training Timing Report - {TIMESTAMP}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n\n")
    f.write("Per-Model Timing:\n")
    f.write("-" * 80 + "\n\n")
    
    for timing in timing_results:
        f.write(f"Model: {timing['model']}\n")
        f.write(f"  GridSearchCV Time:  {timing['gridcv_time']:.2f}s\n")
        f.write(f"  Evaluation Time:    {timing['evaluation_time']:.2f}s\n")
        f.write(f"  Saving Time:        {timing['saving_time']:.2f}s\n")
        f.write(f"  Total Time:         {timing['total_time']:.2f}s\n\n")

    f.write("-" * 80 + "\n")
    f.write("Speed Ranking (fastest first):\n\n")
    sorted_timing = sorted(timing_results, key=lambda x: x['total_time'])
    for i, timing in enumerate(sorted_timing, 1):
        f.write(f"{i}. {timing['model']:12} {timing['total_time']:8.2f}s\n")

print(f"[TIMING] Timing report saved -> {timing_file}")

print("\n" + "=" * 80)
print("                    ⏱️ TIMING SUMMARY")
print("=" * 80)
print(f"\nTotal Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)\n")

for timing in timing_results:
    print(f"🔹 {timing['model'].upper()}")
    print(f"   GridSearchCV:  {timing['gridcv_time']:8.2f}s ({timing['gridcv_time']/timing['total_time']*100:5.1f}%)")
    print(f"   Evaluation:    {timing['evaluation_time']:8.2f}s ({timing['evaluation_time']/timing['total_time']*100:5.1f}%)")
    print(f"   Saving:        {timing['saving_time']:8.2f}s ({timing['saving_time']/timing['total_time']*100:5.1f}%)")
    print(f"   ────────────────────────────")
    print(f"   Total:         {timing['total_time']:8.2f}s ✅\n")

print("🏁 Speed Ranking:")
sorted_timing = sorted(timing_results, key=lambda x: x['total_time'])
for i, timing in enumerate(sorted_timing, 1):
    speedup = sorted_timing[-1]['total_time'] / timing['total_time']
    print(f"   {i}. {timing['model']:12} {timing['total_time']:8.2f}s (⚡ {speedup:.2f}x faster than slowest)")

print("\n" + "=" * 80)


results_df = pd.DataFrame(results_summary).sort_values(by="accuracy", ascending=False)
display(results_df)

csv_path = OUTPUT_DIR / "summary_metrics.csv"
results_df.to_csv(csv_path, index=False)
print(f"[EXPORT] Summary CSV -> {csv_path}")


