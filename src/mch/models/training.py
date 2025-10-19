import os, time, math, logging, heapq
from pathlib import Path
from typing import Dict, Optional
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from differentialMethylationClassifier import DifferentialMethylation
from mch.config.settings import mvalue_df, main_tree, DATA_DIR
from mch.config.modelTrainingParameters import parameter_grid, resultsDirectory

logger = logging.getLogger("mch.training")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(DATA_DIR / "logs" / "training.log"), encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)

def _prefilter_polars_chunked(df_pl: pl.DataFrame, topk: int, id_col: str = "biosample_id"):
    """按列方差挑前 topk 个特征，分块+限额扫描，降低内存峰值。返回(精简后的 df, 被保留列名)。"""
    scan_max = int(os.getenv("MCH_PREFILTER_SCAN_MAX", "20000"))   # 最多扫描多少列
    chunk_size = int(os.getenv("MCH_PREFILTER_CHUNK_SIZE", "5000"))  # 每块多少列
    feat_cols = [c for c in df_pl.columns if c != id_col]
    logger.info(
        "Prefilter start: input shape=(%d,%d), candidates=%d, topk=%d, scan_max=%d, chunk_size=%d",
        df_pl.height, df_pl.width, len(feat_cols), topk, scan_max, chunk_size
    )
    if topk <= 0 or len(feat_cols) <= topk:
        kept = feat_cols if topk <= 0 else feat_cols[:topk]
        logger.info("Prefilter bypass: kept=%d", len(kept))
        return df_pl.select([id_col, *kept]), kept

    if scan_max > 0:
        feat_cols = feat_cols[:min(scan_max, len(feat_cols))]

    heap: list[tuple[float, str]] = []  # (var, col) 的最小堆
    scanned = 0
    for i in range(0, len(feat_cols), chunk_size):
        cols = feat_cols[i:i + chunk_size]
        # 对该块的列做类型转换并计算方差
        var_row = df_pl.select([pl.col(c).cast(pl.Float64, strict=False).var().alias(c) for c in cols]).to_dicts()[0]
        for c, v in var_row.items():
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            if len(heap) < topk:
                heapq.heappush(heap, (v, c))
            elif v > heap[0][0]:
                heapq.heapreplace(heap, (v, c))
        scanned = i + len(cols)
        logger.info("Prefilter progress: scanned %d/%d columns", scanned, len(feat_cols))

    kept_cols = [c for _, c in sorted(heap, key=lambda x: x[0], reverse=True)]
    df_small = df_pl.select([id_col, *kept_cols])
    logger.info("Prefilter done: kept %d features", len(kept_cols))
    return df_small, kept_cols

class BatchModelTrainer:
    """Handles training of multiple models across the disease tree."""

    def __init__(self, tree=main_tree):
        self.tree = tree
        self.models: Dict[str, RandomForestClassifier] = {}
        self.training_stats: Dict[str, Dict] = {}
        self.dataDirectory = DATA_DIR
        self.resultsDirectory = resultsDirectory
        self.filteredMValueFile = mvalue_df
        # 配置用环境变量（由 run_training.py 设置）
        self.disable_dm = (os.getenv("MCH_DISABLE_DM", "0") == "1")
        self.rf_n_jobs = int(os.getenv("RF_N_JOBS", "1"))
        self.cv_n_jobs = int(os.getenv("CV_N_JOBS", "1"))
        self.prefilter_topk = int(os.getenv("MCH_PREFILTER_TOPK", "200"))
        self.only_node = os.getenv("MCH_ONLY_NODE")
        self.rf_params = {
            "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "50")),
            "max_depth": int(os.getenv("RF_MAX_DEPTH", "10")),
        }

    def train_all_models(self, save_dir: Optional[Path] = None, raise_on_error: bool = False) -> Dict:
        """
        Trains models for all nodes in the disease tree.
        """
        nodes = self.tree.get_child_names()
        if self.only_node:
            nodes = [n for n in nodes if n == self.only_node]
            logger.info("Only training specified node: %s", self.only_node)

        for node in nodes:
            try:
                logger.info("Training model for node: %s", node)
                result = self._prepare_node_data(node)
                if result is None:
                    continue
                nodeData, design = result  # nodeData: pl.DataFrame, design: pl.DataFrame

                # 仅在禁用 DM 时做预筛选；启用 DM 则不做预筛选（避免双重特征选择）
                effective_topk = self.prefilter_topk if self.disable_dm else 0
                nodeData_pf, kept_cols = _prefilter_polars_chunked(nodeData, topk=effective_topk, id_col="biosample_id")
                # 只转保留下来的特征列为 pandas float32
                X_all = nodeData_pf.select(kept_cols).to_pandas().astype("float32")
                y_all = design["cancerType"].to_pandas()

                # 丢弃样本数 < 2 的类别，避免 stratify 报错
                cls_counts = y_all.value_counts()
                rare = cls_counts[cls_counts < 2].index.tolist()
                if rare:
                    shown = ", ".join(map(str, rare[:5])) + ("..." if len(rare) > 5 else "")
                    logger.warning("Drop rare classes (<2 samples): %s", shown)
                    mask = ~y_all.isin(rare)
                    X_all = X_all[mask]
                    y_all = y_all[mask]
                # 丢完后仍需至少两个类别
                if y_all.nunique() < 2:
                    logger.warning("Skip %s: fewer than 2 classes after dropping rare classes", node)
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
                )
                logger.info("Train/Test shapes: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

                # 构建 pipeline（真正关闭 DM：不加入 DM）
                rf = RandomForestClassifier(
                    random_state=42,
                    n_jobs=self.rf_n_jobs,
                    n_estimators=self.rf_params["n_estimators"],
                    max_depth=self.rf_params["max_depth"],
                )
                if self.disable_dm:
                    logger.info("DM disabled")
                    pipeline = Pipeline([("modelGeneration", rf)])
                else:
                    pipeline = Pipeline([
                        ("differentialMethylation", DifferentialMethylation()),
                        ("modelGeneration", rf),
                    ])
                logger.info("Pipeline steps: %s", list(pipeline.named_steps.keys()))

                # 动态设置 CV 折数，避免最小类别样本数不足导致报错
                if not parameter_grid:
                    stratified_cv = None
                else:
                    min_cls_train = y_train.value_counts().min()
                    n_splits = max(2, min(3, int(min_cls_train))) if min_cls_train >= 2 else 2
                    stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                if not parameter_grid:
                    logger.warning("parameter_grid is None/empty, fit pipeline without grid search")
                    pipeline.fit(X_train, y_train)
                    self.models[node] = pipeline

                    # 评估与统计
                    metrics = {}
                    if X_test is not None:
                        y_pred = pipeline.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        metrics = {
                            "accuracy": float(acc),
                            "macro_f1": float(report.get("macro avg", {}).get("f1-score", float("nan"))),
                            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", float("nan"))),
                        }
                        logger.info("Test accuracy: %.4f, macro_f1: %.4f, weighted_f1: %.4f",
                                    metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
                    # 特征重要度（前10）
                    top_features = None
                    try:
                        import numpy as np
                        importances = pipeline.named_steps["modelGeneration"].feature_importances_
                        idx = np.argsort(importances)[::-1][:10]
                        top_features = [{"feature": kept_cols[i], "importance": float(importances[i])} for i in idx]
                    except Exception:
                        top_features = None

                    self.training_stats[node] = {
                        "n_samples_total": int(len(y_all)),
                        "n_samples_train": int(len(y_train)),
                        "n_samples_test": int(0 if X_test is None else len(y_test)),
                        "n_features": int(len(kept_cols)),
                        "classes": sorted(map(str, y_all.unique())),
                        "metrics": metrics,
                        "top_features": top_features,
                    }
                else:
                    search = GridSearchCV(
                        pipeline,
                        param_grid=parameter_grid,
                        scoring='accuracy',
                        cv=stratified_cv,
                        verbose=2,
                        n_jobs=self.cv_n_jobs,
                        error_score='raise',
                    )
                    search.fit(X_train, y_train)
                    self.models[node] = search.best_estimator_

                    # 评估与统计（GridSearch 后）
                    metrics = {}
                    if X_test is not None:
                        y_pred = self.models[node].predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        metrics = {
                            "accuracy": float(acc),
                            "macro_f1": float(report.get("macro avg", {}).get("f1-score", float("nan"))),
                            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", float("nan"))),
                        }
                        logger.info("[GS] Test accuracy: %.4f, macro_f1: %.4f, weighted_f1: %.4f",
                                    metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
                    top_features = None
                    try:
                        import numpy as np
                        importances = self.models[node].named_steps["modelGeneration"].feature_importances_
                        idx = np.argsort(importances)[::-1][:10]
                        top_features = [{"feature": kept_cols[i], "importance": float(importances[i])} for i in idx]
                    except Exception:
                        top_features = None
                    self.training_stats[node] = {
                        "n_samples_total": int(len(y_all)),
                        "n_samples_train": int(len(y_train)),
                        "n_samples_test": int(0 if X_test is None else len(y_test)),
                        "n_features": int(len(kept_cols)),
                        "classes": sorted(map(str, y_all.unique())),
                        "metrics": metrics,
                        "top_features": top_features,
                        "best_params": getattr(search, "best_params_", None),
                    }

                if save_dir:
                    logger.info("Model trained for %s, save_dir=%s", node, save_dir)
                else:
                    logger.info("Model trained for %s, not saved", node)

            except Exception:
                logger.exception("Error training model for %s", node)
                if raise_on_error:
                    raise
        return self.training_stats

    def _prepare_node_data(self, node: str):
        """Prepare training data for a specific node."""
        #print(f"Constructing dataset for {node}")
        # truthValues = pd.Series(["otherCancerType"] * len(mvalue_df["biosample_id"]))
        # design = pd.DataFrame({"biosample_id": mvalue_df["biosample_id"], "cancerType": truthValues.values})

        print(f"{node} origin data count:", mvalue_df.height)
        print(f"{node} mvalue_df columns:", mvalue_df.columns[:5])

        truthValues = ["otherCancerType"] * mvalue_df.height
        design = pl.DataFrame({
            "biosample_id": mvalue_df["biosample_id"],
            "cancerType": truthValues
        })


        # find the node in the main tree corresponding to this particular cancer type.
        diseaseTree = main_tree.find_node_by_name(node)
        if diseaseTree is None:
            logger.warning("%s: node not found in disease tree, skip", node)
            return None

        diseaseSamples = diseaseTree.get_samples_recursive()
        print(f"{node} diseaseTree count:", len(diseaseSamples))

        # get the names of the children of that cancer type and the samples associated with each of them
        for cancer in diseaseTree.get_child_names():
            cancerTree = diseaseTree.find_node_by_name(cancer)
            if cancerTree is None:
                logger.warning("%s: child not found: %s, skip", node, cancer)
                continue
            try:
                samples = cancerTree.get_samples_recursive()
            except Exception as ce:
                logger.warning("%s: get_samples_recursive failed for child %s: %s", node, cancer, ce)
                continue
            #print(len(samples), cancer)
            # if the child node has at least 3 samples (this should probaby be 4?), then change the cancertype in the 
            # design dataframe to be the child node name. Otherwise it get's left as "otherCancerType"
            # this means that the samples of the child node remain in the training data, but it will never be used 
            # as a comparator in the differential methylation step. 
            if len(samples) >= 3:
                # design.loc[design['biosample_id'].isin(samples), 'cancerType'] = cancer
                mask = design["biosample_id"].is_in(samples)
                design = design.with_columns(
                    pl.when(mask)
                      .then(pl.lit(cancer))
                      .otherwise(pl.col("cancerType"))
                      .alias("cancerType")
                )

        # this records the samples that we have data for, but are missing from the oncotree. I don't think we need this. 
        # it will only record those few samples that have been missed on the latest update of the oncotree
        # with open("/media/storage/bcurran/classifiers/methylation/data/samplesMissingFromOncotree.csv", "wt") as file:
        #    filteredData[design.cancerType == "otherCancerType"].to_csv(file)

        # remove samples that don't have a cancer type - this prepares the dataset for differential methylation. 
        # actually. Hold on. I want the ones with "otherCancerType", but not the ones that are not in the sample list. 
        # this should be filtering on the sample_id. 
        # filteredData = filteredData[design.cancerType != "otherCancerType"]
        # design = design[design.cancerType != "otherCancerType"]

        biosample_ids = set(mvalue_df["biosample_id"].to_list())
        diseaseSamples_set = set(diseaseSamples)
        intersection = biosample_ids & diseaseSamples_set
        print(f"{node} joint count: {len(intersection)}")
        print(f"{node} joint samples: {list(intersection)[:5]}")

        # filteredData = mvalue_df[mvalue_df["biosample_id"].isin(diseaseSamples)]
        filteredData = mvalue_df.filter(pl.col("biosample_id").is_in(diseaseSamples))
        # design = design[design["biosample_id"].isin(filteredData["biosample_id"])]
        design = design.filter(pl.col("biosample_id").is_in(filteredData["biosample_id"].to_list()))

        print(f"{node} Data after filter:", filteredData.height)
        print(f"{node} Data after design filter:", design.height)

        print(node, filteredData.height, design.height)

        # an entire category with only ten samples isn't sufficiently informative.
        # this should filter any small cancer types and cancers/oncotree nodes with no children types.
        if filteredData["biosample_id"].len() < 10:
            print(f"Skipping, {node} has fewer than 10 samples")
            return None

        
        # filteredDataset = filteredData.dropna(axis="columns")
        filteredDataset = filteredData.drop_nulls()

        # there was another filter here ...
        # making sure there were at least nfolds number of samples for each child category?
        # valueCounts = design.cancerType.value_counts()
        valueCounts = design["cancerType"].value_counts()

        # make sure there are at least two groups being passed back
        if len(design["cancerType"].unique()) <2:
            print(f"Skipping, there is only one subgroup of {diseaseTree.name}")
            return None

        print(f"design type：", type(design))
        return filteredData, design
            # Implementation from your existing code
    
    def _save_model(self, node: str, model: RandomForestClassifier, save_dir: Path):
        """Save a single model to disk."""
        model_path = save_dir / f"{node}_model.joblib"
        joblib.dump(model, model_path)
        
    def get_training_summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing all model training results."""
        return pd.DataFrame.from_dict(self.training_stats, orient='index')