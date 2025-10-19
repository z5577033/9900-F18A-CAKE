#!/usr/bin/env python3
"""
读取 CSV 与 joblib，生成“对齐”的 CSV，保证 biosample_id 可与 joblib 中的样本集合匹配。
两种模式：
- filter：仅保留交集样本
- overwrite：将前 N 行的 biosample_id 覆盖为 joblib 的前 N 个样本ID（不丢行）

支持直接从 DiseaseTree 对象提取样本：
- --disease-node 可指定树中的某个节点名（缺省为整棵树）
- --sample-type 选择 all/training/validation（默认 all）

也支持 --deep-scan 深度扫描 joblib 的嵌套对象，提取形如 *_T_*_M 的样本ID。
"""
import argparse
from pathlib import Path
import sys
import json
import re
import joblib

# 识别项目里的 DiseaseTree
try:
    from mch.core.disease_tree import DiseaseTree
except Exception:
    DiseaseTree = None  # 允许在无该类时继续工作

# 优先用 polars；若无则退回 pandas
try:
    import polars as pl
except Exception:
    pl = None

try:
    import pandas as pd
except Exception:
    pd = None


ID_CANDIDATES = ["biosample_id", "sample_id", "id", "biosample", "sample", "case_id"]
ID_PATTERN = re.compile(r".*_T_.*_M$")  # 如 2J0D2U4J_T_XJAFP6HU_M


def is_candidate_id(x: str) -> bool:
    s = str(x).strip()
    if not s:
        return False
    return bool(ID_PATTERN.match(s))


def to_set_str(seq):
    return {str(x).strip() for x in seq if x is not None}


def extract_ids_from_dataframe(df, id_column=None):
    cols = df.columns if hasattr(df, "columns") else []
    col = id_column if (id_column and id_column in cols) else next((c for c in ID_CANDIDATES if c in cols), None)
    if not col:
        raise ValueError(f"未在 DataFrame 中找到 ID 列，请用 --joblib-id-column 指定。现有列: {list(cols)[:10]}")
    if pl and isinstance(df, pl.DataFrame):
        return to_set_str(df[col].to_list())
    if pd and isinstance(df, pd.DataFrame):
        return to_set_str(df[col].astype(str).str.strip().tolist())
    if hasattr(df, "to_pandas"):
        pdf = df.to_pandas()
        return to_set_str(pdf[col].astype(str).str.strip().tolist())
    raise TypeError(f"不支持的 DataFrame 类型: {type(df)}")


def deep_collect_ids(obj, id_column=None, max_depth=5):
    if max_depth < 0:
        return set()
    ids = set()
    if isinstance(obj, str):
        if is_candidate_id(obj):
            ids.add(obj.strip())
        return ids
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            ids |= deep_collect_ids(v, id_column, max_depth - 1)
        return ids
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            for v in obj.flatten().tolist():
                ids |= deep_collect_ids(v, id_column, max_depth - 1)
            return ids
    except Exception:
        pass
    if (pd and isinstance(obj, pd.DataFrame)) or (pl and isinstance(obj, pl.DataFrame)) or hasattr(obj, "to_pandas"):
        cols = obj.columns if hasattr(obj, "columns") else []
        cand = id_column if (id_column and id_column in cols) else next((c for c in ID_CANDIDATES if c in cols), None)
        if cand:
            return extract_ids_from_dataframe(obj, cand)
        if pl and isinstance(obj, pl.DataFrame):
            for c in cols:
                try:
                    vals = obj[c].to_list()
                    ids |= {v.strip() for v in vals if isinstance(v, str) and is_candidate_id(v)}
                except Exception:
                    continue
        else:
            pdf = obj if (pd and isinstance(obj, pd.DataFrame)) else obj.to_pandas()
            for c in pdf.columns:
                s = pdf[c]
                try:
                    vals = s.astype(str).tolist()
                    ids |= {v.strip() for v in vals if is_candidate_id(v)}
                except Exception:
                    continue
        return ids
    if isinstance(obj, dict):
        for _, v in obj.items():
            ids |= deep_collect_ids(v, id_column, max_depth - 1)
        return ids
    for attr in ("samples", "ids", "biosample_ids"):
        if hasattr(obj, attr):
            try:
                ids |= deep_collect_ids(getattr(obj, attr), id_column, max_depth - 1)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            for v in vars(obj).values():
                ids |= deep_collect_ids(v, id_column, max_depth - 1)
        except Exception:
            pass
    return ids


def extract_ids_from_disease_tree(tree, disease_node=None, sample_type="all"):
    if disease_node:
        node = tree.find_node_by_name(disease_node)
        if not node:
            return set()
        return to_set_str(node.get_samples_recursive(sample_type=sample_type))
    return to_set_str(tree.get_samples_recursive(sample_type=sample_type))


def extract_ids_from_joblib(obj, joblib_key=None, id_column=None, deep_scan=False,
                            disease_node=None, sample_type="all"):
    # DiseaseTree 直接按项目方法取
    if DiseaseTree and isinstance(obj, DiseaseTree):
        return extract_ids_from_disease_tree(obj, disease_node=disease_node, sample_type=sample_type)

    # 指定键优先
    if joblib_key and isinstance(obj, dict) and joblib_key in obj:
        target = obj[joblib_key]
        if DiseaseTree and isinstance(target, DiseaseTree):
            return extract_ids_from_disease_tree(target, disease_node=disease_node, sample_type=sample_type)
        obj = target

    # 尝试 DataFrame
    if (pd and isinstance(obj, pd.DataFrame)) or (pl and isinstance(obj, pl.DataFrame)) or hasattr(obj, "to_pandas"):
        try:
            return extract_ids_from_dataframe(obj, id_column=id_column)
        except Exception:
            pass

    # list/set/tuple
    if isinstance(obj, (list, set, tuple)):
        return to_set_str(obj)

    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return to_set_str(obj.flatten().tolist())
    except Exception:
        pass

    # dict：汇总尝试
    if isinstance(obj, dict):
        agg = set()
        for _, v in obj.items():
            try:
                agg |= extract_ids_from_joblib(v, None, id_column, deep_scan=False,
                                               disease_node=disease_node, sample_type=sample_type)
            except Exception:
                continue
        if agg:
            return agg

    # 常见属性
    for attr in ["samples", "ids", "biosample_ids"]:
        if hasattr(obj, attr):
            try:
                return to_set_str(getattr(obj, attr))
            except Exception:
                pass

    # 深搜兜底
    if deep_scan:
        return deep_collect_ids(obj, id_column=id_column, max_depth=6)

    return set()


def read_csv(path, id_column, lowercase=False):
    if pl:
        df = pl.read_csv(path)
        if id_column not in df.columns:
            alt = next((c for c in ID_CANDIDATES if c in df.columns), None)
            if alt and id_column is None:
                id_column = alt
            else:
                raise ValueError(f"CSV 中找不到列 {id_column}；现有列: {df.columns[:10]}")
        col_expr = pl.col(id_column).cast(pl.Utf8).str.strip_chars()
        if lowercase:
            col_expr = col_expr.str.to_lowercase()
        df = df.with_columns(col_expr.alias(id_column))
        return df
    if pd:
        df = pd.read_csv(path)
        if id_column not in df.columns:
            alt = next((c for c in ID_CANDIDATES if c in df.columns), None)
            if alt and id_column is None:
                id_column = alt
            else:
                raise ValueError(f"CSV 中找不到列 {id_column}；现有列: {list(df.columns)[:10]}")
        df[id_column] = df[id_column].astype(str).str.strip()
        if lowercase:
            df[id_column] = df[id_column].str.lower()
        return df
    raise RuntimeError("需要安装 polars 或 pandas 读取 CSV。请先: pip install polars 或 pip install pandas")


def write_csv(df, out_path):
    if pl and isinstance(df, pl.DataFrame):
        df.write_csv(out_path)
        return
    if pd and isinstance(df, pd.DataFrame):
        df.to_csv(out_path, index=False)
        return
    if hasattr(df, "to_pandas"):
        df.to_pandas().to_csv(out_path, index=False)
        return
    raise RuntimeError(f"无法保存 CSV，未知 DataFrame 类型: {type(df)}")


def main():
    ap = argparse.ArgumentParser(description="对齐 CSV 的 biosample_id 到 joblib 中的样本ID")
    ap.add_argument("--csv", required=True, help="CSV 文件路径")
    ap.add_argument("--joblib", required=True, help="joblib 文件路径")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    ap.add_argument("--mode", choices=["filter", "overwrite"], default="filter",
                    help="filter: 保留交集; overwrite: 覆盖前 N 行的 biosample_id（不丢行）")
    ap.add_argument("--id-column", default="biosample_id", help="CSV 的 ID 列名，默认 biosample_id")
    ap.add_argument("--joblib-key", default=None, help="若 joblib 为 dict，指定包含样本ID的键")
    ap.add_argument("--joblib-id-column", default=None, help="若 joblib 里是 DataFrame，指定其 ID 列名")
    ap.add_argument("--lowercase", action="store_true", help="统一转小写再对齐")

    # 与 DiseaseTree 对齐的参数
    ap.add_argument("--disease-node", default=None, help="若 joblib 是 DiseaseTree，指定节点名，仅提取该节点及子树样本")
    ap.add_argument("--sample-type", choices=["all", "training", "validation"], default="all",
                    help="DiseaseTree 样本类型，默认 all")

    ap.add_argument("--deep-scan", action="store_true", help="对 joblib 进行深度扫描提取 *_T_*_M 的样本ID")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    joblib_path = Path(args.joblib)
    out_path = Path(args.out)

    # 读 CSV
    df = read_csv(csv_path, args.id_column, lowercase=args.lowercase)

    # 读 joblib 并提取 ID 集（优先按 DiseaseTree 方法）
    obj = joblib.load(joblib_path)
    ids = extract_ids_from_joblib(
        obj,
        joblib_key=args.joblib_key,
        id_column=args.joblib_id_column,
        deep_scan=args.deep_scan,
        disease_node=args.disease_node,
        sample_type=args.sample_type,
    )
    if args.lowercase:
        ids = {s.lower() for s in ids}
    ids = {s.strip() for s in ids}

    if not ids:
        print("从 joblib 中未提取到任何样本 ID。可尝试：--disease-node / --sample-type 或 --deep-scan，"
              "或指定 --joblib-key / --joblib-id-column。", file=sys.stderr)
        sys.exit(2)

    # 执行两种模式
        # ...existing code...
        # 执行两种模式
    if pl and isinstance(df, pl.DataFrame):
        if args.mode == "filter":
            out_df = df.filter(pl.col(args.id_column).is_in(list(ids)))
        else:
            # 覆盖前 N 行，保持行数不变（避免向 DataFrame 传 Expr）
            take_n = min(df.height, len(ids))
            new_ids_full = list(ids)[:take_n] + [None] * (df.height - take_n)

            out_df = (
                df.with_columns(
                    # 新增一个辅助列：前 take_n 个为新 ID，其余为 None
                    pl.Series(name=f"{args.id_column}_new", values=new_ids_full).cast(pl.Utf8)
                )
                .with_columns(
                    # 用新列覆盖原 biosample_id（None 时保留原值）
                    pl.coalesce([pl.col(f"{args.id_column}_new"), pl.col(args.id_column)]).alias(args.id_column)
                )
                .drop(f"{args.id_column}_new")
            )
    else:
        if args.mode == "filter":
            out_df = df[df[args.id_column].isin(ids)].copy()
        else:
            take_n = min(len(df), len(ids))
            out_df = df.copy()
            if take_n > 0:
                out_df.loc[:take_n-1, args.id_column] = list(ids)[:take_n]
    # ...existing code...

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_df, out_path)

    # 摘要
    orig_n = df.height if (pl and isinstance(df, pl.DataFrame)) else len(df)
    out_n = out_df.height if (pl and isinstance(out_df, pl.DataFrame)) else len(out_df)
    print(f"完成: 原始行数={orig_n}, 输出行数={out_n}, 模式={args.mode}")

    # 交集校验
    try:
        csv_ids = set(out_df[args.id_column].to_list()) if (pl and isinstance(out_df, pl.DataFrame)) else set(out_df[args.id_column].astype(str).tolist())
        inter = csv_ids & ids
        print(f"校验: 交集数量={len(inter)} 示例={list(inter)[:5]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()