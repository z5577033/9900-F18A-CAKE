import argparse, json, time, joblib, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut
import matplotlib.pyplot as plt

def plot_confusion(cm, classes, out_png, title):
    fig = plt.figure(figsize=(max(4, len(classes)*0.5), max(4, len(classes)*0.5)))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

def stratified_holdout_ok(y):
    counts = Counter(y)
    return min(counts.values()) >= 2 and len(y) >= 10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="CSV with biosample_id + probe columns")
    ap.add_argument("--labels", required=True, help="CSV with biosample_id,label[,patient_id]")
    ap.add_argument("--outdir", default="/app/artifacts")
    ap.add_argument("--feature-sample", type=int, default=5000, help="random subset of features (0=all)")
    ap.add_argument("--n-est", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Load & merge
    Xdf = pd.read_csv(args.features)
    ydf = pd.read_csv(args.labels)
    if "biosample_id" not in Xdf.columns:
        raise ValueError("features CSV must contain 'biosample_id'")
    if "biosample_id" not in ydf.columns or "label" not in ydf.columns:
        raise ValueError("labels CSV must contain 'biosample_id' and 'label'")
    df = Xdf.merge(ydf, on="biosample_id", how="inner")
    assert len(df) > 1, "Need at least 2 samples after merge"

    # Labels
    y = df["label"].astype(str).values

    # Build feature column list:
    # - exclude obvious metadata
    # - keep only numeric dtypes
    EXCLUDE = {"biosample_id", "label", "patient_id"}
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after excluding metadata.")

    # Optional random feature subsample
    rng = np.random.default_rng(args.seed)
    if args.feature_sample and args.feature_sample > 0 and len(feat_cols) > args.feature_sample:
        feat_cols = list(rng.choice(feat_cols, size=args.feature_sample, replace=False))

    # Matrix + simple mean-impute
    X = df[feat_cols].to_numpy(dtype=float)
    col_means = np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=col_means)

    classes = sorted(list(set(y)))

    # Persist training columns/classes for inference alignment
    (out / "feature_list.txt").write_text("\n".join(feat_cols))
    (out / "classes.txt").write_text("\n".join(classes))

    results = {}
    preds_csv = out / "rf_preds.csv"

    # Path A: normal stratified holdout if viable
    if stratified_holdout_ok(y):
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr_idx, te_idx = next(sss1.split(X, y))
        X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        tr2_idx, va_idx = next(sss2.split(X_tr, y_tr))
        X_tr2, y_tr2 = X_tr[tr2_idx], y_tr[tr2_idx]
        X_va,  y_va  = X_tr[va_idx],  y_tr[va_idx]

        clf = RandomForestClassifier(
            n_estimators=args.n_est,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced_subsample"
        )
        t0 = time.time(); clf.fit(X_tr2, y_tr2); train_sec = time.time() - t0

        def eval_split(Xs, ys):
            yp = clf.predict(Xs)
            acc = accuracy_score(ys, yp)
            pr, rc, f1, _ = precision_recall_fscore_support(ys, yp, average="weighted", zero_division=0)
            rep = classification_report(ys, yp, output_dict=True, zero_division=0)
            cm  = confusion_matrix(ys, yp, labels=classes)
            return {"acc": acc, "precision_w": pr, "recall_w": rc, "f1_w": f1, "report": rep, "cm": cm.tolist()}

        results["strategy"] = "stratified_holdout"
        results["train"] = eval_split(X_tr2, y_tr2)
        results["val"]   = eval_split(X_va,  y_va)
        results["test"]  = eval_split(X_te,  y_te)
        results["params"] = {
            "n_estimators": args.n_est,
            "feature_sample": args.feature_sample,
            "seed": args.seed,
            "n_features_used": len(feat_cols),
            "train_seconds": round(train_sec, 3)
        }

        # Confusion matrix for test
        plot_confusion(np.array(results["test"]["cm"]), classes, out/"rf_holdout.cm_test.png",
                       "Confusion (holdout test)")

        # Fit final on all data
        clf_all = RandomForestClassifier(
            n_estimators=args.n_est,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced_subsample"
        )
        clf_all.fit(X, y)
        joblib.dump(clf_all, out / "rf_baseline.joblib")

    # Path B: LOOCV for tiny sets
    else:
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        per_fold = []

        for train_idx, test_idx in loo.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            clf = RandomForestClassifier(
                n_estimators=args.n_est,
                n_jobs=-1,
                random_state=args.seed,
                class_weight="balanced_subsample"
            )
            clf.fit(X_tr, y_tr)
            yp = clf.predict(X_te)
            y_true.extend(y_te.tolist()); y_pred.extend(yp.tolist())
            per_fold.append({
                "test_index": int(test_idx[0]),
                "true": y_te[0],
                "pred": yp[0]
            })

        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm  = confusion_matrix(y_true, y_pred, labels=classes)

        results["strategy"] = "leave_one_out_cv"
        results["loocv"] = {
            "acc": acc, "precision_w": pr, "recall_w": rc, "f1_w": f1,
            "report": rep, "cm": cm.tolist(), "n_folds": int(len(y))
        }
        pd.DataFrame(per_fold).to_csv(preds_csv, index=False)

        # Confusion matrix for LOOCV
        plot_confusion(cm, classes, out/"rf_loocv.cm.png", "Confusion (LOOCV)")

        # Final model on ALL data
        clf_all = RandomForestClassifier(
            n_estimators=args.n_est,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced_subsample"
        )
        clf_all.fit(X, y)
        joblib.dump(clf_all, out / "rf_baseline.joblib")

    # Save metrics + console summary
    (out / "rf_baseline.metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "strategy": results.get("strategy"),
        "summary": results.get("test", results.get("loocv", {})),
        "model": str(out/"rf_baseline.joblib"),
        "metrics": str(out/"rf_baseline.metrics.json"),
        "preds_csv": str(preds_csv) if preds_csv.exists() else None,
        "n_features_used": len(feat_cols)
    }, indent=2))

if __name__ == "__main__":
    main()
