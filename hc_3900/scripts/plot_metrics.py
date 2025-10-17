import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def bar_save(xnames, vals, title, outpath, ylim=(0,1.0), rotation=45, width=0.8):
    fig = plt.figure(figsize=(max(6, len(xnames)*0.6), 4))
    xs = np.arange(len(xnames))
    plt.bar(xs, vals, width=width)
    plt.title(title)
    plt.xticks(xs, xnames, rotation=rotation)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to rf_baseline.metrics.json")
    ap.add_argument("--outdir",   required=True, help="Where to write PNGs")
    ap.add_argument("--split",    default="test", choices=["train","val","test","loocv"],
                    help="Which split to visualize (use 'loocv' if strategy is leave_one_out_cv)")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    data = json.loads(Path(args.metrics).read_text())

    # Detect strategy/split
    strat = data.get("strategy")
    if strat == "stratified_holdout":
        block = data.get(args.split)
        if block is None:
            raise SystemExit(f"Split '{args.split}' not found in metrics (available: {list(k for k in data.keys() if isinstance(data[k], dict))})")
    elif strat == "leave_one_out_cv":
        if args.split != "loocv":
            print("Strategy is LOOCV; using 'loocv' block regardless of --split.")
        block = data.get("loocv")
    else:
        # Try best effort
        block = data.get(args.split) or data.get("test") or data.get("loocv")
        if block is None:
            raise SystemExit("Could not find any usable split metrics in JSON.")

    # Overall metrics
    acc = block.get("acc", None)
    prw = block.get("precision_w", None)
    rcw = block.get("recall_w", None)
    f1w = block.get("f1_w", None)

    overall_names, overall_vals = [], []
    if acc is not None: overall_names.append("Accuracy"); overall_vals.append(acc)
    if prw is not None: overall_names.append("Precision_w"); overall_vals.append(prw)
    if rcw is not None: overall_names.append("Recall_w"); overall_vals.append(rcw)
    if f1w is not None: overall_names.append("F1_w"); overall_vals.append(f1w)

    if overall_names:
        bar_save(overall_names, overall_vals,
                 title=f"Overall Metrics ({args.split})",
                 outpath=outdir / f"overall_metrics_{args.split}.png",
                 rotation=0)

    # Per-class metrics
    rep = block.get("report", {})
    # filter out aggregate keys
    agg_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    class_keys = [k for k in rep.keys() if k not in agg_keys]

    if class_keys:
        # Precision
        precs = [rep[k].get("precision", 0.0) for k in class_keys]
        bar_save(class_keys, precs,
                 title=f"Per-class Precision ({args.split})",
                 outpath=outdir / f"per_class_precision_{args.split}.png")

        # F1
        f1s = [rep[k].get("f1-score", 0.0) for k in class_keys]
        bar_save(class_keys, f1s,
                 title=f"Per-class F1 ({args.split})",
                 outpath=outdir / f"per_class_f1_{args.split}.png")

    # Confusion matrix if present
    cm = block.get("cm", None)
    if cm is not None:
        cm = np.array(cm)
        fig = plt.figure(figsize=(max(6, cm.shape[1]*0.6), max(6, cm.shape[0]*0.6)))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix ({args.split})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        # Use class order from report if available
        if class_keys:
            plt.xticks(range(len(class_keys)), class_keys, rotation=90)
            plt.yticks(range(len(class_keys)), class_keys)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.tight_layout(); fig.savefig(outdir / f"confusion_matrix_{args.split}.png", dpi=180); plt.close(fig)

    print(json.dumps({
        "strategy": strat,
        "split": args.split,
        "wrote": [p.name for p in outdir.iterdir() if p.suffix.lower()==".png"]
    }, indent=2))

if __name__ == "__main__":
    main()
