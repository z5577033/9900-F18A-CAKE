#!/usr/bin/env bash
set -euo pipefail

SERVICE="mch"                           # docker-compose service name
HOST_PORT="8000"                        # exposed port in compose
MODEL_OUTDIR="/app/data/freeze0525"     # where trainer writes model (matches API expectations)
PLOTS_OUTDIR="$MODEL_OUTDIR/plots"      # where plot PNGs go

FEATURES_IN_CONTAINER="/app/datasets/noisy_mvalues.csv"
LABELS_IN_CONTAINER="/app/datasets/noisy_labels.csv"

if ! docker compose ps >/dev/null 2>&1; then
  echo "ERROR: Not in a folder with docker-compose.yml (or docker compose not installed)." >&2
  exit 1
fi

# If noisy set doesn't exist in the container, fallback to synthetic
fallback_to_synth=true

function wait_for_api() {
  local url=$1
  local tries=30
  echo "Waiting for API at ${url} ..."
  for i in $(seq 1 $tries); do
    if curl -fsS "$url" >/dev/null; then
      echo "API is up."
      return 0
    fi
    sleep 1
  done
  echo "ERROR: API didn't respond at ${url}" >&2
  return 1
}

function compose_exec() {
  docker compose exec -T "$SERVICE" sh -lc "$*"
}

# ---------------------------
# 1) Build & start service
# ---------------------------
echo "Building image and starting service..."
docker compose up --build -d

echo "Current services:"
docker compose ps

# ---------------------------
# 2) Confirm volumes & API module path
# ---------------------------
echo "Checking container environment..."
compose_exec 'echo "PYTHONPATH=$PYTHONPATH"; ls -lah /app/src/mch/api || true'

# If our preferred dataset isn't there, fallback to synthetic
if ! compose_exec "[ -f ${FEATURES_IN_CONTAINER} ]"; then
  if [ "${fallback_to_synth}" = "true" ]; then
    echo "No noisy dataset found at ${FEATURES_IN_CONTAINER}. Falling back to synthetic."
    FEATURES_IN_CONTAINER="/app/datasets/synthetic_mvalues.csv"
    LABELS_IN_CONTAINER="/app/datasets/labels_zero2.csv"
  fi
fi

# ---------------------------
# 3) Wait for API to be reachable
# ---------------------------
wait_for_api "http://localhost:${HOST_PORT}/healthz" || true
echo "Healthz says:"
curl -fsS "http://localhost:${HOST_PORT}/healthz" || true
echo

# ---------------------------
# 4) Train model (writes to ${MODEL_OUTDIR})
# ---------------------------
echo "Training model..."
compose_exec "python /app/scripts/baseline_train.py \
  --features ${FEATURES_IN_CONTAINER} \
  --labels   ${LABELS_IN_CONTAINER} \
  --outdir   ${MODEL_OUTDIR} \
  --n-est 800 --feature-sample 3000 --seed 1337"

echo "Artifacts in ${MODEL_OUTDIR}:"
compose_exec "ls -lah ${MODEL_OUTDIR}"

# ---------------------------
# 5) Generate visuals from metrics JSON
# ---------------------------
# Ensure plot script exists
if ! compose_exec "[ -f /app/scripts/plot_metrics.py ]"; then
  echo "plot_metrics.py not found in container. Creating a temporary one..."
  compose_exec 'cat > /app/scripts/plot_metrics.py << "PY"
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def bar_save(xnames, vals, title, outpath, ylim=(0,1.0), rotation=45):
    fig = plt.figure(figsize=(max(6, len(xnames)*0.6), 4))
    xs = np.arange(len(xnames))
    plt.bar(xs, vals)
    plt.title(title)
    plt.xticks(xs, xnames, rotation=rotation)
    if ylim is not None: plt.ylim(*ylim)
    plt.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--split",    default="test", choices=["train","val","test","loocv"])
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    data = json.loads(Path(args.metrics).read_text())
    strat = data.get("strategy")

    if strat == "stratified_holdout":
        block = data.get(args.split)
        if block is None: raise SystemExit(f"Split {args.split} not found.")
    elif strat == "leave_one_out_cv":
        block = data.get("loocv")
    else:
        block = data.get(args.split) or data.get("test") or data.get("loocv")
        if block is None: raise SystemExit("No usable metrics block found.")

    acc  = block.get("acc")
    prw  = block.get("precision_w")
    rcw  = block.get("recall_w")
    f1w  = block.get("f1_w")
    names, vals = [], []
    if acc is not None: names.append("Accuracy");     vals.append(acc)
    if prw is not None: names.append("Precision_w");  vals.append(prw)
    if rcw is not None: names.append("Recall_w");     vals.append(rcw)
    if f1w is not None: names.append("F1_w");         vals.append(f1w)
    if names:
        bar_save(names, vals, f"Overall Metrics ({args.split})", outdir/f"overall_metrics_{args.split}.png", rotation=0)

    rep = block.get("report", {})
    agg = {"accuracy","macro avg","weighted avg","micro avg"}
    class_keys = [k for k in rep.keys() if k not in agg]

    if class_keys:
        precs = [rep[k].get("precision",0.0) for k in class_keys]
        f1s   = [rep[k].get("f1-score",0.0) for k in class_keys]
        bar_save(class_keys, precs, f"Per-class Precision ({args.split})", outdir/f"per_class_precision_{args.split}.png")
        bar_save(class_keys, f1s,   f"Per-class F1 ({args.split})",        outdir/f"per_class_f1_{args.split}.png")

    cm = block.get("cm")
    if cm is not None:
        cm = np.array(cm)
        fig = plt.figure(figsize=(max(6, cm.shape[1]*0.6), max(6, cm.shape[0]*0.6)))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix ({args.split})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        if class_keys:
            plt.xticks(range(len(class_keys)), class_keys, rotation=90)
            plt.yticks(range(len(class_keys)), class_keys)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.tight_layout(); fig.savefig(outdir/f"confusion_matrix_{args.split}.png", dpi=180); plt.close(fig)

    print(json.dumps({"strategy": strat, "split": args.split, "outdir": str(outdir)}, indent=2))

if __name__ == "__main__":
    main()
PY'
fi

echo "Plotting metrics to ${PLOTS_OUTDIR} ..."
compose_exec "python /app/scripts/plot_metrics.py \
  --metrics ${MODEL_OUTDIR}/rf_baseline.metrics.json \
  --outdir  ${PLOTS_OUTDIR} \
  --split   test"

echo
echo "Done!"
echo "Model dir (host): ./data/freeze0525"
echo "Plots (host):      ./data/freeze0525/plots"
echo
echo "Health check:"
curl -fsS "http://localhost:${HOST_PORT}/healthz" || true
echo
