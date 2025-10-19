#!/usr/bin/env python3
import os
import argparse

# limit threads for rpy2 and numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def main():
    ap = argparse.ArgumentParser(description="Run full training via BatchModelTrainer")
    ap.add_argument("--raise-on-error", action="store_true")
    ap.add_argument("--disable-dm", action="store_true")
    ap.add_argument("--rf-n-jobs", type=int, default=1)
    ap.add_argument("--cv-n-jobs", type=int, default=1)
    ap.add_argument("--only-node", type=str, default=None, help="only train specified node")
    ap.add_argument("--prefilter-topk", type=int, default=200)
    ap.add_argument("--prefilter-scan-max", type=int, default=5000)     # limit scanning to top N features
    ap.add_argument("--prefilter-chunk-size", type=int, default=1000)   # Max columns per chunk
    args = ap.parse_args()

    if args.disable_dm:
        os.environ["MCH_DISABLE_DM"] = "1"
    os.environ["RF_N_JOBS"] = str(args.rf_n_jobs)
    os.environ["CV_N_JOBS"] = str(args.cv_n_jobs)
    os.environ["MCH_PREFILTER_TOPK"] = str(args.prefilter_topk)
    if args.only_node:
        os.environ["MCH_ONLY_NODE"] = args.only_node
    os.environ["MCH_PREFILTER_SCAN_MAX"] = str(args.prefilter_scan_max)
    os.environ["MCH_PREFILTER_CHUNK_SIZE"] = str(args.prefilter_chunk_size)

    from mch.models.training import BatchModelTrainer
    trainer = BatchModelTrainer()
    stats = trainer.train_all_models(raise_on_error=args.raise_on_error)
    print(stats)

if __name__ == "__main__":
    main()