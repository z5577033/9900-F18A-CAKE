from fastapi import FastAPI
from pathlib import Path

app = FastAPI(title="mch minimal")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/freeze")
def freeze_info():
    root = Path(__file__).resolve().parents[2]  # repo root
    joblib_file = root / "data" / "freeze0525" / "diseaseTree_mapped.joblib"
    return {
        "freeze_dir_exists": (root / "data" / "freeze0525").exists(),
        "joblib_exists": joblib_file.exists(),
        "joblib_path": str(joblib_file),
    }
