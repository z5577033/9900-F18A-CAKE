from fastapi import FastAPI, HTTPException, Query
from typing import List
import numpy as np

from mch.config.settings import MODEL, read_one_sample, vectorize_for_model

app = FastAPI(title="mch inference")

@app.get("/healthz")
def health():
    return {
        "ok": MODEL is not None,
        "model_loaded": MODEL is not None,
        "classes": (MODEL.classes_.tolist() if MODEL is not None else None)
    }

@app.get("/predict")
def predict(sample_id: str = Query(..., description="biosample_id or sample id in your features table"),
           topk: int = 5):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check data/freeze0525/rf_baseline.joblib")
    try:
        row = read_one_sample(sample_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    X, cols = vectorize_for_model(row)

    # predict
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        classes = MODEL.classes_.tolist()
        order = np.argsort(proba)[::-1]
        k = min(topk, len(order))
        top = [{"label": classes[i], "prob": float(proba[i])} for i in order[:k]]
        pred = classes[int(order[0])]
        return {"sample_id": sample_id, "predicted": pred, "top": top, "n_features": len(cols)}
    else:
        pred = MODEL.predict(X)[0]
        return {"sample_id": sample_id, "predicted": str(pred), "top": None, "n_features": len(cols)}
