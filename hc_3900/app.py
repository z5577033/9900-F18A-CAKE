from fastapi import FastAPI

app = FastAPI(title="mch minimal")

@app.get("/healthz")
def health():
    return {"ok": True}
