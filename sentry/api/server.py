"""SENTRY inference API.

Design choices:
  * FAIL-OPEN: this process never imports or calls into the detector. It
    logs a compact feature record per request and returns. If the detector
    worker (a separate OS process) is killed, /predict keeps serving;
    there is nothing in this file that depends on it being alive.
  * The only enforcement this process performs itself is a per-key
    throttle map, which the detector worker sets via /admin/throttle.
    That is what "automatic throttling of the suspicious key" means here.
  * Clean error handling: malformed base64 -> 400, oversized body -> 413,
    both with a plain JSON error body and no stack trace leaked to the
    client.
"""
import base64
import binascii
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from sentry.api.inference import predict
from sentry.api.keys import is_valid_key
from sentry.detect.projection import project

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LOG_PATH = RESULTS_DIR / "traffic_log.jsonl"
MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB, a single 32x32 image is tiny

app = FastAPI(title="SENTRY inference API")

_log_lock = threading.Lock()
_throttle = {}          # key -> {"until": epoch_seconds, "min_interval": float}
_throttle_lock = threading.Lock()
_last_request_time = {}  # key -> epoch_seconds, for throttle enforcement


def _log_record(rec: dict):
    with _log_lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _decode_image(image_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400,
                             detail=f"malformed base64: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400,
                             detail=f"could not decode image: {e}")
    img = img.resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                        # CHW
    return arr


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/admin/reset")
def admin_reset():
    """Clear throttle state and truncate the traffic log, used between
    demo runs so `make demo && make demo` is idempotent."""
    with _throttle_lock:
        _throttle.clear()
        _last_request_time.clear()
    with _log_lock:
        open(LOG_PATH, "w").close()
    return {"status": "reset"}


@app.post("/admin/throttle/{key}")
def admin_throttle(key: str, min_interval: float = 1.0, duration_s: float = 120.0):
    """Called by the detector worker (a separate process) when a key trips
    the fused signal. Never called by anything inside this file itself."""
    with _throttle_lock:
        _throttle[key] = {"until": time.time() + duration_s,
                           "min_interval": min_interval}
    return {"status": "throttled", "key": key, "min_interval": min_interval,
             "duration_s": duration_s}


@app.get("/admin/status")
def admin_status():
    with _throttle_lock:
        active = {k: v for k, v in _throttle.items() if v["until"] > time.time()}
    return {"throttled_keys": active, "time": time.time()}


@app.post("/predict")
async def predict_endpoint(request: Request, x_api_key: str = Header(None)):
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413,
                             detail=f"payload too large: {len(body)} bytes "
                                    f"> {MAX_BODY_BYTES} limit")

    if not x_api_key or not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="invalid or missing API key")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"malformed JSON body: {e}")

    if not isinstance(payload.get("image_b64"), str) or not payload["image_b64"]:
        raise HTTPException(status_code=400,
                             detail="'image_b64' must be a non-empty base64 string")

    # --- throttle enforcement (fail-DEGRADED, never fail-closed) ---------
    now = time.time()
    with _throttle_lock:
        t = _throttle.get(x_api_key)
        active = t is not None and t["until"] > now
        min_interval = t["min_interval"] if active else 0.0
    if active:
        last = _last_request_time.get(x_api_key, 0.0)
        if now - last < min_interval:
            with _throttle_lock:
                _last_request_time[x_api_key] = now
            raise HTTPException(status_code=429,
                                 detail="rate-limited: this key is under "
                                        "active extraction throttling")
    _last_request_time[x_api_key] = now

    arr = _decode_image(payload["image_b64"])
    result = predict(arr)

    feat = project(result["embedding"]).tolist()
    _log_record({
        "ts": now,
        "key": x_api_key,
        "class_idx": result["class_idx"],
        "confidence": result["confidence"],
        "feat": feat,
        "throttled": active,
    })

    return JSONResponse({
        "class_name": result["class_name"],
        "confidence": round(result["confidence"], 4),
    })


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
