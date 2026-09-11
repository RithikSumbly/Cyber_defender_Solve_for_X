"""HTTP client helpers shared by all traffic generators."""
import base64
import io
import time

import numpy as np
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"


def encode_image(img_hwc_uint8: np.ndarray) -> str:
    img = Image.fromarray(img_hwc_uint8.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def send_query(key: str, img_hwc_uint8: np.ndarray, timeout=5.0):
    b64 = encode_image(img_hwc_uint8)
    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": key},
            json={"image_b64": b64},
            timeout=timeout,
        )
        latency = time.time() - t0
        return {
            "ok": resp.status_code == 200,
            "status_code": resp.status_code,
            "latency": latency,
            "body": resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json") else None,
            "ts": t0,
        }
    except requests.RequestException as e:
        return {"ok": False, "status_code": None, "latency": time.time() - t0,
                "error": str(e), "ts": t0}


def reset_server():
    requests.post(f"{BASE_URL}/admin/reset", timeout=5.0)


def server_status():
    return requests.get(f"{BASE_URL}/admin/status", timeout=5.0).json()


def is_alive() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2.0)
        return r.status_code == 200
    except requests.RequestException:
        return False
