"""Fixed random projection: 256-d victim embedding -> 16-d logged vector.

Used by the API (to log a compact, lossy feature vector instead
of raw pixels) and by the detector / eval harness (to compute diversity,
distance and coverage signals). The matrix is generated once and cached to
disk so every process (API, detector worker, eval script) agrees on the
same projection for the lifetime of the demo.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ_PATH = ROOT / "results" / "projection.npy"

DIM_IN = 256
DIM_OUT = 16
SEED = 42


def get_projection() -> np.ndarray:
    if PROJ_PATH.exists():
        return np.load(PROJ_PATH)
    rng = np.random.RandomState(SEED)
    mat = rng.normal(size=(DIM_IN, DIM_OUT)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=0, keepdims=True)
    PROJ_PATH.parent.mkdir(exist_ok=True)
    np.save(PROJ_PATH, mat)
    return mat


def project(embedding: np.ndarray) -> np.ndarray:
    mat = get_projection()
    return embedding @ mat
