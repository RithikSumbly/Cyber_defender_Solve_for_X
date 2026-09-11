"""Loads CIFAR-10 images as HWC uint8 arrays for traffic generators, honoring
the data ledger in results/data_ledger.json (built by model/train_victim.py).
"""
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import torchvision

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LEDGER_PATH = ROOT / "results" / "data_ledger.json"


@lru_cache(maxsize=1)
def _train_raw():
    ds = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    return ds.data, np.array(ds.targets)  # data: (N,32,32,3) uint8


@lru_cache(maxsize=1)
def _test_raw():
    ds = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
    return ds.data, np.array(ds.targets)


@lru_cache(maxsize=1)
def ledger():
    with open(LEDGER_PATH) as f:
        return json.load(f)


def pool_images(pool_name: str):
    """pool_name in {'B_cal', 'B_eval', 'A_holdout', 'train'} -> (images, labels)."""
    if pool_name == "train":
        return _train_raw()
    idx = ledger()[pool_name]
    data, targets = _test_raw()
    return data[idx], targets[idx]


def sample(pool_name: str, n: int, rng: np.random.RandomState, class_prior=None):
    """Sample n images (HWC uint8) from a pool, optionally matching a
    per-class sampling prior (list of 10 probabilities)."""
    images, labels = pool_images(pool_name)
    if class_prior is None:
        idx = rng.randint(0, len(images), size=n)
    else:
        prior = np.asarray(class_prior, dtype=np.float64)
        prior = prior / prior.sum()
        chosen_classes = rng.choice(10, size=n, p=prior)
        idx = np.empty(n, dtype=np.int64)
        by_class = {c: np.where(labels == c)[0] for c in range(10)}
        for i, c in enumerate(chosen_classes):
            idx[i] = rng.choice(by_class[c])
    return images[idx], labels[idx]


def random_noise_images(n: int, rng: np.random.RandomState):
    return rng.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


BENIGN_CLASS_PRIOR = [0.16, 0.06, 0.14, 0.14, 0.06, 0.14, 0.06, 0.06, 0.12, 0.06]
