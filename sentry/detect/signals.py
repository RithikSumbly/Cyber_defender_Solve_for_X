"""Raw signal computation over a per-key window of traffic-log records.

Each record: {"ts": float, "key": str, "class_idx": int, "confidence": float,
              "feat": [16 floats], "throttled": bool}

Every function here takes RAW records only, no ground-truth label ever
enters this file, which keeps the reported TPR/FPR numbers free of label
leakage.
"""
import math
import numpy as np

LOW_CONF_THRESHOLD = 0.55

# Coarse binning grid for callers that threshold the RAW coverage_growth
# value directly instead of z-scoring it (see coverage_growth below).
FLEET_BINS = 2
FLEET_DIMS = 6  # 2^6 = 64 total bins, coarse enough to actually saturate

# Minimum per-key window samples before a key counts toward the fleet
# specialization-gap pooling (calibrate.py / simulate.py / detector.py all
# import this, so calibration and detection apply the identical gate and
# the fitted threshold carries over exactly). Entropy
# computed from 2-3 samples is dominated by sampling noise, not real signal:
# a key that just started its episode can look artificially "narrow" purely
# from having almost no data yet. Attack keys in this system's traffic model
# run well above this floor (35+ samples per 5s window), so the gate costs
# no attack-detection sensitivity while filtering out cold-start noise.
FLEET_MIN_SAMPLES = 5

# Both cross-key legs (campaign correlation, fleet specialization-gap) are
# percentile thresholds, re-checked every STEP_SECONDS while 2+ keys stay
# concurrently active. A long episode is many correlated checks, not one.
# Requiring the SAME key-group to cross threshold on this many CONSECUTIVE
# 1-second polls before firing turns a single crossing into a sustained
# multi-second signal, a much rarer coincidence given how heavily
# overlapping windows correlate (adjacent 1-second-apart windows share ~80%
# of their underlying 5-second content, so genuine signal moves smoothly).
# This persistence gate is how the cross-key legs handle continuous
# re-checking; the per-key leg handles it with its sequential e-value test
# (sentry/detect/evalue.py).
CROSS_KEY_PERSISTENCE_POLLS = 3


def rate_qps(window: list, span_s: float) -> float:
    if span_s <= 0:
        return 0.0
    return len(window) / span_s


def class_entropy(window: list) -> float:
    """Normalized Shannon entropy of predicted-class distribution, in [0,1].
    High = querying broadly across classes (boundary mapping)."""
    if not window:
        return 0.0
    counts = np.zeros(10)
    for r in window:
        counts[r["class_idx"]] += 1
    p = counts / counts.sum()
    p = p[p > 0]
    ent = -(p * np.log(p)).sum()
    return float(ent / math.log(10))


def mean_consecutive_distance(window: list) -> float:
    """Average L2 distance between consecutive queries' feature vectors.
    LOW values mean near-duplicate / perturbation-sweep querying, which is
    itself suspicious, so the caller inverts this signal's sign."""
    if len(window) < 2:
        return float("nan")
    feats = np.array([r["feat"] for r in sorted(window, key=lambda r: r["ts"])])
    diffs = np.linalg.norm(np.diff(feats, axis=0), axis=1)
    return float(diffs.mean())


def low_confidence_share(window: list) -> float:
    if not window:
        return 0.0
    n_low = sum(1 for r in window if r["confidence"] < LOW_CONF_THRESHOLD)
    return n_low / len(window)


def coverage_growth(history: list, window: list, n_bins_per_dim=3, n_dims=16) -> float:
    """Fraction of this window's queries that land in a feature-space bin
    never seen before in the key's full history. Benign users saturate a
    small region quickly (this -> 0 over time); an attacker systematically
    mapping the input space keeps discovering new bins.

    n_dims restricts binning to the first n_dims of the 16-d feature vector.
    The default (16 dims x 3 bins = 3^16 ~= 43M possible bins) is
    deliberately fine-grained for the PER-KEY z-score use of this signal:
    the raw value can sit near its ceiling most of the time, and z-scoring
    against each key's own calibrated mu/sigma extracts signal from the
    remaining variance. A caller that thresholds the RAW value directly
    should pass a much coarser grid, e.g. (FLEET_BINS, FLEET_DIMS) =
    (2, 6) = 64 total bins, which saturates for repetitive traffic while
    still growing for systematic space-mapping; in a 43M-bin space nearly
    every query reads as novel regardless of behavior.
    """
    if not window or not history:
        return 0.0
    hist_before = [r for r in history if r["ts"] < window[0]["ts"]]
    seen = set(_binned_hash(r["feat"], n_bins_per_dim, n_dims) for r in hist_before)
    new_count = 0
    for r in window:
        h = _binned_hash(r["feat"], n_bins_per_dim, n_dims)
        if h not in seen:
            new_count += 1
            seen.add(h)
    return new_count / len(window)


def _binned_hash(feat, n_bins_per_dim, n_dims=16):
    # feat values are roughly in a bounded range post-projection; clip and
    # bin into a coarse grid, hash the tuple of bin indices.
    bins = tuple(int(np.clip((v + 2.0) / 4.0, 0, 0.999) * n_bins_per_dim)
                 for v in feat[:n_dims])
    return bins


def centroid(window: list) -> np.ndarray:
    return np.array([r["feat"] for r in window]).mean(axis=0)
