"""Turns raw per-window signals into a single calibrated, explainable score.

Fusion rule: each raw signal is converted to a z-score against its
benign-only calibration mean/std, clipped at 0 (only "more suspicious than
typical benign" deviations count), then summed. The signal with the
largest positive z-score is reported as the one that "fired"; this is
what lets the alert line name a cause instead of just a number.
"""
import math
import numpy as np

from sentry.detect import signals as sig

SIGNAL_NAMES = ["rate", "entropy", "near_duplicate", "low_confidence", "coverage_growth"]

# direction: +1 means "higher raw value = more suspicious",
#            -1 means "lower raw value = more suspicious" (near-duplicate distance)
DIRECTION = {"rate": 1, "entropy": 1, "near_duplicate": -1,
             "low_confidence": 1, "coverage_growth": 1}


def raw_signals(history: list, window: list, window_span_s: float) -> dict:
    dist = sig.mean_consecutive_distance(window)
    return {
        "rate": sig.rate_qps(window, window_span_s),
        "entropy": sig.class_entropy(window),
        "near_duplicate": dist if not math.isnan(dist) else None,
        "low_confidence": sig.low_confidence_share(window),
        "coverage_growth": sig.coverage_growth(history, window),
    }


def z_scores(raw: dict, calib: dict, key: str = None) -> dict:
    """Normalizes against KEY'S OWN calibrated baseline when one exists
    (calib['per_key_mu'][key]), falling back to the global pooled baseline
    otherwise (calib['mu']). This matters: a global-only baseline makes the
    highest-volume LEGITIMATE persona look like an outlier relative to
    everyone else's traffic, which is exactly how a static rate limit
    misfires, so the fused detector normalizes per key by design. A
    brand-new key (e.g. a first-time attacker) has no history yet and
    correctly falls back to the population baseline until it earns one.
    """
    per_key_mu = calib.get("per_key_mu", {}).get(key) if key else None
    per_key_sigma = calib.get("per_key_sigma", {}).get(key) if key else None

    z = {}
    for name in SIGNAL_NAMES:
        val = raw.get(name)
        if val is None:
            z[name] = 0.0
            continue
        if per_key_mu is not None and name in per_key_mu:
            mu = per_key_mu[name]
            sd = max(per_key_sigma[name], 1e-6)
        else:
            mu = calib["mu"][name]
            sd = max(calib["sigma"][name], 1e-6)
        direction = DIRECTION[name]
        z_raw = direction * (val - mu) / sd
        z[name] = max(0.0, z_raw)
    return z


def fused_score(z: dict) -> float:
    return float(sum(z.values()))


def top_signal(z: dict) -> str:
    if not z or max(z.values()) <= 0:
        return "none"
    return max(z, key=z.get)
