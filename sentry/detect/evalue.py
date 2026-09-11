"""Per-key sequential detection: e-values + a Shiryaev-Roberts e-detector.

Design rationale: a fixed percentile threshold re-checked on every 1-second
step over a sliding window is continuous monitoring with optional
stopping, and a fresh percentile check at each step is not a sequential
test. This module accumulates evidence per key with anytime-valid e-values
instead (Vovk & Wang, "E-values: Calibration, Combination and
Applications", Annals of Statistics 49(3), 2021):

  1. Per-signal empirical conformal p-value against the B_cal reference
     distribution (already computed and stored by calibrate.py).
  2. Calibrator p -> e = kappa * p^(kappa-1), which integrates to 1 over
     [0,1] for any kappa in (0,1), making e a valid bet against the null.
  3. The weighted arithmetic mean of e-values is a valid e-value under
     ARBITRARY dependence between signals (Vovk & Wang, Thm 3.1), so the
     five correlated signals combine directly, with no second calibration
     stage.
  4. Shiryaev-Roberts accumulation R_t = (1 + R_{t-1}) * E_t, alerting
     when R_t >= 1/alpha. By Ville's inequality this bounds the false-alarm
     probability by alpha for a SEQUENCE OF INDEPENDENT windows.

Empirical validation: SENTRY's windows slide with 5s width / 1s step, so
consecutive windows share up to 80% of their queries. The 1/alpha
threshold is therefore validated empirically as well: its false-positive
rate is measured on held-out B_eval traffic and reported in
results/metrics.json (per_method_pooled_fpr.sentry_evalue).
"""
import numpy as np

from sentry.detect.fusion import SIGNAL_NAMES, DIRECTION

KAPPA = 0.4
ALPHA = 1e-3
SR_THRESHOLD = 1.0 / ALPHA  # = 1000

# Equal weights by default; the theory permits any fixed weights w/ sum=1
# without breaking validity (Vovk & Wang Thm 3.1); we keep it simple here.
WEIGHTS = {name: 1.0 / len(SIGNAL_NAMES) for name in SIGNAL_NAMES}


def p_to_e(p: float, kappa: float = KAPPA) -> float:
    p = min(max(p, 1e-12), 1.0)
    return kappa * p ** (kappa - 1.0)


def empirical_p_value(raw_value: float, cal_samples: list, direction: int) -> float:
    """Conformal p-value: probability, under the benign-only calibration
    distribution, of seeing a value at least this extreme in the suspicious
    direction. direction=+1: extreme means >=. direction=-1: extreme means <=."""
    n = len(cal_samples)
    if n == 0:
        return 1.0
    arr = np.asarray(cal_samples)
    if direction > 0:
        count = int(np.sum(arr >= raw_value))
    else:
        count = int(np.sum(arr <= raw_value))
    return (1 + count) / (1 + n)


def window_e_value(raw: dict, calib: dict, key: str = None) -> float:
    """raw: output of fusion.raw_signals() for one window.
    calib: the loaded calibration.json (must contain 'raw_samples', and
    optionally 'per_key_raw_samples'). Uses the key's OWN reference
    distribution when one was calibrated for it, else the pooled one, same
    reasoning as fusion.z_scores: a pooled-only reference makes the
    highest-volume legitimate persona look anomalous just for differing
    from everyone else."""
    per_key_samples = calib.get("per_key_raw_samples", {}).get(key) if key else None
    e_terms = []
    for name in SIGNAL_NAMES:
        val = raw.get(name)
        if val is None:
            continue
        ref = (per_key_samples[name] if per_key_samples and per_key_samples.get(name)
               else calib["raw_samples"][name])
        p = empirical_p_value(val, ref, DIRECTION[name])
        e_terms.append(WEIGHTS[name] * p_to_e(p))
    if not e_terms:
        return 1.0  # neutral bet: no evidence either way
    return float(sum(e_terms))


class ShiryaevRoberts:
    """One instance per key. Call .update(E) each new window; .fired tells
    you whether R has crossed 1/alpha since the last .reset()."""

    def __init__(self, threshold: float = SR_THRESHOLD):
        self.R = 0.0
        self.threshold = threshold
        self.fired = False
        self.history = []

    def update(self, e_value: float) -> float:
        self.R = (1.0 + self.R) * e_value
        self.history.append(self.R)
        if self.R >= self.threshold:
            self.fired = True
        return self.R

    def reset(self):
        self.R = 0.0
        self.fired = False
