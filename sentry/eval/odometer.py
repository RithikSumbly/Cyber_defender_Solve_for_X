"""The extraction odometer: a pooled, cross-key leakage ledger.

Per-account detection asks "is this account an attacker?". An attacker
willing to lose an account every few thousand queries can keep opening new
ones, so the odometer complements it with a second question: "how much of
my model has left the building, regardless of who carried it?" This is the
same move differential privacy makes by accounting for cumulative
information loss rather than judging analysts.

Design: one global leak curve (no per-region calibration), validated by
correlating the odometer's reading against the ATTACKER'S OWN measured
substitute-model fidelity (sentry.eval.fidelity) at the same checkpoints:
a real measurement, not an assumed proxy.

Per-query leak = I(response) x N(x | history-so-far):
  I(response): how much we told them, derived from confidence: an
    uncertain answer reveals more about where the boundary sits than a
    confident one. I = 1 - confidence, in [0, 1].
  N(x | history): how much was new. 1 if this query's (coarsely binned)
    feature vector was never seen before in the pooled stream, decaying
    with repetition, reusing the same coverage logic as the
    coverage_growth detector signal, but evaluated per query and pooled
    ACROSS EVERY KEY, not per-key, which is the whole point: splitting
    across accounts does not reduce what the odometer sees.
"""
import numpy as np

from sentry.detect.signals import _binned_hash

N_BINS_PER_DIM = 3


def run_odometer(query_stream: list) -> list:
    """query_stream: list of (ts, confidence, feat) tuples, ALREADY POOLED
    across every key in the campaign, sorted by timestamp. Returns the
    cumulative leak reading after each query (same length as input)."""
    seen_bins = {}  # bin -> visit count, for a soft decay on repeats
    cumulative = 0.0
    readings = []
    for ts, confidence, feat in query_stream:
        info = max(0.0, 1.0 - confidence)
        h = _binned_hash(feat, N_BINS_PER_DIM)
        visits = seen_bins.get(h, 0)
        novelty = 1.0 / (1.0 + visits)   # 1.0 first time, decays with repeats
        seen_bins[h] = visits + 1
        cumulative += info * novelty
        readings.append(cumulative)
    return readings


def pooled_query_stream(records_by_key: dict) -> list:
    """Flattens every key's traffic-log records into one pooled, time-sorted
    stream of (ts, confidence, feat); this pooling, not per-key isolation,
    is what makes the odometer immune to account-splitting."""
    all_records = [r for records in records_by_key.values() for r in records]
    all_records.sort(key=lambda r: r["ts"])
    return [(r["ts"], r["confidence"], np.array(r["feat"])) for r in all_records
            if "confidence" in r]
