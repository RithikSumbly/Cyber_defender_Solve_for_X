"""Fit detector thresholds on B_cal (benign-only) traffic.

CRITICAL RULE (enforced by the data ledger, see model/train_victim.py):
this script must NEVER be pointed at anything but pool='B_cal'. No attack
traffic, and no B_eval/A_holdout traffic, ever touches this file, so the
FPR/TPR numbers on B_eval and A_holdout are genuine out-of-sample numbers,
not numbers the detector was tuned to hit.

Run:  python -m sentry.detect.calibrate
"""
import itertools
import json
from pathlib import Path

import numpy as np

from sentry.traffic import benign
from sentry.traffic.runner import run_episode, run_concurrent
from sentry.traffic.client import reset_server, is_alive
from sentry.detect.cluster import find_clusters
from sentry.detect.windows import load_log, sliding_windows, WINDOW_SECONDS, STEP_SECONDS
from sentry.detect.fusion import raw_signals, SIGNAL_NAMES, z_scores, fused_score
from sentry.detect.signals import centroid, class_entropy, FLEET_MIN_SAMPLES

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "results" / "traffic_log.jsonl"
CALIB_PATH = ROOT / "results" / "calibration.json"

CAL_SEEDS = list(range(1, 16))  # 15 concurrent-4-persona episodes, matching
# N_TRIALS=15 everywhere else, enough repetitions to fit
# cluster_threshold/fleet_coverage_threshold on a real, adequately-sized
# sample of concurrent-benign clusters.
PCT = 99
SIM_THRESHOLD = 0.80
STATIC_RATE_LIMIT_QPS = 10.0  # a plausible ops-chosen flat default, NOT fit to
                               # our traffic, that is the whole point of the
                               # "static rate limit" baseline in the comparison.


def generate_cal_traffic():
    """Runs all 4 personas CONCURRENTLY within each seed (not sequentially).
    This matters beyond speed: it lets the campaign-correlation threshold
    be fit on real overlapping-in-time benign traffic. Sequential personas
    never overlap, so no pair of benign keys would ever be 'active at the
    same time' to calibrate against."""
    if not is_alive():
        raise RuntimeError("API server not running, start it first (make serve)")
    reset_server()
    for seed in CAL_SEEDS:
        rng = np.random.RandomState(seed)
        episodes = [persona_fn(rng, pool="B_cal") for persona_fn in benign.PERSONAS]
        run_concurrent(episodes)


def collect_windows_with_meta():
    """Returns list of (key, window_end_ts, raw_signals_dict, centroid_vec).

    Skips windows with < 2 samples: live detection never computes raw_signals
    or a centroid for a window that small either (detector.py / simulate.py
    both gate on `len(window) >= 2` before doing so). Fitting a threshold on
    statistics the live system could never have produced isn't calibration,
    it's noise. Calibration and detection must use the same eligibility
    rule, or the threshold that results doesn't correspond to anything the
    live system actually does.
    """
    by_key = load_log(LOG_PATH)
    out = []
    for key, records in by_key.items():
        for t_end, win in sliding_windows(records, WINDOW_SECONDS, STEP_SECONDS):
            if len(win) < 2:
                continue
            hist = [r for r in records if r["ts"] <= t_end]
            raw = raw_signals(hist, win, WINDOW_SECONDS)
            out.append((key, t_end, raw, centroid(win)))
    return out


def fit_mu_sigma(windows_meta):
    values = {name: [] for name in SIGNAL_NAMES}
    for _, _, raw, _ in windows_meta:
        for name in SIGNAL_NAMES:
            v = raw.get(name)
            if v is not None and not np.isnan(v):
                values[name].append(v)
    mu = {name: float(np.mean(values[name])) if values[name] else 0.0
          for name in SIGNAL_NAMES}
    sigma = {name: float(np.std(values[name])) if values[name] else 1.0
             for name in SIGNAL_NAMES}
    for name in SIGNAL_NAMES:
        if sigma[name] < 1e-6:
            sigma[name] = 1e-6
    return mu, sigma


def fit_per_key_stats(windows_meta):
    """Per-key mu/sigma: each KNOWN calibrated key (our 4 benign personas)
    is judged against its OWN normal range, not the pooled population. This
    is what stops the fused detector from flagging the high-volume batch
    persona just for being different from the low-volume ones, which is
    exactly how a flat rate limit misfires. A key with no calibration history
    (e.g. a first-seen attacker key) has no entry here and falls back to
    the global pooled mu/sigma in fusion.z_scores.
    """
    by_key = {}
    for key, _, raw, _ in windows_meta:
        by_key.setdefault(key, {name: [] for name in SIGNAL_NAMES})
        for name in SIGNAL_NAMES:
            v = raw.get(name)
            if v is not None and not np.isnan(v):
                by_key[key][name].append(v)

    per_key_mu, per_key_sigma = {}, {}
    for key, values in by_key.items():
        per_key_mu[key] = {}
        per_key_sigma[key] = {}
        for name in SIGNAL_NAMES:
            vals = values[name]
            per_key_mu[key][name] = float(np.mean(vals)) if vals else 0.0
            sd = float(np.std(vals)) if vals else 1.0
            per_key_sigma[key][name] = sd if sd >= 1e-6 else 1e-6
    return per_key_mu, per_key_sigma


def fit_per_key_threshold(windows_meta, mu, sigma, per_key_mu, per_key_sigma):
    calib = {"mu": mu, "sigma": sigma, "per_key_mu": per_key_mu, "per_key_sigma": per_key_sigma}
    scores = []
    for key, _, raw, _ in windows_meta:
        z = z_scores(raw, calib, key)
        scores.append(fused_score(z))
    return float(np.percentile(scores, PCT)), scores


def fit_cluster_threshold(windows_meta, mu, sigma, per_key_mu, per_key_sigma, per_key_threshold):
    """Groups keys active at the same timestamp with similar feature
    centroids (cosine >= SIM_THRESHOLD) into full connected-component
    clusters (not just pairs, see sentry.detect.cluster), and calibrates
    a threshold on the cluster's SUMMED member z-score. Sum, not mean, is
    the deliberate choice: the whole point of campaign correlation is that
    several individually-unremarkable keys should combine into strong
    evidence, which only a sum (not a size-invariant average) actually
    does. The threshold itself is still fit on real benign cluster sums
    (not an arbitrary multiple), so a bigger benign cluster doesn't get a
    free pass just for being bigger.

    generate_cal_traffic() runs all 4 personas CONCURRENTLY so this fit has
    real overlapping-in-time benign clusters to calibrate against.
    """
    calib = {"mu": mu, "sigma": sigma, "per_key_mu": per_key_mu, "per_key_sigma": per_key_sigma}
    by_ts = {}
    for key, t_end, raw, cent in windows_meta:
        by_ts.setdefault(round(t_end, 1), []).append((key, raw, cent))

    pooled_scores = []
    for t_end, entries in by_ts.items():
        centroids = {key: cent for key, _, cent in entries}
        raws = {key: raw for key, raw, _ in entries}
        for group in find_clusters(centroids, SIM_THRESHOLD):
            member_scores = [fused_score(z_scores(raws[k], calib, k)) for k in group]
            pooled_scores.append(float(np.sum(member_scores)))

    if len(pooled_scores) >= 20:
        cluster_threshold = float(np.percentile(pooled_scores, PCT))
        source = f"calibrated_on_{len(pooled_scores)}_benign_clusters"
    else:
        # Guard for short calibration runs with fewer than 20 same-timestamp
        # benign clusters: derive the threshold from the per-key threshold
        # (1.2x) and record the source. The shipped calibration fits it
        # directly on observed benign clusters (cluster_threshold_source).
        cluster_threshold = per_key_threshold * 1.2
        source = f"fallback (only {len(pooled_scores)} benign clusters observed)"

    return cluster_threshold, source, len(pooled_scores)


def fit_fleet_coverage_threshold(by_key):
    """Identity-BLIND signal: the SPECIALIZATION GAP between how narrow each
    individual active key looks (its own class entropy) and how complete
    the FLEET looks pooled together (pooled class entropy).

    gap = pooled_fleet_entropy - mean(individual_key_entropy)

    An attacker who deliberately restricts each key to a narrow class
    subset (see sentry/traffic/attacks.py::a7_adaptive_decorrelated) to
    evade centroid-similarity clustering shows a large gap (~0.45
    measured): each key looks like a narrow specialist, but the fleet
    pooled together covers everything. Real benign customers don't
    produce this pattern: individuals are already reasonably diverse, so
    pooling them adds little apparent completeness (~0.07 measured gap).
    No legitimate customer relationship requires "I only ever ask about
    two things, but my four neighbors' two things happen to be the other
    eight."

    Calibrated on the same real concurrent B_cal traffic
    generate_cal_traffic() already produces.

    A key only counts toward the pool once it has FLEET_MIN_SAMPLES
    samples in its window (signals.py); entropy computed from a
    near-empty window is dominated by sampling noise rather than actual
    behavior, so a sparse, just-started key is excluded from the pool
    until it has enough data to contribute a meaningful reading.
    """
    all_records = [r for records in by_key.values() for r in records]
    if not all_records:
        return 0.0, 1.0, 1.0, 0
    all_records.sort(key=lambda r: r["ts"])
    t_start, t_end = all_records[0]["ts"], all_records[-1]["ts"]

    values = []
    t = t_start + WINDOW_SECONDS
    while t <= t_end + STEP_SECONDS:
        window = [r for r in all_records if t - WINDOW_SECONDS < r["ts"] <= t]
        by_key_window = {}
        for r in window:
            by_key_window.setdefault(r["key"], []).append(r)
        fleet_eligible = [k for k, recs in by_key_window.items() if len(recs) >= FLEET_MIN_SAMPLES]
        if len(fleet_eligible) >= 2:
            fleet_pooled = [r for r in window if r["key"] in fleet_eligible]
            individual_entropies = [class_entropy(by_key_window[k]) for k in fleet_eligible]
            pooled_entropy = class_entropy(fleet_pooled)
            gap = pooled_entropy - float(np.mean(individual_entropies))
            values.append(gap)
        t += STEP_SECONDS

    mu = float(np.mean(values)) if values else 0.0
    sigma = float(np.std(values)) if values else 1.0
    if sigma < 1e-6:
        sigma = 1e-6
    threshold = float(np.percentile(values, PCT)) if values else 1.0
    return mu, sigma, threshold, len(values)


def fit_baseline_thresholds(windows_meta):
    """Single-signal baselines used for the head-to-head comparison table:
      - distance_only: fires when consecutive queries are near-duplicates
        (a simplified, single-signal reproduction of the PRADA (2019) idea).
      - mmd_style: fires when a window's mean feature vector drifts from the
        calibration-global mean (a simplified, single-signal proxy for a
        fixed-window MMD two-sample test).
    Both are simplified reproductions of the published ideas, used as
    comparison baselines rather than line-for-line implementations.
    """
    dists = [raw["near_duplicate"] for _, _, raw, _ in windows_meta
             if raw.get("near_duplicate") is not None]
    distance_only_threshold = float(np.percentile(dists, 100 - PCT)) if dists else 0.0

    all_feats = [c for _, _, _, c in windows_meta]
    cal_global_feat_mean = np.mean(all_feats, axis=0)
    mmd_stats = [float(np.sum((c - cal_global_feat_mean) ** 2)) for c in all_feats]
    mmd_threshold = float(np.percentile(mmd_stats, PCT)) if mmd_stats else 0.0

    return {
        "static_rate_limit_qps": STATIC_RATE_LIMIT_QPS,
        "distance_only_threshold": distance_only_threshold,
        "mmd_threshold": mmd_threshold,
        "cal_global_feat_mean": cal_global_feat_mean.tolist(),
    }


def main():
    print("[calibrate] generating B_cal benign traffic...")
    generate_cal_traffic()

    print("[calibrate] computing sliding-window signals...")
    windows_meta = collect_windows_with_meta()
    print(f"[calibrate] {len(windows_meta)} calibration windows")

    mu, sigma = fit_mu_sigma(windows_meta)
    per_key_mu, per_key_sigma = fit_per_key_stats(windows_meta)
    per_key_threshold, scores = fit_per_key_threshold(
        windows_meta, mu, sigma, per_key_mu, per_key_sigma)
    cluster_threshold, source, n_pairs = fit_cluster_threshold(
        windows_meta, mu, sigma, per_key_mu, per_key_sigma, per_key_threshold)
    baseline_calib = fit_baseline_thresholds(windows_meta)

    by_key_for_fleet = load_log(LOG_PATH)
    fleet_mu, fleet_sigma, fleet_threshold, n_fleet_windows = fit_fleet_coverage_threshold(
        by_key_for_fleet)

    # Raw per-signal calibration samples, kept for the e-value detector mode
    # (sentry.detect.evalue): empirical conformal p-values need the actual
    # benign-only reference distribution, not just its mean/std. Kept BOTH
    # pooled (fallback, for unseen keys) and per-key (for known keys),
    # same reasoning as per_key_mu/sigma above: a pooled-only reference
    # would flag the batch persona for being different from everyone else.
    raw_samples = {name: [] for name in SIGNAL_NAMES}
    per_key_raw_samples = {}
    for key, _, raw, _ in windows_meta:
        per_key_raw_samples.setdefault(key, {name: [] for name in SIGNAL_NAMES})
        for name in SIGNAL_NAMES:
            v = raw.get(name)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                raw_samples[name].append(v)
                per_key_raw_samples[key][name].append(v)

    calibration = {
        "mu": mu,
        "sigma": sigma,
        "per_key_mu": per_key_mu,
        "per_key_sigma": per_key_sigma,
        "per_key_threshold": per_key_threshold,
        "cluster_threshold": cluster_threshold,
        "cluster_threshold_source": source,
        "n_benign_pairs_for_cluster_calib": n_pairs,
        "sim_threshold": SIM_THRESHOLD,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "percentile": PCT,
        "n_calibration_windows": len(windows_meta),
        "cal_seeds": CAL_SEEDS,
        "raw_samples": raw_samples,
        "per_key_raw_samples": per_key_raw_samples,
        "fleet_coverage_mu": fleet_mu,
        "fleet_coverage_sigma": fleet_sigma,
        "fleet_coverage_threshold": fleet_threshold,
        "n_fleet_coverage_windows": n_fleet_windows,
        **baseline_calib,
    }
    CALIB_PATH.parent.mkdir(exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"[calibrate] mu={mu}")
    print(f"[calibrate] sigma={sigma}")
    print(f"[calibrate] per_key_threshold={per_key_threshold:.3f} "
          f"(P{PCT} of {len(scores)} benign window scores)")
    print(f"[calibrate] cluster_threshold={cluster_threshold:.3f} ({source})")
    print(f"[calibrate] fleet_coverage: mu={fleet_mu:.3f} sigma={fleet_sigma:.3f} "
          f"threshold={fleet_threshold:.3f} (n={n_fleet_windows} pooled windows)")
    print(f"[calibrate] saved -> {CALIB_PATH}")


if __name__ == "__main__":
    main()
