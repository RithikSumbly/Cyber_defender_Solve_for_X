"""Replays one episode's logged traffic through every detection method at
once, using the exact same window/signal/fusion code the live API and the
live detector worker use, which is what makes the eval numbers a faithful
measurement of the deployed logic, not a separate simulation of it.
"""
import numpy as np

from sentry.detect.windows import WINDOW_SECONDS, STEP_SECONDS
from sentry.detect.fusion import raw_signals, z_scores, fused_score, top_signal
from sentry.detect.signals import (centroid, class_entropy, FLEET_MIN_SAMPLES,
                                    CROSS_KEY_PERSISTENCE_POLLS)
from sentry.detect.evalue import window_e_value, ShiryaevRoberts, SR_THRESHOLD
from sentry.detect.cluster import find_clusters
from sentry.eval.baselines import static_rate_fires, distance_only_fires, mmd_style_fires

METHODS = ["static_rate", "distance_only", "mmd_style", "sentry_zscore", "sentry_evalue",
           "fleet_coverage", "sentry_combined"]

# sentry_combined is the primary/headline detector: three independent
# legs, OR'd together, each covering a distinct extraction pattern:
#   1. e-value per-key (sentry_evalue): an anytime-valid sequential test
#      for sustained single-key extraction, built so continuous re-checking
#      does not accumulate false alarms the way a raw percentile threshold
#      does.
#   2. z-score campaign correlation: catches keys grouped by similar
#      feature centroids (cosine >= sim_threshold); it requires 2+ active
#      keys, so it adds zero FPR to single-key episodes.
#   3. fleet_coverage: identity-BLIND "specialization gap": pooled class
#      entropy across every currently-active key minus the MEAN of their
#      individual class entropies. Catches an adaptive attacker who
#      decorrelates their keys' content specifically to evade leg 2 (see
#      sentry/traffic/attacks.py::a7_adaptive_decorrelated for the
#      adversarial test case this leg is validated against); see the
#      docstring in sentry/detect/calibrate.py::fit_fleet_coverage_threshold
#      for the measured separation between attack and benign traffic.


def run_detection_simulation(records_by_key: dict, calib: dict,
                              window=WINDOW_SECONDS, step=STEP_SECONDS):
    all_ts = [r["ts"] for recs in records_by_key.values() for r in recs]
    if not all_ts:
        return {m: {"detected": False, "t_detect": None, "keys": None, "signal": None}
                for m in METHODS}
    t_start, t_end = min(all_ts), max(all_ts)

    sr = {k: ShiryaevRoberts(SR_THRESHOLD) for k in records_by_key}
    results = {m: {"detected": False, "t_detect": None, "keys": None, "signal": None}
               for m in METHODS}
    cluster_fired = False
    cluster_streak = {}   # frozenset(group) -> consecutive polls over threshold
    fleet_streak = {}     # frozenset(fleet keys) -> consecutive polls over threshold

    def mark(method, t, keys, signal=None):
        if not results[method]["detected"]:
            results[method] = {"detected": True, "t_detect": t - t_start,
                                "keys": keys, "signal": signal}

    t = t_start + window
    while t <= t_end + step:
        active = {}
        window_sizes = {}
        pooled_window = []
        for key, records in records_by_key.items():
            hist = [r for r in records if r["ts"] <= t]
            win = [r for r in records if t - window < r["ts"] <= t]
            pooled_window.extend(win)
            window_sizes[key] = len(win)
            if len(win) < 2:
                continue
            raw = raw_signals(hist, win, window)
            cent = centroid(win)
            active[key] = (raw, cent)

            if static_rate_fires(raw, calib):
                mark("static_rate", t, [key])
            if distance_only_fires(raw, calib):
                mark("distance_only", t, [key])
            if mmd_style_fires(cent, calib):
                mark("mmd_style", t, [key])

            z = z_scores(raw, calib, key)
            score = fused_score(z)
            if score >= calib["per_key_threshold"]:
                mark("sentry_zscore", t, [key], top_signal(z))

            e = window_e_value(raw, calib, key)
            R = sr[key].update(e)
            if R >= SR_THRESHOLD:
                mark("sentry_evalue", t, [key])
                mark("sentry_combined", t, [key], "evalue")

        if not cluster_fired:
            centroids = {k: v[1] for k, v in active.items()}
            seen_cluster_ids = set()
            for group in find_clusters(centroids, calib["sim_threshold"]):
                member_scores = [fused_score(z_scores(active[k][0], calib, k)) for k in group]
                group_id = frozenset(group)
                seen_cluster_ids.add(group_id)
                if float(np.sum(member_scores)) >= calib["cluster_threshold"]:
                    cluster_streak[group_id] = cluster_streak.get(group_id, 0) + 1
                else:
                    cluster_streak[group_id] = 0
                if cluster_streak[group_id] >= CROSS_KEY_PERSISTENCE_POLLS:
                    cluster_fired = True
                    mark("sentry_zscore", t, group, "campaign_correlation")
                    mark("sentry_combined", t, group, "campaign_correlation")
                    break
            for gid in list(cluster_streak):
                if gid not in seen_cluster_ids:
                    cluster_streak[gid] = 0

        fleet_eligible = [k for k in active if window_sizes[k] >= FLEET_MIN_SAMPLES]
        fleet_id = frozenset(fleet_eligible) if len(fleet_eligible) >= 2 else None
        if fleet_id is not None and "fleet_coverage_threshold" in calib:
            fleet_pooled = [r for r in pooled_window if r["key"] in fleet_eligible]
            individual_entropies = [active[k][0]["entropy"] for k in fleet_eligible]
            pooled_entropy = class_entropy(fleet_pooled)
            gap = pooled_entropy - float(np.mean(individual_entropies))
            if gap >= calib["fleet_coverage_threshold"]:
                fleet_streak[fleet_id] = fleet_streak.get(fleet_id, 0) + 1
            else:
                fleet_streak[fleet_id] = 0
            if fleet_streak[fleet_id] >= CROSS_KEY_PERSISTENCE_POLLS:
                fleet_keys = list(fleet_eligible)
                mark("fleet_coverage", t, fleet_keys)
                mark("sentry_combined", t, fleet_keys, "fleet_coverage")
        for fid in list(fleet_streak):
            if fid != fleet_id:
                fleet_streak[fid] = 0

        t += step

    return results
