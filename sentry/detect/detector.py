"""Live detector worker, a SEPARATE OS process from the API.

It only ever reads results/traffic_log.jsonl (written by the API) and, on
firing, calls the API's /admin/throttle/<key> endpoint over HTTP. It never
imports sentry.api.server and the API never imports this module. Stop this
process (Ctrl+C, or `make kill` to stop every SENTRY process) and /predict
keeps serving, un-throttled and un-monitored: that is the fail-open
property the video demonstrates.

Detection runs three complementary legs, each covering a distinct
extraction pattern:

1. Per-key e-values + a Shiryaev-Roberts e-detector. A fixed percentile
   threshold re-checked on every 1-second window is many correlated
   chances to false-alarm over an episode, so per-key alerting accumulates
   evidence as a sequential test instead. Measured: 0% FPR across 60
   held-out benign trials, at a detection latency of ~5-10s.
2. Campaign correlation: the summed z-score of a connected component of
   similar, concurrently-active keys. It requires 2+ active keys, so it
   adds cross-key coverage without adding single-key false positives, and
   it catches split attacks such as A6, where no single key looks
   suspicious on its own. See results/metrics.json for per-leg detection
   rates.
3. Fleet specialization gap (fleet_coverage): pools EVERY currently-active
   key's queries (identity-blind, independent of whether the keys look
   similar to each other) and fires on a large SPECIALIZATION GAP: pooled
   fleet-wide class entropy minus the mean of each active key's own class
   entropy. This catches an adaptive attacker who decorrelates keys'
   content to evade centroid-similarity campaign correlation (see
   sentry/traffic/attacks.py::a7_adaptive_decorrelated, the adversarial
   test case this leg is validated against): each key looks like a narrow,
   unremarkable specialist individually, but the fleet pooled together
   looks suspiciously complete.

Run:  python -m sentry.detect.detector [--duration SECONDS]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from sentry.detect.windows import load_log, WINDOW_SECONDS
from sentry.detect.fusion import raw_signals, z_scores, fused_score, top_signal
from sentry.detect.signals import (centroid, class_entropy, FLEET_MIN_SAMPLES,
                                    CROSS_KEY_PERSISTENCE_POLLS)
from sentry.detect.cluster import find_clusters
from sentry.detect.evalue import window_e_value, ShiryaevRoberts, SR_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "results" / "traffic_log.jsonl"
CALIB_PATH = ROOT / "results" / "calibration.json"
ALERTS_PATH = ROOT / "results" / "alerts.jsonl"
API_BASE = "http://127.0.0.1:8000"

POLL_INTERVAL = 1.0  # matches sentry.detect.windows.STEP_SECONDS exactly; the
                      # e-value/Shiryaev-Roberts FPR numbers in results/metrics.json
                      # were measured at this cadence; polling faster would update
                      # the SR tracker more often than what was validated.
COOLDOWN_S = 15.0  # don't re-alert the same key/cluster within this window


def load_calibration():
    with open(CALIB_PATH) as f:
        return json.load(f)


def throttle(key: str):
    try:
        requests.post(f"{API_BASE}/admin/throttle/{key}",
                       params={"min_interval": 1.0, "duration_s": 120.0},
                       timeout=2.0)
    except requests.RequestException as e:
        print(f"[detector] WARN could not reach API to throttle {key}: {e}")


def log_alert(rec: dict):
    with open(ALERTS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    kind = rec["kind"]
    if kind == "evalue":
        print(f"[ALERT] key={rec['key']} signal=evalue (Shiryaev-Roberts) "
              f"wealth={rec['fused_score']:.1f} threshold={rec['threshold']:.0f} "
              f"top_z_signal={rec['signal']} t={rec['t']:.1f}s")
    elif kind == "fleet_coverage":
        print(f"[ALERT] fleet specialization-gap anomaly keys={rec['keys']} "
              f"gap={rec['fused_score']:.3f} "
              f"threshold={rec['threshold']:.3f} t={rec['t']:.1f}s "
              f"(identity-blind: each key looks narrow individually, but the "
              f"fleet pooled together looks suspiciously complete)")
    else:
        print(f"[ALERT] campaign correlation keys={rec['keys']} "
              f"summed_pooled_score={rec['fused_score']:.2f} "
              f"threshold={rec['threshold']:.2f} t={rec['t']:.1f}s")


def run(duration=None):
    calib = load_calibration()
    if ALERTS_PATH.exists():
        ALERTS_PATH.unlink()

    alerted_single = {}   # key -> last alert time
    alerted_cluster = {}   # frozenset(group) -> last alert time
    alerted_fleet = {}      # frozenset(active keys) -> last alert time
    sr_trackers = {}        # key -> ShiryaevRoberts, persistent across polls
    cluster_streak = {}      # frozenset(group) -> consecutive polls over threshold
    fleet_streak = {}        # frozenset(fleet keys) -> consecutive polls over threshold

    t_start = time.time()
    print(f"[detector] started, per_key mode=e-value (SR threshold={SR_THRESHOLD:.0f}) "
          f"cluster_threshold={calib['cluster_threshold']:.2f}")

    while True:
        now = time.time()
        if duration is not None and now - t_start > duration:
            break

        by_key = load_log(LOG_PATH)
        active_windows = {}  # key -> (raw, centroid, window_records)
        window_sizes = {}    # key -> len(window), for the fleet-eligibility gate
        pooled_window = []

        for key, records in by_key.items():
            window = [r for r in records if now - WINDOW_SECONDS < r["ts"] <= now]
            history = [r for r in records if r["ts"] <= now]
            pooled_window.extend(window)
            window_sizes[key] = len(window)
            if len(window) < 2:
                continue
            raw = raw_signals(history, window, WINDOW_SECONDS)
            z = z_scores(raw, calib, key)
            score = fused_score(z)
            active_windows[key] = (raw, z, score, centroid(window))

            if key not in sr_trackers:
                sr_trackers[key] = ShiryaevRoberts(SR_THRESHOLD)
            e = window_e_value(raw, calib, key)
            R = sr_trackers[key].update(e)

            last = alerted_single.get(key, 0)
            if R >= SR_THRESHOLD and now - last > COOLDOWN_S:
                alerted_single[key] = now
                throttle(key)
                log_alert({"kind": "evalue", "key": key, "signal": top_signal(z),
                           "fused_score": R, "threshold": SR_THRESHOLD,
                           "t": now - t_start, "ts": now,
                           "z_breakdown": z, "raw_signals": raw})

        centroids = {k: v[3] for k, v in active_windows.items()}
        seen_cluster_ids = set()
        for group in find_clusters(centroids, calib["sim_threshold"]):
            member_scores = [active_windows[k][2] for k in group]
            pooled = float(np.sum(member_scores))
            group_id = frozenset(group)
            seen_cluster_ids.add(group_id)
            if pooled >= calib["cluster_threshold"]:
                cluster_streak[group_id] = cluster_streak.get(group_id, 0) + 1
            else:
                cluster_streak[group_id] = 0
            last = alerted_cluster.get(group_id, 0)
            if (cluster_streak[group_id] >= CROSS_KEY_PERSISTENCE_POLLS
                    and now - last > COOLDOWN_S):
                alerted_cluster[group_id] = now
                for k in group:
                    throttle(k)
                log_alert({"kind": "cluster", "keys": group, "fused_score": pooled,
                           "threshold": calib["cluster_threshold"],
                           "t": now - t_start, "ts": now,
                           "member_scores": {k: active_windows[k][2] for k in group},
                           "persistence_polls": cluster_streak[group_id]})
        # a group that didn't appear this poll (keys dropped out, or no
        # longer similar enough) stops accumulating a streak
        for gid in list(cluster_streak):
            if gid not in seen_cluster_ids:
                cluster_streak[gid] = 0

        fleet_eligible = [k for k in active_windows if window_sizes[k] >= FLEET_MIN_SAMPLES]
        fleet_id = frozenset(fleet_eligible) if len(fleet_eligible) >= 2 else None
        if fleet_id is not None and "fleet_coverage_threshold" in calib:
            fleet_pooled = [r for r in pooled_window if r["key"] in fleet_eligible]
            individual_entropies = {k: active_windows[k][0]["entropy"] for k in fleet_eligible}
            pooled_entropy = class_entropy(fleet_pooled)
            gap = pooled_entropy - float(np.mean(list(individual_entropies.values())))
            if gap >= calib["fleet_coverage_threshold"]:
                fleet_streak[fleet_id] = fleet_streak.get(fleet_id, 0) + 1
            else:
                fleet_streak[fleet_id] = 0
            last = alerted_fleet.get(fleet_id, 0)
            if (fleet_streak[fleet_id] >= CROSS_KEY_PERSISTENCE_POLLS
                    and now - last > COOLDOWN_S):
                fleet_keys = list(fleet_eligible)
                alerted_fleet[fleet_id] = now
                for k in fleet_keys:
                    throttle(k)
                log_alert({"kind": "fleet_coverage", "keys": fleet_keys,
                           "fused_score": gap,
                           "threshold": calib["fleet_coverage_threshold"],
                           "t": now - t_start, "ts": now,
                           "pooled_entropy": pooled_entropy,
                           "individual_entropies": individual_entropies,
                           "persistence_polls": fleet_streak[fleet_id]})
        # the active fleet composition changed (someone joined/left), reset
        # every OTHER tracked fleet composition's streak, same reasoning as
        # the cluster loop above
        for fid in list(fleet_streak):
            if fid != fleet_id:
                fleet_streak[fid] = 0

        time.sleep(POLL_INTERVAL)

    print("[detector] stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None,
                     help="seconds to run then exit (omit to run until killed)")
    args = ap.parse_args()
    run(duration=args.duration)
