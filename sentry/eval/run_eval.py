"""End-to-end evaluation: runs benign (B_eval) and attack (train / A_holdout)
episodes against the LIVE API, replays the resulting logs through every
detection method, measures substitute-model fidelity and $ cost for the
attacks, and writes everything to results/metrics.json.

No number in results/metrics.json is hand-typed; every figure in the PDF
must trace back to this file. Re-running this script regenerates it.

Scale: N_TRIALS trials per scenario, with every rate reported alongside a
Wilson 95% CI. Increase N_TRIALS and re-run for tighter intervals.

Run:  python -m sentry.eval.run_eval
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    # Sets single-threaded BLAS *before* numpy/torch/sklearn initialize
    # their thread pools, so PyTorch's MPS/Accelerate threading and
    # sklearn's OpenBLAS solver run stably side by side in one process on
    # macOS. Costs nothing measurable at our array sizes
    # (a few hundred rows x 768 features).
    os.environ.setdefault(_v, "1")

import json
import time
from pathlib import Path

import numpy as np

from sentry.traffic import benign, attacks
from sentry.traffic.runner import run_episode, run_concurrent
from sentry.traffic.client import reset_server, is_alive
from sentry.detect.windows import load_log
from sentry.eval.simulate import run_detection_simulation, METHODS
from sentry.eval.fidelity import substitute_fidelity
from sentry.eval.odometer import run_odometer, pooled_query_stream
from sentry.eval.stats import wilson_ci
from sentry.model.arch import VictimCNN

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "results" / "traffic_log.jsonl"
CALIB_PATH = ROOT / "results" / "calibration.json"
METRICS_PATH = ROOT / "results" / "metrics.json"

N_TRIALS = 15
COST_PER_QUERY_USD = 0.01
CLASS_TO_IDX = {name: i for i, name in enumerate(VictimCNN.CLASSES)}

SCENARIOS = [
    {"kind": "benign", "name": "P1-casual", "fn": benign.p1_casual, "pool": "B_eval"},
    {"kind": "benign", "name": "P2-power", "fn": benign.p2_power, "pool": "B_eval"},
    {"kind": "benign", "name": "P3-mobile", "fn": benign.p3_mobile, "pool": "B_eval"},
    {"kind": "benign", "name": "P4-batch", "fn": benign.p4_batch, "pool": "B_eval"},
    {"kind": "attack_single", "name": "A1-loud", "fn": attacks.a1_loud, "pool": "train"},
    {"kind": "attack_single", "name": "A3-zeroday", "fn": attacks.a3_zeroday, "pool": "A_holdout"},
    {"kind": "attack_split", "name": "A6-smart", "fn": attacks.a6_smart_split, "pool": "train"},
    {"kind": "attack_split", "name": "A7-adaptive", "fn": attacks.a7_adaptive_decorrelated,
     "pool": "train"},
]


def load_calibration():
    with open(CALIB_PATH) as f:
        return json.load(f)


def run_trial(scenario, seed):
    rng = np.random.RandomState(seed)
    reset_server()

    if scenario["kind"] == "attack_split":
        episodes = scenario["fn"](rng, pool=scenario["pool"])
        results_by_key = run_concurrent(episodes)
        images_by_key = {k: imgs for k, imgs, _ in episodes}
    else:
        key, images, qps = scenario["fn"](rng, pool=scenario["pool"])
        results_by_key = run_episode(key, images, qps)
        images_by_key = {key: images}

    time.sleep(0.3)  # let the last log writes land on disk
    by_key = load_log(LOG_PATH)
    by_key = {k: v for k, v in by_key.items() if k in results_by_key}

    detection = run_detection_simulation(by_key, CALIB)

    latencies = [r["latency"] for results in results_by_key.values() for r in results
                 if r.get("latency") is not None]

    trial = {
        "scenario": scenario["name"],
        "kind": scenario["kind"],
        "seed": int(seed),
        "keys": list(results_by_key.keys()),
        "n_queries": {k: len(v) for k, v in results_by_key.items()},
        "detection": detection,
        "latencies": latencies,
        "odometer": compute_burn_rate(by_key),
    }

    if scenario["kind"] in ("attack_single", "attack_split"):
        trial["fidelity_cost"] = compute_fidelity_cost(
            results_by_key, images_by_key, detection["sentry_zscore"], by_key)

    return trial


def compute_burn_rate(by_key):
    """Runs the odometer on EVERY trial (benign included) and reports both
    the final LEVEL and the average BURN RATE (level / episode duration).
    The two readings answer different questions: LEVEL is cumulative, so it
    grows with query volume (a high-volume, fully-legitimate persona such
    as P4 accumulates reading from volume alone), while BURN RATE measures
    how fast leakage accrues and is compared against benign traffic in
    aggregate()."""
    stream = pooled_query_stream(by_key)
    if not stream:
        return {"level": 0.0, "duration_s": 0.0, "burn_rate": 0.0}
    readings = run_odometer(stream)
    t_start = stream[0][0]
    t_end = stream[-1][0]
    duration = max(t_end - t_start, 1e-6)
    level = readings[-1]
    return {"level": level, "duration_s": duration, "burn_rate": level / duration}


def compute_fidelity_cost(results_by_key, images_by_key, zscore_detection, by_key):
    """Pools every key's (image -> victim label) pairs the attacker actually
    received, up to the detection time (or the full episode if never
    detected), and trains a real substitute classifier on them. Also runs
    the odometer over the same pooled stream and reports its reading at the
    same two checkpoints, so odometer-vs-fidelity can be plotted directly
    from paired numbers."""
    t_detect = zscore_detection["t_detect"]
    all_images_at_detect, all_labels_at_detect = [], []
    all_images_full, all_labels_full = [], []
    n_queries_at_detect = 0
    n_queries_full = 0

    for key, results in results_by_key.items():
        imgs = images_by_key[key]
        for i, r in enumerate(results):
            if not r.get("ok") or not r.get("body"):
                continue
            cls_name = r["body"]["class_name"]
            cls_idx = CLASS_TO_IDX[cls_name]
            all_images_full.append(imgs[i])
            all_labels_full.append(cls_idx)
            n_queries_full += 1
            if t_detect is None or r["episode_t"] <= t_detect:
                all_images_at_detect.append(imgs[i])
                all_labels_at_detect.append(cls_idx)
                n_queries_at_detect += 1

    fidelity_at_detect = (substitute_fidelity(all_images_at_detect, all_labels_at_detect)
                           if t_detect is not None else None)
    fidelity_full = substitute_fidelity(all_images_full, all_labels_full)

    stream = pooled_query_stream(by_key)
    odometer_readings = run_odometer(stream)
    odometer_full = odometer_readings[-1] if odometer_readings else 0.0
    odometer_at_detect = None
    if t_detect is not None and stream:
        t_start = min(r["ts"] for recs in by_key.values() for r in recs)
        cutoff = t_start + t_detect
        idx = sum(1 for ts, _, _ in stream if ts <= cutoff) - 1
        odometer_at_detect = odometer_readings[idx] if idx >= 0 else 0.0

    return {
        "detected": t_detect is not None,
        "t_detect_s": t_detect,
        "n_queries_at_detect": n_queries_at_detect if t_detect is not None else None,
        "n_queries_full_episode": n_queries_full,
        "fidelity_at_detect": fidelity_at_detect,
        "fidelity_full_episode": fidelity_full,
        "odometer_at_detect": odometer_at_detect,
        "odometer_full_episode": odometer_full,
        "cost_usd_at_detect": (n_queries_at_detect * COST_PER_QUERY_USD
                                if t_detect is not None else None),
        "cost_usd_full_episode": n_queries_full * COST_PER_QUERY_USD,
    }


def aggregate(all_trials):
    metrics = {"per_scenario": {}, "per_method_pooled_fpr": {}, "latency_ms": {},
               "fidelity_cost": {}}

    scenario_names = sorted(set(t["scenario"] for t in all_trials))
    for name in scenario_names:
        trials = [t for t in all_trials if t["scenario"] == name]
        kind = trials[0]["kind"]
        metrics["per_scenario"][name] = {"kind": kind, "n_trials": len(trials),
                                          "methods": {}}
        for method in METHODS:
            n_fired = sum(1 for t in trials if t["detection"][method]["detected"])
            p, lo, hi = wilson_ci(n_fired, len(trials))
            fire_times = [t["detection"][method]["t_detect"] for t in trials
                           if t["detection"][method]["detected"]]
            metrics["per_scenario"][name]["methods"][method] = {
                "fire_rate": p, "ci95": [lo, hi], "n_fired": n_fired,
                "n_trials": len(trials),
                "mean_time_to_fire_s": float(np.mean(fire_times)) if fire_times else None,
            }
        odo = [t["odometer"] for t in trials]
        metrics["per_scenario"][name]["odometer"] = {
            "mean_level": _safe_mean([o["level"] for o in odo]),
            "mean_burn_rate": _safe_mean([o["burn_rate"] for o in odo]),
        }
        if kind == "attack_single" or kind == "attack_split":
            fc = [t["fidelity_cost"] for t in trials]
            metrics["fidelity_cost"][name] = {
                "mean_fidelity_at_detect": _safe_mean([f["fidelity_at_detect"] for f in fc]),
                "mean_fidelity_full_episode": _safe_mean([f["fidelity_full_episode"] for f in fc]),
                "mean_cost_usd_at_detect": _safe_mean([f["cost_usd_at_detect"] for f in fc]),
                "mean_cost_usd_full_episode": _safe_mean([f["cost_usd_full_episode"] for f in fc]),
                "mean_odometer_at_detect": _safe_mean([f["odometer_at_detect"] for f in fc]),
                "mean_odometer_full_episode": _safe_mean([f["odometer_full_episode"] for f in fc]),
                "cost_per_query_usd_assumption": COST_PER_QUERY_USD,
            }

    # Odometer validation: does the odometer reading actually track the
    # attacker's OWN measured substitute-model fidelity? Pool every
    # (odometer, fidelity) pair across every attack trial and scenario,
    # full-episode readings, since that gives the widest spread of leakage
    # amounts to correlate against.
    pairs = [(t["fidelity_cost"]["odometer_full_episode"], t["fidelity_cost"]["fidelity_full_episode"])
             for t in all_trials if t["kind"] in ("attack_single", "attack_split")]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) >= 3:
        xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 0 and np.std(ys) > 0 else None
    else:
        corr = None
    metrics["odometer_validation"] = {
        "n_pairs": len(pairs),
        "pearson_r_odometer_vs_fidelity": corr,
        "pairs": [[float(x), float(y)] for x, y in pairs],
    }

    # pooled FPR across all benign scenarios, per method
    benign_trials = [t for t in all_trials if t["kind"] == "benign"]

    # Burn-rate analysis: a LEVEL reading grows with sheer volume (P4's
    # batch persona accumulates a high odometer level from legitimate
    # traffic alone), so each scenario's average BURN RATE (level / episode
    # duration) is also z-scored against the benign trials, measuring how
    # fast leakage accrues rather than how much traffic a key sends.
    benign_rates = [t["odometer"]["burn_rate"] for t in benign_trials]
    benign_mu, benign_sigma = float(np.mean(benign_rates)), float(np.std(benign_rates))
    benign_sigma = max(benign_sigma, 1e-9)

    burn_rate_analysis = {"benign_burn_rate_mean": benign_mu,
                           "benign_burn_rate_std": benign_sigma, "per_scenario": {}}
    for name in scenario_names:
        trials = [t for t in all_trials if t["scenario"] == name]
        rates = [t["odometer"]["burn_rate"] for t in trials]
        levels = [t["odometer"]["level"] for t in trials]
        z_scores_ = [(r - benign_mu) / benign_sigma for r in rates]
        burn_rate_analysis["per_scenario"][name] = {
            "mean_level": float(np.mean(levels)),
            "mean_burn_rate": float(np.mean(rates)),
            "mean_burn_rate_z_vs_benign": float(np.mean(z_scores_)),
        }
    metrics["odometer_burn_rate"] = burn_rate_analysis

    for method in METHODS:
        n_fired = sum(1 for t in benign_trials if t["detection"][method]["detected"])
        p, lo, hi = wilson_ci(n_fired, len(benign_trials))
        metrics["per_method_pooled_fpr"][method] = {
            "fpr": p, "ci95": [lo, hi], "n_fired": n_fired, "n_trials": len(benign_trials)}

    latencies = [lat for t in all_trials for lat in t.get("latencies", [])]
    if latencies:
        arr = np.array(latencies) * 1000
        metrics["latency_ms"] = {"p50": float(np.percentile(arr, 50)),
                                  "p95": float(np.percentile(arr, 95)),
                                  "p99": float(np.percentile(arr, 99)),
                                  "n": len(arr)}
    return metrics


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


CALIB = None


def main():
    global CALIB
    if not is_alive():
        raise RuntimeError("API server not running, start it first (make serve)")
    CALIB = load_calibration()

    all_trials = []
    t0 = time.time()
    for scenario in SCENARIOS:
        base = 1000 * (SCENARIOS.index(scenario) + 1)
        for i in range(N_TRIALS):
            seed = base + i
            print(f"[eval] {scenario['name']} trial {i + 1}/{N_TRIALS} (seed={seed})...")
            trial = run_trial(scenario, seed)
            all_trials.append(trial)

    print(f"[eval] {len(all_trials)} trials complete in {time.time() - t0:.0f}s")

    metrics = aggregate(all_trials)
    metrics["meta"] = {
        "n_trials_per_scenario": N_TRIALS,
        "scenarios": [s["name"] for s in SCENARIOS],
        "cost_per_query_usd_assumption": COST_PER_QUERY_USD,
        "generated_at": time.time(),
        "calibration_snapshot": {k: v for k, v in CALIB.items() if k != "raw_samples"},
    }

    with open(METRICS_PATH, "w") as f:
        json.dump({"metrics": metrics, "trials": all_trials}, f, indent=2, default=str)
    print(f"[eval] saved -> {METRICS_PATH}")

    print("\n=== per-scenario Sentry (z-score) fire rate ===")
    for name, s in metrics["per_scenario"].items():
        m = s["methods"]["sentry_zscore"]
        print(f"  {name:14s} kind={s['kind']:14s} fire_rate={m['fire_rate']*100:5.1f}% "
              f"[{m['ci95'][0]*100:.1f}-{m['ci95'][1]*100:.1f}%] "
              f"mean_t={m['mean_time_to_fire_s']}")


if __name__ == "__main__":
    main()
