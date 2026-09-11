"""SENTRY live ops dashboard, a SEPARATE process on a SEPARATE port (8080).

This is a demo/observability tool, not part of the graded inference path.
It never imports the detector and never modifies sentry/api/server.py or
sentry/detect/detector.py; it only (a) drives the SAME traffic-generator
functions used everywhere else in this repo against the real /predict API,
(b) tails the same results/traffic_log.jsonl and results/alerts.jsonl files
the detector already writes, and (c) reads results/calibration.json (the
same file the detector loads) to show live threshold values. Killing this
dashboard changes nothing about the API or detector, same fail-open
guarantee the rest of the system already makes.

Layout follows patterns used by production AI/API security consoles
(HiddenLayer AIDR, Cloudflare API Shield) and general SOC-dashboard UX
guidance: a KPI strip up top, a severity-tiered incident feed (critical
multi-key campaigns first, single-key alerts as medium, benign traffic as
a muted log) with acknowledge/resolve/false-positive triage state, a full
key inventory (not just currently-active keys), and a transparent
"detection policy" view of the actual calibrated thresholds, so a SOC
analyst never has to read source code to know why something fired.

Run: python -m sentry.dashboard.server   (needs `make serve` and
`python -m sentry.detect.detector` already running in other panes)
Open: http://127.0.0.1:8080
"""
import bisect
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from sentry.api.keys import KEYS
from sentry.detect.signals import class_entropy
from sentry.traffic import attacks, benign
from sentry.traffic.client import is_alive, reset_server, server_status
from sentry.traffic.runner import run_concurrent, run_episode

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
TRAFFIC_LOG = RESULTS_DIR / "traffic_log.jsonl"
ALERTS_LOG = RESULTS_DIR / "alerts.jsonl"
CALIB_PATH = RESULTS_DIR / "calibration.json"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

FEED_WINDOW_S = 10.0      # "live" per-key stats computed over this trailing window
KEY_IDLE_AFTER_S = 30.0   # a key drops off "active" once quiet this long
MAX_ALERTS_SHOWN = 60
MAX_TRAFFIC_LINES_READ = 4000   # tail this many lines max, so a long-running
                                 # dashboard session doesn't slow down as the
                                 # log grows
TIMESERIES_WINDOW_S = 120
TIMESERIES_BUCKET_S = 2
COST_PER_QUERY_USD = 0.01  # same stated assumption as sentry/eval/run_eval.py:
                            # a modeling input, not an observed market price

app = FastAPI(title="SENTRY live ops dashboard")

_running_lock = threading.Lock()
_running = {}          # scenario_name -> bool
_status_lock = threading.Lock()
_alert_status = {}     # alert_id -> "new" | "acknowledged" | "resolved" | "false_positive"
_t_dashboard_start = time.time()


# --- log tailing --------------------------------------------------------
def _tail_lines(path: Path, max_lines: int) -> list:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        block = 65536
        data = b""
        while f.tell() > 0 and data.count(b"\n") <= max_lines:
            step = min(block, f.tell())
            f.seek(-step, 1)
            data = f.read(step) + data
            f.seek(-step, 1)
    lines = data.decode("utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def _read_jsonl_tail(path: Path, max_lines: int) -> list:
    out = []
    for line in _tail_lines(path, max_lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _alert_id(rec: dict) -> str:
    keys = rec.get("keys") or [rec.get("key")]
    basis = f"{rec.get('kind')}|{sorted(keys)}|{round(rec.get('t', 0.0), 2)}"
    return hashlib.md5(basis.encode()).hexdigest()[:12]


SEVERITY = {"fleet_coverage": "critical", "cluster": "critical", "evalue": "medium"}

LEG_DESCRIPTIONS = {
    "evalue": {
        "name": "E-value / Shiryaev–Roberts (per key)",
        "paper": "Vovk & Wang 2021",
        "summary": "Anytime-valid sequential test on each key alone. Catches loud, "
                    "sustained single-key extraction and near-duplicate "
                    "perturbation sweeps.",
        "threshold_field": None,  # SR_THRESHOLD is a fixed constant, shown separately
    },
    "cluster": {
        "name": "Campaign correlation (z-score, summed)",
        "paper": "in-house",
        "summary": "Groups concurrently-active keys by feature-centroid cosine "
                    "similarity into connected components, sums per-member z-scores. "
                    "Catches campaigns split across similar keys, where no single "
                    "key looks suspicious. Requires 2+ active keys, so it never "
                    "fires on a single-key episode.",
        "threshold_field": "cluster_threshold",
    },
    "fleet_coverage": {
        "name": "Fleet specialization gap (identity-blind)",
        "paper": "in-house",
        "summary": "Pooled fleet-wide class entropy minus the mean of each key's own "
                    "entropy. Catches an attacker who splits into narrow specialists "
                    "specifically to evade campaign correlation. The fleet looks "
                    "complete even though no single key does.",
        "threshold_field": "fleet_coverage_threshold",
    },
}


def _alert_human_action(rec: dict) -> str:
    kind = rec.get("kind")
    if kind == "evalue":
        return f"throttled key {rec.get('key')}: sequential e-value crossed the alert threshold"
    if kind == "cluster":
        keys = ", ".join(rec.get("keys", []))
        return f"throttled campaign [{keys}]: summed z-score across correlated keys crossed threshold"
    if kind == "fleet_coverage":
        keys = ", ".join(rec.get("keys", []))
        return f"throttled fleet [{keys}]: pooled fleet looks complete even though no single key does"
    return "throttled: signal threshold crossed"


# --- scenario registry (traffic simulation) -----------------------------
def _run_p(fn):
    def go():
        rng = np.random.RandomState()
        key, images, qps = fn(rng, pool="B_eval")
        run_episode(key, images, qps)
    return go


def _run_single_attack(fn):
    def go():
        rng = np.random.RandomState()
        key, images, qps = fn(rng)
        run_episode(key, images, qps)
    return go


def _run_concurrent_attack(fn):
    def go():
        rng = np.random.RandomState()
        episodes = fn(rng)
        run_concurrent(episodes)
    return go


SCENARIOS = {
    "p1-casual": {"label": "P1 casual browser (benign)", "fn": _run_p(benign.p1_casual)},
    "p2-power": {"label": "P2 power user (benign)", "fn": _run_p(benign.p2_power)},
    "p3-mobile": {"label": "P3 mobile app (benign)", "fn": _run_p(benign.p3_mobile)},
    "p4-batch": {"label": "P4 nightly batch, ~20 qps (benign)", "fn": _run_p(benign.p4_batch)},
    "a1-loud": {"label": "A1 high-volume flood", "fn": _run_single_attack(attacks.a1_loud)},
    "a3-zeroday": {"label": "A3 boundary-probing sweep, sealed pool", "fn": _run_single_attack(attacks.a3_zeroday)},
    "a6-smart": {"label": "A6 five-key split campaign", "fn": _run_concurrent_attack(attacks.a6_smart_split)},
    "a7-adaptive": {"label": "A7 adaptive five-key split", "fn": _run_concurrent_attack(attacks.a7_adaptive_decorrelated)},
}


@app.get("/", response_class=HTMLResponse)
def index():
    return TEMPLATE_PATH.read_text()


@app.get("/api/scenarios")
def api_scenarios():
    with _running_lock:
        running = dict(_running)
    return {name: {"label": v["label"], "running": running.get(name, False)}
            for name, v in SCENARIOS.items()}


@app.post("/api/reset")
def api_reset():
    """Truncates the traffic log and clears throttle state on the real API
    (same /admin/reset the existing `make demo` flow already uses), and
    clears this dashboard's own triage state. Does not touch alerts.jsonl:
    that file belongs to the detector process, which manages it on its own
    startup."""
    if not is_alive():
        return JSONResponse({"error": "API unreachable, is `make serve` running?"}, status_code=503)
    reset_server()
    with _status_lock:
        _alert_status.clear()
    return {"status": "reset"}


@app.post("/api/run/{scenario}")
def api_run(scenario: str):
    if scenario not in SCENARIOS:
        return JSONResponse({"error": f"unknown scenario '{scenario}'"}, status_code=404)
    with _running_lock:
        if _running.get(scenario):
            return JSONResponse({"error": f"'{scenario}' is already running"}, status_code=409)
        _running[scenario] = True

    def worker():
        try:
            SCENARIOS[scenario]["fn"]()
        finally:
            with _running_lock:
                _running[scenario] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started", "scenario": scenario}


def _session_starts(records: list) -> dict:
    ts_by_key = {}
    for r in records:
        ts_by_key.setdefault(r["key"], []).append(r.get("ts", 0))
    for ts in ts_by_key.values():
        ts.sort()
    return ts_by_key


def _time_to_detect(keys: list, alert_ts, ts_by_key: dict):
    """Seconds from the first query of the activity session behind an alert to
    the alert itself. A session is a run of queries with no gap longer than
    KEY_IDLE_AFTER_S; for multi-key alerts the earliest session start counts."""
    if alert_ts is None:
        return None
    starts = []
    for key in keys:
        ts = ts_by_key.get(key, [])
        i = bisect.bisect_right(ts, alert_ts) - 1
        if i < 0:
            continue
        while i > 0 and ts[i] - ts[i - 1] <= KEY_IDLE_AFTER_S:
            i -= 1
        starts.append(ts[i])
    return round(alert_ts - min(starts), 1) if starts else None


def _load_alerts():
    """Read alerts.jsonl, attach stable ids + severity + triage status."""
    alerts_raw = _read_jsonl_tail(ALERTS_LOG, 2000)
    alerts_raw.sort(key=lambda r: r.get("t", r.get("ts", 0)), reverse=True)
    ts_by_key = _session_starts(_read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ))
    out = []
    with _status_lock:
        for rec in alerts_raw[:MAX_ALERTS_SHOWN]:
            aid = _alert_id(rec)
            out.append({
                "id": aid,
                "t": rec.get("t", rec.get("ts")),
                "ts": rec.get("ts"),
                "kind": rec.get("kind"),
                "severity": SEVERITY.get(rec.get("kind"), "medium"),
                "keys": rec.get("keys") or [rec.get("key")],
                "ttd_s": _time_to_detect(rec.get("keys") or [rec.get("key")], rec.get("ts"), ts_by_key),
                "fused_score": rec.get("fused_score"),
                "threshold": rec.get("threshold"),
                "action": _alert_human_action(rec),
                "status": _alert_status.get(aid, "new"),
                # raw evidence behind the score: evalue alerts carry
                # z_breakdown/raw_signals, cluster alerts carry member_scores,
                # fleet_coverage alerts carry the entropy pair; fields an
                # alert kind does not carry are returned as null
                "z_breakdown": rec.get("z_breakdown"),
                "raw_signals": rec.get("raw_signals"),
                "member_scores": rec.get("member_scores"),
                "pooled_entropy": rec.get("pooled_entropy"),
                "individual_entropies": rec.get("individual_entropies"),
            })
    return out


@app.get("/api/feed")
def api_feed():
    now = time.time()
    api_up = is_alive()

    throttled = {}
    if api_up:
        try:
            throttled = server_status().get("throttled_keys", {})
        except Exception:
            throttled = {}

    records = _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
    by_key = {}
    for rec in records:
        by_key.setdefault(rec["key"], []).append(rec)

    keys_out = {}
    for key, recs in by_key.items():
        recent = [r for r in recs if now - r.get("ts", 0) <= FEED_WINDOW_S]
        last_ts = max((r.get("ts", 0) for r in recs), default=0)
        keys_out[key] = {
            "n_recent": len(recent),
            "rate_qps": round(len(recent) / FEED_WINDOW_S, 2),
            "entropy": round(class_entropy(recent), 3) if recent else 0.0,
            "last_seen_s_ago": max(0.0, round(now - last_ts, 1)) if last_ts else None,
            "throttled": key in throttled,
        }

    alerts_out = _load_alerts()

    with _running_lock:
        running = dict(_running)

    return {
        "now": now,
        "api_up": api_up,
        "keys": keys_out,
        "alerts": alerts_out,
        "running": running,
    }


@app.get("/api/overview")
def api_overview():
    now = time.time()
    alerts = _load_alerts()
    alerts_1h = [a for a in alerts if a.get("ts") and now - a["ts"] <= 3600]
    open_alerts = [a for a in alerts if a["status"] in ("new", "acknowledged")]
    times_to_detect = [a["ttd_s"] for a in alerts if isinstance(a.get("ttd_s"), (int, float))]
    mean_ttd = round(sum(times_to_detect) / len(times_to_detect), 1) if times_to_detect else None

    throttled = {}
    api_up = is_alive()
    if api_up:
        try:
            throttled = server_status().get("throttled_keys", {})
        except Exception:
            throttled = {}

    records = _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
    active_keys = {r["key"] for r in records if now - r.get("ts", 0) <= KEY_IDLE_AFTER_S}

    return {
        "api_up": api_up,
        "dashboard_uptime_s": round(now - _t_dashboard_start, 0),
        "known_keys": len(KEYS),
        "active_keys": len(active_keys),
        "throttled_now": len(throttled),
        "alerts_1h": len(alerts_1h),
        "open_incidents": len(open_alerts),
        "mean_time_to_detect_s": mean_ttd,
    }


@app.post("/api/alerts/{alert_id}/status")
def api_set_alert_status(alert_id: str, value: str):
    if value not in ("new", "acknowledged", "resolved", "false_positive"):
        return JSONResponse({"error": "invalid status"}, status_code=400)
    with _status_lock:
        _alert_status[alert_id] = value
    return {"id": alert_id, "status": value}


@app.get("/api/keys/registry")
def api_keys_registry():
    now = time.time()
    records = _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
    last_seen = {}
    for r in records:
        last_seen[r["key"]] = max(last_seen.get(r["key"], 0), r.get("ts", 0))

    throttled = {}
    if is_alive():
        try:
            throttled = server_status().get("throttled_keys", {})
        except Exception:
            throttled = {}

    out = []
    for key, meta in KEYS.items():
        ts = last_seen.get(key)
        if ts is None:
            status = "never seen"
        elif key in throttled:
            status = "throttled"
        elif now - ts <= KEY_IDLE_AFTER_S:
            status = "active"
        else:
            status = "idle"
        out.append({
            "key": key,
            "label": meta["label"],
            "demo_ground_truth": meta["ground_truth"],  # UI convenience only:
            # the live detector NEVER reads this field; see sentry/api/keys.py
            "status": status,
            "last_seen_s_ago": max(0.0, round(now - ts, 1)) if ts else None,
        })
    out.sort(key=lambda r: (r["status"] != "throttled", r["status"] != "active",
                             r["last_seen_s_ago"] if r["last_seen_s_ago"] is not None else 1e9))
    return {"keys": out}


@app.get("/api/keys/{key}/detail")
def api_key_detail(key: str):
    """Per-key drill-down: class histogram, confidence distribution, and a
    query-timeline sparkline, all computed from traffic_log.jsonl, the
    same raw records the detector itself reads. Read-only, no side effects."""
    now = time.time()
    records = [r for r in _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
               if r["key"] == key]
    if not records:
        return {"key": key, "n_queries": 0, "class_hist": [0]*10,
                "confidences": [], "timeline": [], "cost_usd": 0.0}

    class_hist = [0] * 10
    confidences = []
    for r in records:
        idx = r.get("class_idx")
        if idx is not None and 0 <= idx < 10:
            class_hist[idx] += 1
        confidences.append(r.get("confidence", 0.0))

    window_s, n_buckets = 60, 30
    bucket_s = window_s / n_buckets
    start = now - window_s
    timeline = [0] * n_buckets
    for r in records:
        ts = r.get("ts", 0)
        if ts < start:
            continue
        idx = int((ts - start) / bucket_s)
        if 0 <= idx < n_buckets:
            timeline[idx] += 1

    conf_buckets = [0] * 10  # confidence 0.0-1.0 in 10 bins
    for c in confidences:
        idx = min(9, max(0, int(c * 10)))
        conf_buckets[idx] += 1

    return {
        "key": key,
        "n_queries": len(records),
        "class_hist": class_hist,
        "confidence_hist": conf_buckets,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "timeline": timeline,
        "timeline_bucket_s": bucket_s,
        "cost_usd": round(len(records) * COST_PER_QUERY_USD, 2),
    }


@app.get("/api/impact")
def api_impact():
    """Cost & impact panel: for every key that has ever appeared in an
    alert, estimate spend so far at the stated $/query assumption (same
    constant sentry/eval/run_eval.py uses for the report's cost-frontier
    numbers, a modeling input rather than an observed market price). Every
    dollar figure on this panel derives from that single stated per-query
    rate."""
    alerts = _load_alerts()
    flagged_keys = set()
    for a in alerts:
        flagged_keys.update(a["keys"])

    records = _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
    counts = {}
    for r in records:
        if r["key"] in flagged_keys:
            counts[r["key"]] = counts.get(r["key"], 0) + 1

    rows = [{"key": k, "n_queries": n, "cost_usd": round(n * COST_PER_QUERY_USD, 2)}
            for k, n in counts.items()]
    rows.sort(key=lambda r: -r["cost_usd"])
    return {
        "cost_per_query_usd": COST_PER_QUERY_USD,
        "total_flagged_keys": len(flagged_keys),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 2),
        "keys": rows,
    }


@app.get("/api/policy")
def api_policy():
    calib = {}
    if CALIB_PATH.exists():
        try:
            calib = json.loads(CALIB_PATH.read_text())
        except json.JSONDecodeError:
            calib = {}

    legs = []
    for kind, desc in LEG_DESCRIPTIONS.items():
        threshold = None
        if desc["threshold_field"]:
            threshold = calib.get(desc["threshold_field"])
        elif kind == "evalue":
            threshold = 1000.0  # SR_THRESHOLD = 1/alpha, alpha=1e-3, see evalue.py
        legs.append({"kind": kind, **desc, "live_threshold": threshold})

    return {
        "legs": legs,
        "per_key_threshold": calib.get("per_key_threshold"),
        "sim_threshold": calib.get("sim_threshold"),
        "window_seconds": calib.get("window_seconds"),
        "step_seconds": calib.get("step_seconds"),
        "static_rate_limit_qps_baseline": calib.get("static_rate_limit_qps"),
        "calibrated_on_n_windows": calib.get("n_calibration_windows"),
        "calibrated_on_n_clusters": calib.get("n_benign_pairs_for_cluster_calib"),
        "calibrated_on_n_fleet_windows": calib.get("n_fleet_coverage_windows"),
    }


@app.get("/api/timeseries")
def api_timeseries():
    now = time.time()
    start = now - TIMESERIES_WINDOW_S
    n_buckets = int(TIMESERIES_WINDOW_S / TIMESERIES_BUCKET_S)

    records = _read_jsonl_tail(TRAFFIC_LOG, MAX_TRAFFIC_LINES_READ)
    alerts = _read_jsonl_tail(ALERTS_LOG, 500)

    req_buckets = [0] * n_buckets
    for r in records:
        ts = r.get("ts", 0)
        if ts < start:
            continue
        idx = int((ts - start) / TIMESERIES_BUCKET_S)
        if 0 <= idx < n_buckets:
            req_buckets[idx] += 1

    alert_buckets = [0] * n_buckets
    for a in alerts:
        ts = a.get("ts", 0)
        if ts < start:
            continue
        idx = int((ts - start) / TIMESERIES_BUCKET_S)
        if 0 <= idx < n_buckets:
            alert_buckets[idx] += 1

    return {
        "bucket_seconds": TIMESERIES_BUCKET_S,
        "window_seconds": TIMESERIES_WINDOW_S,
        "requests": req_buckets,
        "alerts": alert_buckets,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
