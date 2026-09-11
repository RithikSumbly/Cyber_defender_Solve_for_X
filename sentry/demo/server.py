"""SENTRY guided walkthrough.

A visitor-paced tour of the live system. Each step sends real traffic from
sentry.traffic to the running inference API, and each result is read back
from the alerts the detector writes. The operations console is served from
the same origin and embedded beside the steps, so every number shown in the
tour can be checked in the console itself.

Run:   make walkthrough      (or: python -m sentry.demo.server)
Open:  http://127.0.0.1:8081
"""
import atexit
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from sentry.api.keys import ADAPTIVE_SPLIT_KEYS, SMART_SPLIT_KEYS
from sentry.dashboard import server as console
from sentry.traffic import attacks, benign
from sentry.traffic.client import is_alive, reset_server
from sentry.traffic.runner import run_concurrent, run_episode

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
PORT = 8081

PROFILE_QUERY_CAP = {"P1-casual": 5, "P2-power": 12, "P3-mobile": 8, "P4-batch": 60}
GAP_BETWEEN_PROFILES_S = 5.5   # longer than the detector's 5 s window, so profiles never share one
ALERT_WAIT_AFTER_TRAFFIC_S = 20.0
ATTACK_CHUNK_S = 2.0
KEEP_ATTACKING_AFTER_ALERT_S = 6.0

LEG_NAMES = {
    "evalue": "per-key sequential test (e-values)",
    "cluster": "campaign correlation",
    "fleet_coverage": "fleet specialization gap",
}

STEPS = [
    {
        "key": "start", "kind": "setup",
        "title": "Start from a clean slate",
        "what": ("Starts a fresh detector process, empties the traffic log and clears every "
                 "throttle on the inference API."),
        "watch": "The KPI strip at the top of the console: zero active keys and zero open incidents.",
        "during": ["overview", ".kpi-strip"], "after": ["overview", ".kpi-strip"],
    },
    {
        "key": "normal", "kind": "benign",
        "title": "Normal traffic: the console stays green",
        "what": ("Four real customer profiles query the model one after another at their real "
                 "rates: a casual user at 0.5 queries/s, a power user at 2, a mobile app at 1.2 "
                 "and a nightly batch partner at 20, well above a static rate limit."),
        "watch": ("The Simulate tab's active keys table fills with each customer while the incident "
                  "feed stays empty."),
        "during": ["simulate", "table"], "after": ["overview", "#incidents-empty"],
    },
    {
        "key": "a1", "kind": "attack", "keys": ["A1-loud"],
        "title": "Attack 1: high-volume extraction",
        "what": ("One API key floods the model at 25 queries/s with a mix of random-noise and "
                 "natural images to map its decision boundary."),
        "watch": ("The incident feed. The alert names the detection leg and the signal that fired, "
                  "and the key is throttled automatically."),
        "during": ["overview", ".kpi-strip"], "after": ["overview", "#incidents-list .incident"],
    },
    {
        "key": "a3", "kind": "attack", "keys": ["A3-zeroday"],
        "title": "Attack 2: boundary-probing sweeps from a sealed pool",
        "what": ("One key sends many small perturbations of a few seed images, back to back. The "
                 "images come from a pool the detector's calibration never saw."),
        "watch": "A new incident from the sequential test, with its per-signal evidence chart open.",
        "during": ["overview", ".kpi-strip"], "after": ["overview", "#incidents-list .incident"],
    },
    {
        "key": "a6", "kind": "attack", "keys": list(SMART_SPLIT_KEYS),
        "title": "Attack 3: one campaign split across five keys",
        "what": ("Five keys share one extraction job. Each runs at 7 queries/s and matches the "
                 "normal class mix, so no single key stands out."),
        "watch": ("A critical incident grouping all five keys into one campaign, with the campaign "
                  "graph in its evidence."),
        "during": ["overview", ".kpi-strip"], "after": ["overview", "#incidents-list .incident"],
    },
    {
        "key": "a7", "kind": "attack", "keys": list(ADAPTIVE_SPLIT_KEYS),
        "title": "Attack 4: an adaptive split built to avoid grouping",
        "what": ("Five keys again, but each asks only about its own two classes, so the keys do not "
                 "resemble one another. Pooled together, they still cover every class."),
        "watch": ("A critical incident from the fleet specialization gap, comparing pooled entropy "
                  "with each key's own."),
        "during": ["overview", ".kpi-strip"], "after": ["overview", "#incidents-list .incident"],
    },
    {
        "key": "explore", "kind": "info",
        "title": "Explore the console",
        "what": ("Every number in this tour came from the running system. Key Inventory shows each "
                 "key's status and history, Detection Policy shows the live calibrated thresholds, "
                 "and the Simulate tab replays any profile or attack on demand."),
        "watch": "Detection Policy: the three legs and the thresholds they fired against.",
        "during": ["policy", "#policy-legs"], "after": ["policy", "#policy-legs"],
    },
]
STEP_BY_KEY = {s["key"]: s for s in STEPS}

HOW = {
    "start": ("The detector runs as its own process beside the inference API. The API never depends on "
              "it, so the model keeps serving customers even if detection is stopped."),
    "normal": ("Every key is scored against its own calibrated baseline, built from normal traffic only. A "
               "busy customer is judged against its own normal rate, never against one global limit."),
    "a1": ("Every second the detector turns the key's last five seconds of traffic into statistical "
           "evidence and accumulates it. When the evidence passes a threshold of 1,000, set by theory "
           "rather than tuned on traffic, the key is throttled."),
    "a3": ("Back-to-back near-identical inputs push the distance between consecutive queries far below this "
           "key's normal, and the same sequential test accumulates that evidence until the alert fires."),
    "a6": ("Campaign correlation links keys whose queries look alike and adds up their evidence, so five "
           "individually quiet keys are scored as the single campaign they are."),
    "a7": ("The fleet specialization gap compares how broad the whole fleet's queries are with how narrow "
           "each key's own queries are. Narrow keys that together cover every class are the signature of a "
           "split extraction."),
    "explore": ("Every decision the detector made is visible in the console, from the evidence behind each "
                "incident to the thresholds it fired against, so an operator never needs the source code "
                "to understand an alert."),
}

app = FastAPI(title="SENTRY guided walkthrough")

_lock = threading.Lock()
_state = {"running": None, "phase": "idle", "detail": "", "started_at": None}
_results = {}
_detector = None
_children = []


def _set(**kw):
    with _lock:
        _state.update(kw)


# --- detector process ------------------------------------------------------------
def _other_detectors():
    try:
        out = subprocess.run(["pgrep", "-f", "sentry.detect.detector"],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    own = _detector.pid if _detector is not None and _detector.poll() is None else None
    return [int(p) for p in out.stdout.split() if int(p) != own]


def _start_detector():
    global _detector
    if _detector is not None and _detector.poll() is None:
        _detector.terminate()
        try:
            _detector.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _detector.kill()
    _detector = subprocess.Popen([sys.executable, "-m", "sentry.detect.detector"], cwd=str(ROOT))
    time.sleep(2.0)


def _ensure_detector():
    if _detector is not None and _detector.poll() is None:
        return
    if _other_detectors():
        return
    _start_detector()


# --- reading back what the system logged ------------------------------------------
def _alerts_since(t0):
    return [r for r in console._read_jsonl_tail(console.ALERTS_LOG, 500) if r.get("ts", 0) >= t0]


def _queries_since(t0):
    records = console._read_jsonl_tail(console.TRAFFIC_LOG, console.MAX_TRAFFIC_LINES_READ)
    return sum(1 for r in records if r.get("ts", 0) >= t0)


def _alert_keys(rec):
    return set(rec.get("keys") or [rec.get("key")])


def _describe(rec, t0):
    kind = rec.get("kind")
    after = max(0.0, rec.get("ts", t0) - t0)
    keys = sorted(_alert_keys(rec))
    leg = LEG_NAMES.get(kind, kind)
    if kind == "evalue":
        return (f"Detected {after:.1f} s after the attack started by the {leg}. Top signal: "
                f"{rec.get('signal')}. Evidence {rec.get('fused_score', 0):,.0f} against a threshold of "
                f"{rec.get('threshold', 0):,.0f}. Key {keys[0]} throttled.")
    if kind == "cluster":
        return (f"Detected {after:.1f} s after the attack started by {leg}, which grouped {len(keys)} "
                f"keys into one campaign (summed score {rec.get('fused_score', 0):.2f}, threshold "
                f"{rec.get('threshold', 0):.2f}). All {len(keys)} keys throttled.")
    return (f"Detected {after:.1f} s after the attack started by the {leg} across {len(keys)} keys "
            f"(gap {rec.get('fused_score', 0):.2f}, threshold {rec.get('threshold', 0):.2f}). "
            f"All {len(keys)} keys throttled.")


# --- steps ---------------------------------------------------------------------------
def _start_clean():
    others = _other_detectors()
    if others:
        pids = ", ".join(str(p) for p in others)
        return False, (f"Another detector process is already running (PID {pids}). Stop it with "
                       f"Ctrl+C in its terminal, then run this step again so the tour starts from a "
                       f"fresh detector.")
    reset_server()
    with console._status_lock:
        console._alert_status.clear()
    _start_detector()
    return True, "Fresh detector process running, traffic log empty, all throttles cleared."


def _normal_traffic():
    rng = np.random.RandomState()
    t0 = time.time()
    for i, fn in enumerate(benign.PERSONAS):
        key, images, qps = fn(rng, pool="B_eval")
        images = images[:PROFILE_QUERY_CAP.get(key, len(images))]
        _set(detail=f"{key}: {len(images)} queries at {qps:g} queries/s")
        run_episode(key, images, qps)
        if i < len(benign.PERSONAS) - 1:
            _set(detail=f"{key} finished")
            time.sleep(GAP_BETWEEN_PROFILES_S)
    _set(detail="Detector evaluating the final windows")
    time.sleep(GAP_BETWEEN_PROFILES_S)
    alerts = _alerts_since(t0)
    n = _queries_since(t0)
    if not alerts:
        return True, (f"{n} real queries from four customer profiles, including "
                      f"{PROFILE_QUERY_CAP['P4-batch']} from the batch partner at 20 queries/s. 0 alerts.")
    return False, (f"{n} real queries from four customer profiles. {len(alerts)} alert(s) logged; "
                   f"see the incident feed for the evidence.")


ATTACK_GENERATORS = {
    "a1": lambda rng: [attacks.a1_loud(rng)],
    "a3": lambda rng: [attacks.a3_zeroday(rng)],
    "a6": attacks.a6_smart_split,
    "a7": attacks.a7_adaptive_decorrelated,
}


def _attack(step):
    episodes = ATTACK_GENERATORS[step["key"]](np.random.RandomState())
    targets = set(step["keys"])
    t0 = time.time()
    n_keys = len(episodes)
    _set(detail=f"{n_keys} attacker key{'s' if n_keys > 1 else ''} sending traffic", started_at=t0)

    offsets = [0] * n_keys
    found, detected_at = None, None
    while True:
        chunk = []
        for i, (key, images, qps) in enumerate(episodes):
            n = max(1, int(round(qps * ATTACK_CHUNK_S)))
            part = images[offsets[i]:offsets[i] + n]
            offsets[i] += n
            if len(part):
                chunk.append((key, part, qps))
        if not chunk:
            break
        run_concurrent(chunk)
        if found is None:
            for rec in _alerts_since(t0):
                if _alert_keys(rec) & targets:
                    found, detected_at = rec, time.time()
                    _set(phase="detected", detail=_describe(rec, t0))
                    break
        elif time.time() - detected_at >= KEEP_ATTACKING_AFTER_ALERT_S:
            break

    deadline = time.time() + ALERT_WAIT_AFTER_TRAFFIC_S
    while found is None and time.time() < deadline:
        for rec in _alerts_since(t0):
            if _alert_keys(rec) & targets:
                found = rec
                break
        time.sleep(0.5)

    if found is None:
        return False, "No alert was logged for these keys. Check that the detector process is running."
    return True, f"{_describe(found, t0)} Only {_queries_since(t0):,} attacker queries were answered."


def _run(step):
    _set(running=step["key"], phase="running", detail="", started_at=time.time())
    try:
        if step["kind"] == "setup":
            ok, text = _start_clean()
        elif step["kind"] == "benign":
            _ensure_detector()
            ok, text = _normal_traffic()
        elif step["kind"] == "attack":
            _ensure_detector()
            ok, text = _attack(step)
        else:
            ok, text = True, ("Tour complete. Replay any step, or use the Simulate tab to launch traffic "
                              "yourself.")
    except Exception as e:
        ok, text = False, f"Step stopped: {e}"
    with _lock:
        _results[step["key"]] = {"ok": ok, "text": text, "at": time.time()}
        _state.update(running=None, phase="done", detail="")


# --- routes --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return TEMPLATE_PATH.read_text()


@app.get("/console", response_class=HTMLResponse)
def console_page():
    return console.index()


@app.get("/tour/api/steps")
def tour_steps():
    fields = ("key", "kind", "title", "what", "watch", "during", "after")
    return [{**{f: s[f] for f in fields}, "how": HOW.get(s["key"], "")} for s in STEPS]


@app.post("/tour/api/run/{key}")
def tour_run(key: str):
    step = STEP_BY_KEY.get(key)
    if step is None:
        return JSONResponse({"error": "unknown step"}, status_code=404)
    with _lock:
        if _state["running"]:
            return JSONResponse({"error": "a step is already running"}, status_code=409)
        _state["running"] = key
    threading.Thread(target=_run, args=(step,), daemon=True).start()
    return {"status": "started", "step": key}


@app.get("/tour/api/status")
def tour_status():
    with _lock:
        s = dict(_state)
        results = dict(_results)
    s["elapsed_s"] = round(time.time() - s["started_at"], 1) if s["running"] and s["started_at"] else None
    s["results"] = results
    s["api_up"] = is_alive()
    return s


app.mount("/", console.app)


def _stop_children():
    for proc in _children + [_detector]:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    import uvicorn
    atexit.register(_stop_children)
    if not is_alive():
        print("[walkthrough] starting the inference API...")
        _children.append(subprocess.Popen([sys.executable, "-m", "sentry.api.server"], cwd=str(ROOT)))
        for _ in range(150):
            if is_alive():
                break
            time.sleep(0.2)
        else:
            print("[walkthrough] the inference API did not come up; run `make setup` first")
            sys.exit(1)
    print(f"[walkthrough] open http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
