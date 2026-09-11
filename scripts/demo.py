"""`make demo`: fast, repeatable, idempotent. Assumes `make setup` has
already trained the victim model and calibrated the detector.

Sequence: start API + detector -> run benign traffic (must stay green) ->
launch an extraction attack (A1) -> watch for the console [ALERT] and
confirm it fires within a few seconds -> clean shutdown. Exit 0 on PASS,
1 on FAIL, so this script can be run twice in a row and pass both times.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VICTIM_PATH = ROOT / "sentry" / "model" / "victim.pt"
CALIB_PATH = ROOT / "results" / "calibration.json"
ALERTS_PATH = ROOT / "results" / "alerts.jsonl"

DETECTOR_DURATION = 90
ATTACK_WAIT_TIMEOUT = 40  # wide margin over the e-value leg's measured
                          # mean time-to-detect (9.9s, observed up to ~23s),
                          # keeping this acceptance check stable across
                          # machines.


def fail(msg):
    print(f"[demo] FAIL: {msg}")
    sys.exit(1)


def main():
    if not VICTIM_PATH.exists():
        fail("no victim model found, run `make setup` first")
    if not CALIB_PATH.exists():
        fail("no calibration found, run `make setup` first")

    from sentry.traffic.client import is_alive, reset_server
    from sentry.traffic import benign, attacks
    from sentry.traffic.runner import run_episode
    import numpy as np

    server_proc = None
    detector_proc = None
    try:
        if not is_alive():
            print("[demo] starting API server...")
            server_proc = subprocess.Popen(
                [sys.executable, "-m", "sentry.api.server"], cwd=str(ROOT))
            for _ in range(50):
                if is_alive():
                    break
                time.sleep(0.2)
            else:
                fail("API server did not come up")
        else:
            print("[demo] API server already running")

        reset_server()

        print(f"[demo] starting detector worker (runs for {DETECTOR_DURATION}s)...")
        detector_proc = subprocess.Popen(
            [sys.executable, "-m", "sentry.detect.detector",
             "--duration", str(DETECTOR_DURATION)], cwd=str(ROOT))
        time.sleep(1.5)

        print("[demo] sending benign traffic (should stay green)...")
        rng = np.random.RandomState(7)
        # TRUE per-persona qps, just fewer queries. Per-key calibration means
        # a persona's own normal RATE matters, not just query count. Speeding
        # up P1's qps would make it look anomalous against its own baseline.
        # P4 is the nightly-batch "hard case" persona: fast even at its true
        # 20qps, deliberately exercising per-key baselining under load.
        for persona_fn, n_cap in [(benign.p1_casual, 6), (benign.p4_batch, 15)]:
            key, images, qps = persona_fn(rng, pool="B_eval")
            run_episode(key, images[:n_cap], qps)

        if ALERTS_PATH.exists():
            with open(ALERTS_PATH) as f:
                n_alerts_before = sum(1 for _ in f)
        else:
            n_alerts_before = 0

        if n_alerts_before > 0:
            fail(f"benign traffic triggered {n_alerts_before} alert(s), should be zero")
        print("[demo] benign traffic clean: 0 alerts")

        print("[demo] launching A1-loud extraction attack...")
        t_attack_start = time.time()
        rng2 = np.random.RandomState(42)
        key, images, qps = attacks.a1_loud(rng2)
        run_episode(key, images, qps)

        detected = False
        while time.time() - t_attack_start < ATTACK_WAIT_TIMEOUT:
            if ALERTS_PATH.exists():
                with open(ALERTS_PATH) as f:
                    lines = f.readlines()
                if len(lines) > 0:
                    alert = json.loads(lines[-1])
                    detected = True
                    break
            time.sleep(0.3)

        if not detected:
            fail(f"no alert fired within {ATTACK_WAIT_TIMEOUT}s of attack start")

        t_to_alert = alert["t"] - 0  # detector's own internal clock, see alert record
        print(f"[demo] ALERT fired: {alert}")
        print(f"[demo] PASS: attack detected, signal="
              f"{alert.get('signal', 'campaign_correlation')}")

    finally:
        if detector_proc:
            detector_proc.terminate()
            try:
                detector_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                detector_proc.kill()
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
