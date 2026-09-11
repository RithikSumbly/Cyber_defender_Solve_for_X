"""Sends episodes to the live API at a target rate, with real wall-clock
timing, so 'time to alert' and rate-based signals reflect genuine elapsed
time; nothing is pre-computed or simulated after the fact."""
import time
import random
from concurrent.futures import ThreadPoolExecutor

from sentry.traffic.client import send_query


def run_single_key(key: str, images, qps: float, jitter: float = 0.15):
    interval = 1.0 / qps
    results = []
    t_start = time.time()
    for img in images:
        t0 = time.time()
        r = send_query(key, img)
        r["episode_t"] = t0 - t_start
        results.append(r)
        sleep_for = interval * (1.0 + random.uniform(-jitter, jitter))
        elapsed = time.time() - t0
        if sleep_for > elapsed:
            time.sleep(sleep_for - elapsed)
    return key, results


def run_concurrent(key_images_qps: list):
    """key_images_qps: list of (key, images, qps). Runs all keys in
    parallel threads so split-key attacks (A6) actually overlap in time,
    exactly as a real distributed attacker would."""
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, len(key_images_qps))) as ex:
        futures = [ex.submit(run_single_key, k, imgs, q)
                   for k, imgs, q in key_images_qps]
        for fut in futures:
            key, res = fut.result()
            results[key] = res
    return results


def run_episode(key: str, images, qps: float):
    return run_concurrent([(key, images, qps)])
