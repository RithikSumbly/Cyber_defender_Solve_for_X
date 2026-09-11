"""Sliding-window construction over per-key traffic-log records."""
import json
from pathlib import Path
from collections import defaultdict

WINDOW_SECONDS = 5.0
STEP_SECONDS = 1.0


def load_log(path: Path) -> dict:
    """Returns {key: [records sorted by ts]}."""
    by_key = defaultdict(list)
    if not Path(path).exists():
        return by_key
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_key[rec["key"]].append(rec)
    for k in by_key:
        by_key[k].sort(key=lambda r: r["ts"])
    return by_key


def sliding_windows(records: list, window_s=WINDOW_SECONDS, step_s=STEP_SECONDS):
    """Yields (window_end_ts, window_records) for a single key's sorted
    record list, stepping through the whole episode."""
    if not records:
        return
    t_start = records[0]["ts"]
    t_end = records[-1]["ts"]
    t = t_start + window_s
    while t <= t_end + step_s:
        win = [r for r in records if t - window_s < r["ts"] <= t]
        if win:
            yield t, win
        t += step_s
