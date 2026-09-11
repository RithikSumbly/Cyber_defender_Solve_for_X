"""API key registry.

`ground_truth` exists ONLY for the eval harness to compute TPR/FPR after
the fact. The live API and the live detector never read `ground_truth`;
they only ever see a bare key string, exactly like a real gateway would.
Baking ground truth into the detector's decision path would be label
leakage and would invalidate every number in results/metrics.json.
"""

KEYS = {
    # --- benign personas -------------------------------------------------
    "P1-casual":     {"label": "Casual browser",        "ground_truth": "benign"},
    "P2-power":      {"label": "Power user",             "ground_truth": "benign"},
    "P3-mobile":     {"label": "Mobile app",    "ground_truth": "benign"},
    "P4-batch":      {"label": "Nightly batch partner",  "ground_truth": "benign"},

    # --- attacker: naive / loud (A1) --------------------------------------
    "A1-loud":       {"label": "Naive extractor",        "ground_truth": "attack"},

    # --- attacker: sealed zero-day (A3), never used to tune the detector --
    "A3-zeroday":    {"label": "Boundary-probing attacker (sealed pool)",      "ground_truth": "attack"},

    # --- attacker: smart / split across keys (A6) ------------------------
    "A6-smart-01":   {"label": "Smart split attacker #1", "ground_truth": "attack"},
    "A6-smart-02":   {"label": "Smart split attacker #2", "ground_truth": "attack"},
    "A6-smart-03":   {"label": "Smart split attacker #3", "ground_truth": "attack"},
    "A6-smart-04":   {"label": "Smart split attacker #4", "ground_truth": "attack"},
    "A6-smart-05":   {"label": "Smart split attacker #5", "ground_truth": "attack"},


    # --- attacker: adaptive (A7), has read our source code, deliberately
    # decorrelates its 5 keys' feature centroids to evade centroid-similarity
    # campaign correlation -----------------------------------------------
    "A7-adaptive-01": {"label": "Adaptive decorrelated attacker #1", "ground_truth": "attack"},
    "A7-adaptive-02": {"label": "Adaptive decorrelated attacker #2", "ground_truth": "attack"},
    "A7-adaptive-03": {"label": "Adaptive decorrelated attacker #3", "ground_truth": "attack"},
    "A7-adaptive-04": {"label": "Adaptive decorrelated attacker #4", "ground_truth": "attack"},
    "A7-adaptive-05": {"label": "Adaptive decorrelated attacker #5", "ground_truth": "attack"},
}

ADAPTIVE_SPLIT_KEYS = [k for k in KEYS if k.startswith("A7-adaptive-")]

BENIGN_KEYS = [k for k, v in KEYS.items() if v["ground_truth"] == "benign"]
ATTACK_KEYS = [k for k, v in KEYS.items() if v["ground_truth"] == "attack"]
SMART_SPLIT_KEYS = [k for k in ATTACK_KEYS if k.startswith("A6-smart")]


def is_valid_key(key: str) -> bool:
    return key in KEYS


def ground_truth(key: str) -> str:
    return KEYS.get(key, {}).get("ground_truth", "unknown")
