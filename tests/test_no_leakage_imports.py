"""Structural leakage firewall: statically verifies that nothing under
sentry/detect or sentry/api (the actual deployed detection/serving code)
imports from sentry/traffic/attacks.py or sentry/eval/fidelity.py, the
places attack traffic and substitute-model training live.

This makes "no threshold is tuned on attack data" a statically enforced
property, not just a convention: calibrate.py draws only on benign.py, and
if detection or serving code ever imports attacks.py, this test fails on
the next run, protecting every FPR/TPR number in the report.

Run:  python -m pytest tests/test_no_leakage_imports.py -q
  or: python tests/test_no_leakage_imports.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUBSTRINGS = ("sentry.traffic.attacks", "sentry.eval.fidelity",
                         "sentry.eval.substitute")

# Files whose whole JOB is to touch attack traffic / substitute training:
# excluded because the firewall is about the DETECTOR and API, not the eval
# harness itself (which legitimately needs both attacks and calibration to
# compare them against each other).
ALLOWED_DIRS = {"eval", "traffic", "report"}


def iter_forbidden_source_files():
    for base in ["sentry/detect", "sentry/api", "sentry/model"]:
        d = ROOT / base
        if not d.exists():
            continue
        for path in d.rglob("*.py"):
            yield path


def imports_forbidden(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            if name and any(f in name for f in FORBIDDEN_SUBSTRINGS):
                hits.append(name)
    return hits


def test_detector_and_api_never_import_attack_or_substitute_code():
    violations = {}
    for path in iter_forbidden_source_files():
        hits = imports_forbidden(path)
        if hits:
            violations[str(path.relative_to(ROOT))] = hits
    assert not violations, (
        "Leakage firewall violated: detector/API code imports attack or "
        f"substitute-training code, which can invalidate every calibrated "
        f"threshold: {violations}"
    )


if __name__ == "__main__":
    try:
        test_detector_and_api_never_import_attack_or_substitute_code()
        print("[firewall] PASS: no forbidden imports found in sentry/detect, "
              "sentry/api, sentry/model")
        sys.exit(0)
    except AssertionError as e:
        print(f"[firewall] FAIL: {e}")
        sys.exit(1)
