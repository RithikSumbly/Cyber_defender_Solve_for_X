"""Small stats helpers: Wilson score intervals for binomial rates."""
import math


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Returns (point_estimate, lower, upper): Wilson score 95% CI by default."""
    if n == 0:
        return 0.0, 0.0, 1.0
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return phat, lower, upper


def fmt_rate_ci(successes: int, n: int) -> str:
    p, lo, hi = wilson_ci(successes, n)
    return f"{p * 100:.1f}% [{lo * 100:.1f}-{hi * 100:.1f}%] (n={n})"
