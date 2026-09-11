"""Benign traffic personas.

Each persona function returns (key, images[N,32,32,3] uint8, qps) for one
episode. `pool` controls which slice of the data ledger is used, so the
same persona code path is reused for calibration traffic (B_cal) and for
held-out reporting traffic (B_eval) without ever mixing the two.
"""
import numpy as np
from sentry.traffic.data_pool import sample, BENIGN_CLASS_PRIOR

UNIFORM_PRIOR = [0.1] * 10


def p1_casual(rng: np.random.RandomState, pool: str):
    n = rng.randint(15, 25)
    images, _ = sample(pool, n, rng, class_prior=UNIFORM_PRIOR)
    return "P1-casual", images, 0.5


def p2_power(rng: np.random.RandomState, pool: str):
    n = rng.randint(50, 70)
    images, _ = sample(pool, n, rng, class_prior=UNIFORM_PRIOR)
    return "P2-power", images, 2.0


def p3_mobile(rng: np.random.RandomState, pool: str):
    n = rng.randint(25, 40)
    images, _ = sample(pool, n, rng, class_prior=BENIGN_CLASS_PRIOR)
    return "P3-mobile", images, 1.2  # bursty pattern applied by the runner


def p4_batch(rng: np.random.RandomState, pool: str):
    """The nightly batch partner: high, sustained, perfectly legitimate
    volume. Any detector that fires on rate alone bans this customer."""
    n = rng.randint(150, 220)
    images, _ = sample(pool, n, rng, class_prior=UNIFORM_PRIOR)
    return "P4-batch", images, 20.0  # scaled down from 200 qps for a laptop demo


PERSONAS = [p1_casual, p2_power, p3_mobile, p4_batch]
