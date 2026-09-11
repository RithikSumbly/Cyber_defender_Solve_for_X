"""Attack traffic profiles.

A1  naive/loud       - high qps, single key, half random noise + half
                        class-balanced natural images (aggressive boundary
                        probing, high label diversity). Easiest to catch.

A3  zero-day (sealed) - near-duplicate perturbation sweeps: take one seed
                        image, query many small Gaussian-noise variants of
                        it. This pattern is never shown to the calibration
                        step; it is used ONLY at final eval to test whether
                        the fused detector generalises to an attack family
                        it was not designed against.

A6  smart / split     - natural images resampled to the BENIGN class prior,
                        split across 5 keys, each individually running at
                        a power-user-like rate. No single key looks
                        suspicious; only cross-key correlation catches it.

A7  adaptive / split  - same rate and volume as A6, but each of the 5 keys
                        is confined to a disjoint 2-class subset, built to
                        evade centroid-similarity campaign correlation. The
                        fleet specialization-gap leg is designed to catch it.
"""
import numpy as np
from sentry.traffic.data_pool import sample, random_noise_images, BENIGN_CLASS_PRIOR
from sentry.api.keys import SMART_SPLIT_KEYS, ADAPTIVE_SPLIT_KEYS

UNIFORM_PRIOR = [0.1] * 10


def a1_loud(rng: np.random.RandomState, pool: str = "train"):
    n = rng.randint(600, 900)
    n_noise = n // 2
    n_natural = n - n_noise
    noise = random_noise_images(n_noise, rng)
    natural, _ = sample(pool, n_natural, rng, class_prior=UNIFORM_PRIOR)
    images = np.concatenate([noise, natural], axis=0)
    rng.shuffle(images)
    return "A1-loud", images, 25.0


def a3_zeroday(rng: np.random.RandomState, pool: str = "A_holdout"):
    seeds, _ = sample(pool, 6, rng, class_prior=UNIFORM_PRIOR)
    n_per_seed = rng.randint(45, 60)
    all_imgs = []
    for seed_img in seeds:
        base = seed_img.astype(np.float32)
        for _ in range(n_per_seed):
            noise = rng.normal(0, 12, size=base.shape)
            variant = np.clip(base + noise, 0, 255).astype(np.uint8)
            all_imgs.append(variant)
    # Deliberately NOT shuffled: this attack's whole signature is
    # near-duplicate CONSECUTIVE queries around each seed image. Shuffling
    # across seeds would scatter same-seed variants apart in the sequence
    # and erase the exact pattern the near-duplicate-distance signal is
    # designed to catch.
    images = np.stack(all_imgs, axis=0)
    return "A3-zeroday", images, 8.0


def a6_smart_split(rng: np.random.RandomState, pool: str = "train"):
    """Returns a list of (key, images, qps) tuples, one per split key,
    meant to be sent CONCURRENTLY within a single episode."""
    episodes = []
    for key in SMART_SPLIT_KEYS:
        n = rng.randint(220, 260)
        images, _ = sample(pool, n, rng, class_prior=BENIGN_CLASS_PRIOR)
        episodes.append((key, images, 7.0))  # power-user-like rate per key,
                                              # individually unremarkable;
                                              # five of them running at once
                                              # is the pattern campaign
                                              # correlation is designed to catch
    return episodes


def a7_adaptive_decorrelated(rng: np.random.RandomState, pool: str = "train"):
    """Adaptive attacker who has read SENTRY's source and targets campaign
    correlation's mechanism (cosine similarity of feature centroids,
    threshold 0.80). Instead of all 5 keys drawing from the SAME benign
    class prior (which is what groups A6's keys together), each key
    specializes in a DISJOINT 2-class subset of CIFAR-10. Rate and volume
    match A6; the difference is that each key's content is deliberately
    decorrelated so the keys' centroids diverge, a design built to evade
    centroid-similarity campaign correlation. Individually each key looks
    like a narrow specialist, while the fleet pooled together covers all
    ten classes: the pattern SENTRY's fleet specialization-gap leg
    (sentry/detect/detector.py) is designed to catch.
    """
    class_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    episodes = []
    for key, (c1, c2) in zip(ADAPTIVE_SPLIT_KEYS, class_pairs):
        prior = [0.0] * 10
        prior[c1] = 0.5
        prior[c2] = 0.5
        n = rng.randint(220, 260)
        images, _ = sample(pool, n, rng, class_prior=prior)
        episodes.append((key, images, 7.0))
    return episodes


ATTACK_SINGLE = [a1_loud, a3_zeroday]  # each returns one (key, images, qps)
