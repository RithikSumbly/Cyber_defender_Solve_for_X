"""Comparison baselines evaluated on the exact same logged traffic as Sentry
(no separate traffic generation, same episodes, same data, fair comparison).

  static_rate   - a flat, uncalibrated qps cutoff. What most teams ship first.
  distance_only - single-signal near-duplicate detector (a simplified,
                  single-signal reproduction of the PRADA (2019) idea:
                  extraction via boundary-probing produces near-duplicate
                  consecutive queries).
  mmd_style     - single-signal distributional-shift detector (a simplified
                  proxy for a fixed-window MMD two-sample test against the
                  benign reference distribution).

Methodology: each baseline is a simplified, single-signal reproduction of
the published *idea*, evaluated on the same logged traffic as Sentry,
rather than a line-for-line implementation of the cited paper.
"""
import numpy as np


def static_rate_fires(raw: dict, calib: dict) -> bool:
    return raw["rate"] > calib["static_rate_limit_qps"]


def distance_only_fires(raw: dict, calib: dict) -> bool:
    d = raw.get("near_duplicate")
    if d is None:
        return False
    return d < calib["distance_only_threshold"]


def mmd_style_fires(window_centroid: np.ndarray, calib: dict) -> bool:
    ref = np.asarray(calib["cal_global_feat_mean"])
    stat = float(np.sum((window_centroid - ref) ** 2))
    return stat > calib["mmd_threshold"]


BASELINES = {
    "static_rate": static_rate_fires,
    "distance_only": distance_only_fires,
    "mmd_style": mmd_style_fires,
}
