"""Measures how good a clone the attacker actually built, not simulated,
trained for real on the attacker's own (query image -> victim's returned
label) pairs, exactly as a real extraction attacker would.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from sentry.traffic.data_pool import pool_images
from sentry.api.inference import predict as victim_predict

REFERENCE_N = 400
_reference_cache = None


def _downsize_flat(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """32x32x3 -> 16x16x3 via simple stride subsampling, flattened. This is
    exactly the kind of cheap feature representation a real attacker (who
    only has pixels, not the victim's internal embedding) would use."""
    small = img_hwc_uint8[::2, ::2, :]
    return small.astype(np.float32).flatten() / 255.0


def get_reference_set():
    """Victim's own predictions on a fixed slice of B_eval, computed once
    directly in-process (not through the API) purely as a fidelity yardstick,
    never used to fit or threshold the detector."""
    global _reference_cache
    if _reference_cache is not None:
        return _reference_cache
    images, _ = pool_images("B_eval")
    images = images[:REFERENCE_N]
    feats = np.stack([_downsize_flat(im) for im in images])
    victim_labels = []
    for im in images:
        chw = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)
        victim_labels.append(victim_predict(chw)["class_idx"])
    _reference_cache = (feats, np.array(victim_labels))
    return _reference_cache


def substitute_fidelity(query_images: list, query_labels: list) -> float:
    """query_images: list of HWC uint8 arrays the attacker sent.
    query_labels: victim's returned class_idx for each, in the same order.
    Returns fraction of the fixed reference set where the substitute model
    (trained ONLY on those pairs) agrees with the victim's own prediction."""
    if len(set(query_labels)) < 2 or len(query_images) < 10:
        return 0.0  # not enough signal/diversity to fit a classifier yet
    X = np.stack([_downsize_flat(im) for im in query_images])
    y = np.array(query_labels)
    clf = LogisticRegression(max_iter=300, multi_class="auto")
    try:
        clf.fit(X, y)
    except ValueError:
        return 0.0
    ref_X, ref_victim_labels = get_reference_set()
    pred = clf.predict(ref_X)
    return float(np.mean(pred == ref_victim_labels))
