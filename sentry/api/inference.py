"""Victim model loading + single-image inference helper."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from sentry.model.arch import VictimCNN

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "sentry" / "model" / "victim.pt"

_device = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
_model = None


def load_model():
    global _model
    if _model is None:
        m = VictimCNN()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        m.to(_device)
        m.eval()
        _model = m
    return _model


@torch.no_grad()
def predict(img_chw01: np.ndarray):
    """img_chw01: float32 array, shape (3, 32, 32), values in [0, 1]."""
    model = load_model()
    x = torch.from_numpy(img_chw01).unsqueeze(0).to(_device)
    logits, emb = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    emb = emb.cpu().numpy()[0]
    class_idx = int(probs.argmax())
    confidence = float(probs[class_idx])
    return {
        "class_idx": class_idx,
        "class_name": VictimCNN.CLASSES[class_idx],
        "confidence": confidence,
        "probs": probs,
        "embedding": emb,
    }
