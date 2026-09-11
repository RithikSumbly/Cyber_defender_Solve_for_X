"""Train the victim model and freeze the data ledger.

Data ledger (never violated anywhere else in this repo):
  V_train      CIFAR-10 train (50000)         -> victim model weights only
  B_cal        CIFAR-10 test[0:5000]          -> benign traffic used to CALIBRATE
                                                  detector thresholds. Never used
                                                  to report FPR.
  B_eval       CIFAR-10 test[5000:8000]       -> benign traffic used ONLY to
                                                  report FPR. Never used for fitting.
  A_holdout    CIFAR-10 test[8000:10000]      -> image pool for the sealed
                                                  zero-day attack (A3). Not
                                                  touched until final eval.

Run:  python -m sentry.model.train_victim
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from sentry.model.arch import VictimCNN

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "sentry" / "model" / "victim.pt"
LEDGER_PATH = RESULTS_DIR / "data_ledger.json"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return DEVICE


def load_cifar():
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True,
                                          download=True, transform=tfm)
    test = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False,
                                         download=True, transform=tfm)
    return train, test


def build_ledger(test_ds):
    n = len(test_ds)
    assert n == 10000, f"expected CIFAR-10 test set of 10000, got {n}"
    ledger = {
        "B_cal": list(range(0, 5000)),
        "B_eval": list(range(5000, 8000)),
        "A_holdout": list(range(8000, 10000)),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)
    return ledger


def train(epochs=8, batch_size=128, lr=1e-3):
    print(f"[train] device={DEVICE}")
    train_ds, test_ds = load_cifar()
    ledger = build_ledger(test_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0)
    cal_loader = DataLoader(Subset(test_ds, ledger["B_cal"]), batch_size=256)

    model = VictimCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        avg_loss = running / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in cal_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"[train] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
              f"B_cal_acc={acc:.4f} elapsed={time.time() - t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[train] saved victim model -> {MODEL_PATH}")

    with open(RESULTS_DIR / "victim_train_log.json", "w") as f:
        json.dump({"epochs": epochs, "final_B_cal_acc": acc,
                    "device": str(DEVICE),
                    "train_seconds": time.time() - t0}, f, indent=2)


if __name__ == "__main__":
    train()
