"""Small CNN victim model, a stand-in for a proprietary CIFAR-10 classifier
exposed behind a paid inference API."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VictimCNN(nn.Module):
    NUM_CLASSES = 10
    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, self.NUM_CLASSES)
        self.dropout = nn.Dropout(0.25)

    def embed(self, x):
        """Penultimate-layer embedding, used by the detector, not exposed
        over the API, only used internally for signal computation."""
        x = self.pool(F.relu(self.conv1(x)))   # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))   # 128x4x4
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        emb = self.embed(x)
        emb = self.dropout(emb)
        logits = self.fc2(emb)
        return logits, emb
