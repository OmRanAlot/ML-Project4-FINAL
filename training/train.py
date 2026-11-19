"""
Train the explainer's MLP on two moons and export training snapshots.

Architecture (exactly the one in the assignment):
    input(2) -> [128 ReLU -> 2 linear] x 3 -> out (logistic regression on the last 2D space)

Design choices that make the learned transformation GRADUAL (professor's feedback):
  1. Identity-like initialization: each 2 -> 128 -> 2 block starts out (approximately)
     as the identity map, solved by least squares over random ReLU features. At step 0
     every 2D representation looks like the input, so training visibly "bends" it.
  2. Depth-smoothness penalty: lambda * sum_k mean ||h_k - h_{k-1}||^2. A sum of squared
     step sizes is minimized by splitting the work evenly across the three blocks, so
     no single block does all of the untangling.
  3. Small-step, full-batch Adam, and snapshots taken by *visual displacement* rather than
     by a fixed step interval, so consecutive frames differ by a similar amount.

The browser recomputes every forward pass from the exported weights, which lets the site
warp grids, follow user-placed points, and draw decision boundaries at any snapshot.

Usage:  python training/train.py   (writes public/data/network.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

SEED = 7
N_POINTS = 400
NOISE = 0.1
HIDDEN = 128
N_BLOCKS = 3
LEARNING_RATE = 2e-3
MAX_STEPS = 1200
SMOOTH_LAMBDA = 0.08
SNAPSHOT_DISPLACEMENT = 0.02  # mean point displacement (std units) that triggers a frame
MAX_FRAMES = 100
OUT_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "network.json"


def make_data() -> tuple[np.ndarray, np.ndarray]:
    x, y = make_moons(n_samples=N_POINTS, noise=NOISE, random_state=SEED)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float32), y.astype(np.float32)


class Block(nn.Module):
    def __init__(self, use_relu: bool) -> None:
        super().__init__()
        self.up = nn.Linear(2, HIDDEN)
        self.down = nn.Linear(HIDDEN, 2)
        self.use_relu = use_relu

    def hidden(self, h: torch.Tensor) -> torch.Tensor:
        z = self.up(h)
        return torch.relu(z) if self.use_relu else z

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.down(self.hidden(h))


class Net(nn.Module):
    def __init__(self, use_relu: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(use_relu) for _ in range(N_BLOCKS)])
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        reps = [x]
        h = x
        for block in self.blocks:
            h = block(h)
            reps.append(h)
        return self.head(h).squeeze(-1), reps


def identity_init(net: Net, rng: np.random.Generator) -> None:
    """Make every block start as (approximately) the identity on [-3, 3]^2."""
    probe = rng.uniform(-3.2, 3.2, size=(4000, 2)).astype(np.float32)
    with torch.no_grad():
        for block in net.blocks:
            angles = rng.uniform(0, 2 * np.pi, HIDDEN)
            w1 = np.stack([np.cos(angles), np.sin(angles)], axis=1) * rng.uniform(0.6, 1.4, (HIDDEN, 1))
            b1 = rng.uniform(-2.5, 2.5, HIDDEN)
            block.up.weight.copy_(torch.tensor(w1, dtype=torch.float32))
            block.up.bias.copy_(torch.tensor(b1, dtype=torch.float32))
            feats = block.hidden(torch.tensor(probe)).numpy()
            design = np.hstack([feats, np.ones((len(probe), 1), dtype=np.float32)])
            ridge = 1e-2 * np.eye(design.shape[1])
            sol = np.linalg.solve(design.T @ design + ridge, design.T @ probe)
            block.down.weight.copy_(torch.tensor(sol[:-1].T, dtype=torch.float32))
            block.down.bias.copy_(torch.tensor(sol[-1], dtype=torch.float32))
        net.head.weight.copy_(torch.tensor([[0.3, -0.3]]))
        net.head.bias.zero_()


def fit_probe(rep: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    """Fit a logistic-regression probe; return its accuracy and its line [a, b, c] (a*x + b*y + c = 0)."""
    clf = LogisticRegression(C=1e4, max_iter=2000)
    clf.fit(rep, y)
    a, b = clf.coef_[0]
    line = [float(a), float(b), float(clf.intercept_[0])]
    return float(clf.score(rep, y)), [round(v, 5) for v in line]


def round_list(arr: np.ndarray, digits: int = 4) -> list:
    return np.round(arr.astype(np.float64), digits).tolist()


def export_params(net: Net) -> dict:
    blocks = [
        {
            "W1": round_list(b.up.weight.detach().numpy()),    # 128 x 2
            "b1": round_list(b.up.bias.detach().numpy()),      # 128
            "W2": round_list(b.down.weight.detach().numpy()),  # 2 x 128
            "b2": round_list(b.down.bias.detach().numpy()),    # 2
        }
        for b in net.blocks
    ]
    return {
        "blocks": blocks,
        "head": {
            "w": round_list(net.head.weight.detach().numpy()[0]),
            "b": float(round(net.head.bias.item(), 4)),
        },
    }


def train(use_relu: bool, x: np.ndarray, y: np.ndarray) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = Net(use_relu)
    identity_init(net, rng)
    xt, yt = torch.tensor(x), torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()

    frames: list[dict] = []
    last_reps: list[np.ndarray] = []

    def snapshot(step: int, loss: float) -> None:
        nonlocal last_reps
        with torch.no_grad():
            logits, reps = net(xt)
        reps_np = [r.numpy() for r in reps]
        acc = float(((logits > 0).float() == yt).float().mean())
        probes = [fit_probe(r, y) for r in reps_np]
        frames.append({
            "step": step,
            "loss": round(loss, 5),
            "accuracy": round(acc, 4),
            "probe": [round(acc, 4) for acc, _ in probes],
            "probeLines": [line for _, line in probes],
            "params": export_params(net),
        })
        last_reps = reps_np

    for step in range(MAX_STEPS + 1):
        logits, reps = net(xt)
        task = bce(logits, yt)
        smooth = sum(((reps[k] - reps[k - 1]) ** 2).sum(dim=1).mean() for k in range(1, len(reps)))
        loss = task + SMOOTH_LAMBDA * smooth

        if step == 0:
            snapshot(0, float(task.detach()))
        else:
            moved = max(
                float(np.linalg.norm(r.detach().numpy() - prev, axis=1).mean())
                for r, prev in zip(reps[1:], last_reps[1:])
            )
            if moved > SNAPSHOT_DISPLACEMENT or step == MAX_STEPS:
                snapshot(step, float(task.detach()))

        opt.zero_grad()
        loss.backward()
        opt.step()

    if len(frames) > MAX_FRAMES:
        keep = np.unique(np.round(np.linspace(0, len(frames) - 1, MAX_FRAMES)).astype(int))
        frames = [frames[i] for i in keep]
    final = frames[-1]
    print(f"relu={use_relu}: frames={len(frames)} final acc={final['accuracy']} probe={final['probe']}")
    return {"useRelu": use_relu, "frames": frames}


def main() -> None:
    x, y = make_data()
    payload = {
        "meta": {
            "architecture": "2 -> [128 ReLU -> 2 linear] x3 -> logistic",
            "seed": SEED, "noise": NOISE, "learningRate": LEARNING_RATE,
            "smoothLambda": SMOOTH_LAMBDA, "hidden": HIDDEN,
        },
        "data": {"x": round_list(x), "y": y.astype(int).tolist()},
        "relu": train(True, x, y),
        "linear": train(False, x, y),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
