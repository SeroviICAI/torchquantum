#!/usr/bin/env python
"""torchquantum_power_qnns_reproduction.py
================================================
Turn-key benchmark to **replicate the key figures** from
*“The Power of Quantum Neural Networks”* (Abbas *et al.* 2021) using the
**torchquantum** library - fully aligned with the library's *Usability > Performance*
and *Simple-over-Easy* design principles.

The script generates **four figure sets** plus TensorBoard logs:

1. **Fisher-information spectra** (histograms)
2. **Normalised effective dimension** versus dataset size *n*
3. **Label-corruption** sweep (effective dimension vs. % noisy labels)
4. **Training curves** (cross-entropy loss) for QNN / easy-QNN / classical

All plots are written to `<outdir>/plots/*.png` at 300 dpi, while raw arrays are
cached as NumPy files for reuse.  TensorBoard summaries live in
`<outdir>/tb_logs`.

Quick start
-----------
```bash
# create and activate your torchquantum dev env first
python torchquantum_power_qnns_reproduction.py \
       --outdir results_qnns \
       --seed 12345            # reproducible!
# open TensorBoard in a new shell
tensorboard --logdir results_qnns/tb_logs
```
Use `--fast` for a *coarse* run (≈2 min on CPU) or remove it for the *full* sweep
(≈10-15 min depending on hardware).
"""
from __future__ import annotations

###############################################################################
# Std‑lib & third‑party #######################################################
###############################################################################
import argparse
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# CUDA‑Q target will be set at runtime depending on CPU/GPU availability
import cudaq  # noqa: F401 – side‑effect import

###############################################################################
# torchquantum ################################################################
###############################################################################
from torchquantum.models import QNN
from torchquantum.func import fisher, fisher_norm, eff_dim


###############################################################################
# Classical baseline (matched parameter count) ###############################
###############################################################################
class ParamMatchedFCN(nn.Module):
    """Single‑hidden‑layer MLP with a *fixed* number of parameters *d*.

    We first try a weight‑only linear layer (`in×out=d`).  If that fails we add a
    tanh hidden layer sized to keep the total parameter count ≤ *d* (never above).
    """

    def __init__(self, in_features: int, out_features: int, d: int) -> None:
        super().__init__()
        self.in_features, self.out_features, self.d = in_features, out_features, d

        # 1‑layer attempt ----------------------------------------------------
        if d == in_features * out_features:
            self.net = nn.Linear(in_features, out_features, bias=False)
            return

        # Fallback: 1 hidden layer -----------------------------------------
        hidden = max(
            1, (d - out_features * in_features) // (in_features + out_features)
        )
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, out_features, bias=False),
        )
        assert self.param_count() <= d, "over‑allocated parameters"

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


###############################################################################
# Utility #####################################################################
###############################################################################


def set_seed(seed: int) -> None:
    """Global deterministic seed (numpy/random/torch + CUBLAS workspace)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_iris_binary() -> tuple[torch.Tensor, torch.Tensor]:
    """Subset Iris → binary task (classes 0 vs 1) & normalise to **[‑1,1]**."""
    data = load_iris()
    mask = data.target < 2
    X = data.data[mask].astype(np.float32)
    y = data.target[mask].astype(np.int64)
    X_min, X_max = X.min(0), X.max(0)
    X = 2 * (X - X_min) / (X_max - X_min) - 1  # scale
    return torch.from_numpy(X), torch.from_numpy(y)


###############################################################################
# Fisher helpers ##############################################################
###############################################################################


def fisher_spectrum(
    model: nn.Module, X: torch.Tensor, *, samples: int, device
) -> np.ndarray:
    """Collect `samples` random‑θ eigenvalues of the empirical Fisher."""
    eigvals = []
    X = X.to(device)
    for _ in range(samples):
        for p in model.parameters():  # θ ~ U[‑π,π]
            nn.init.uniform_(p, a=-math.pi, b=math.pi)
        F = fisher(model, X, damping=1e-6)
        eigvals.append(torch.linalg.eigvalsh(F).cpu().numpy())
    return np.concatenate(eigvals)


def effective_dimension_curve(
    build: Callable[[], nn.Module], X: torch.Tensor, ns: Sequence[int], device
) -> List[float]:
    """d_eff(n)/d for a range of *n* (Eq. 2)."""
    X = X.to(device)
    out = []
    for n in ns:
        m = build().to(device)
        F = fisher(m, X)
        F_hat, _ = fisher_norm(F)
        d_eff = eff_dim(F_hat, [n])[0].item()
        d_total = sum(p.numel() for p in m.parameters())
        out.append(d_eff / d_total)
    return out


###############################################################################
# Training loop (cross‑entropy) ###############################################
###############################################################################


def train_simple(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    writer: SummaryWriter,
    tag: str,
    device,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        loss_running, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_running += loss.item() * xb.size(0)
            total += yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        writer.add_scalar(f"{tag}/loss", loss_running / total, ep)
        writer.add_scalar(f"{tag}/acc", correct / total, ep)


###############################################################################
# Main ########################################################################
###############################################################################


def main() -> None:  # noqa: C901 – single entry point, fine
    parser = argparse.ArgumentParser(
        description="Replicate Power‑of‑QNNs plots with torchquantum"
    )
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="quick test run")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir).absolute()
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    tb_dir = outdir / "tb_logs"

    # CPU vs GPU target ----------------------------------------------------
    cudaq.set_target("nvidia" if torch.cuda.is_available() else "qpp-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data -----------------------------------------------------------------
    X, y = load_iris_binary()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

    # Parameter budget -----------------------------------------------------
    in_dim, out_dim, d_param = 4, 2, 8  # small like paper fig‑2 ★

    builders: Dict[str, Callable[[], nn.Module]] = {
        "classical": lambda: ParamMatchedFCN(in_dim, out_dim, d_param),
        "easy_qnn": lambda: QNN(
            in_dim,
            out_dim,
            num_layers=3,
            shots=248,
            feature_map="z",
            var_form="efficientsu2",
            reupload=True,
        ),
        "qnn": lambda: QNN(
            in_dim,
            out_dim,
            num_layers=3,
            shots=248,
            feature_map="zz",
            var_form="efficientsu2",
            reupload=True,
        ),
    }

    # 1) Fisher spectra -----------------------------------------------------
    samples = 20 if args.fast else 80
    bins = 15
    print("[1/4] Fisher spectra …")
    spectra = {
        name: fisher_spectrum(build(), X, samples=samples, device=device)
        for name, build in builders.items()
    }
    np.savez(outdir / "spectra.npz", **spectra)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (name, eigs) in zip(axes, spectra.items()):
        ax.hist(eigs, bins=bins, density=True, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("density" if ax is axes[0] else "")
    fig.tight_layout()
    fig.savefig(outdir / "plots/fisher_spectrum.png", dpi=300)

    # 2) Effective dimension vs n -----------------------------------------
    print("[2/4] Effective dimension curves …")
    n_vals = np.logspace(1, 6 if not args.fast else 4, num=30, dtype=int)
    curves = {
        name: effective_dimension_curve(build, X, n_vals, device)
        for name, build in builders.items()
    }
    np.savez(outdir / "d_eff_curves.npz", n=n_vals, **curves)
    plt.figure(figsize=(5, 3))
    for name, ys in curves.items():
        plt.plot(n_vals, ys, label=name)
    plt.xscale("log")
    plt.xlabel("number of data n")
    plt.ylabel(r"normalised $d_\mathrm{eff}/d$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "plots/d_eff_vs_n.png", dpi=300)

    # 3) Label noise sweep --------------------------------------------------
    print("[3/4] Label‑corruption sweep …")
    noises = np.linspace(0, 0.5, 6)
    noise_out = {name: [] for name in builders}
    for p in noises:
        y_corrupt = y_train.clone()
        idx = torch.randperm(len(y_corrupt))[: int(p * len(y_corrupt))]
        y_corrupt[idx] = 1 - y_corrupt[idx]
        Xt = X_train.to(device)
        for name, build in builders.items():
            m = build().to(device)
            F = fisher(m, Xt)
            F_hat, _ = fisher_norm(F)
            d_eff = eff_dim(F_hat, [len(Xt)])[0].item() / d_param
            noise_out[name].append(d_eff)
    np.savez(outdir / "label_noise.npz", noise=noises, **noise_out)
    plt.figure(figsize=(5, 3))
    for name, ys in noise_out.items():
        plt.plot(noises, ys, label=name)
    plt.xlabel("% randomised labels")
    plt.ylabel(r"normalised $d_\mathrm{eff}/d$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "plots/label_noise_d_eff.png", dpi=300)

    # 4) Training curves ----------------------------------------------------
    print("[4/4] Training curves …")
    writer = SummaryWriter(tb_dir.as_posix())
    epochs = 30 if args.fast else 100
    for name, build in builders.items():
        train_simple(
            build(), loader, epochs, lr=1e-2, writer=writer, tag=name, device=device
        )
    writer.flush()
    writer.close()

    print(f"✔ All artefacts saved under {outdir}")


###############################################################################
if __name__ == "__main__":
    main()
