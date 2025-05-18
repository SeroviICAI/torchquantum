import math
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call
from typing import Sequence, Tuple

__all__ = ["fisher", "fisher_norm", "eff_dim"]


def _make_flat_to_param_dict(model: nn.Module):
    slices, offset = [], 0
    for p in model.parameters():
        cnt = p.numel()
        slices.append((offset, p.shape))
        offset += cnt

    def flat_to_param_dict(flat: torch.Tensor):
        pd = {}
        for (n, _), (off, shape) in zip(model.named_parameters(), slices):
            length = int(torch.tensor(shape).prod().item())
            pd[n] = flat[off : off + length].view(shape)
        return pd

    return flat_to_param_dict


def fisher(
    model: nn.Module,
    inputs: torch.Tensor,
    damping: float = 1e-6,
) -> torch.Tensor:
    """
    Empirical Fisher via a Python loop + autograd.grad per sample.

    Args:
        model:   nn.Module that maps (B, ...) → (B, C)
        inputs:  (B, in_features)
        damping: small diagonal reg.

    Returns:
        F:       (D, D) Fisher matrix, D = total #params
    """
    device = next(model.parameters()).device
    # 1) flatten θ into a leaf tensor
    flat0 = parameters_to_vector(model.parameters()).to(device)
    flat = flat0.clone().detach().requires_grad_(True)
    to_dict = _make_flat_to_param_dict(model)

    B = inputs.size(0)
    D = flat.numel()
    G = torch.zeros(B, D, device=device)

    # 2) for each sample, compute ∂ℓ/∂flat
    for i in range(B):
        x = inputs[i].to(device)
        # forward + scalar loss
        out = functional_call(model, to_dict(flat), (x.unsqueeze(0),))  # (1, C)
        loss = -out.sum()
        # gradient of loss wrt flat
        (grad_flat,) = torch.autograd.grad(loss, flat)
        G[i] = grad_flat

    # 3) empirical Fisher + damping
    F = (G.T @ G) / float(B)
    F += damping * torch.eye(D, device=device)
    return F


def fisher_norm(
    fisher: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    d = fisher.size(0)
    tr = float(torch.trace(fisher))
    return (d / tr) * fisher, tr


def eff_dim(
    fhat: torch.Tensor,
    dataset_sizes: Sequence[int],
) -> torch.Tensor:
    d = fhat.size(0)
    out = []
    for n in dataset_sizes:
        scale = n / (2 * math.pi)
        M = torch.eye(d, device=fhat.device) + scale * fhat
        sign, logdet = torch.linalg.slogdet(M)
        logdet = logdet if sign > 0 else torch.tensor(0.0, device=fhat.device)
        out.append(2.0 * logdet / math.log(scale))
    return torch.stack(out)
