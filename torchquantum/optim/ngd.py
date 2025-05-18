import torch
from torch.optim import Optimizer
from torchquantum.utils.fisher_information import FisherInformation
from typing import Optional, Callable


class NGD(Optimizer):
    r"""
    Natural Gradient Descent (NGD) optimizer for quantum or hybrid quantum-classical
    neural networks. Moves parameters in the direction of F^{-1} grad(L), where F is the
    Fisher Information Matrix (FIM). This enforces a geometry based on the Fubini-Study
    metric for quantum parameters (or the standard Fisher metric for classical parameters).

    For each step, we:
      1) Compute the gradient \nabla L wrt parameters (via backward or parameter-shift).
      2) Compute or retrieve F from FisherInformation.
      3) Solve F * delta = grad to get the natural gradient direction delta = F^{-1} grad.
      4) Update parameters: theta <- theta - lr * delta

    Because F can be large, you may want a block-diagonal or diagonal approximation to reduce cost.

    Usage:
    >>> optimizer = NGD(model, lr=0.1, fish_comp=fisher_comp)
    >>> for epoch in range(epochs):
    >>>     for x, y in dataloader:
    >>>         def closure():
    >>>             optimizer.zero_grad()
    >>>             out = model(x)
    >>>             loss = loss_fn(out, y)
    >>>             loss.backward()
    >>>             return loss
    >>>         optimizer.step(closure)

    Args:
        params (iterable): Iterable of model parameters to optimize.
        lr (float): Learning rate.
        fish_comp (FisherInformation): Utility to compute or fetch Fisher matrix for the model.
        damping (float, optional): Additional damping to add to F. Defaults to 1e-5.
        block_diagonal (bool, optional): If True, uses block-diagonal approximation in F. Defaults to False.
    """

    def __init__(
        self,
        params,
        lr: float,
        fish_comp: FisherInformation,
        damping: float = 1e-5,
        block_diagonal: bool = False,
    ):
        defaults = dict(lr=lr, damping=damping, block_diagonal=block_diagonal)
        super().__init__(params, defaults)
        self.fish_comp = fish_comp
        self._param_shapes = []
        self._total_params = 0

        # Flatten parameters for index tracking
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    numel = p.numel()
                    self._param_shapes.append(p.shape)
                    self._total_params += numel

    @torch.no_grad()
    def step(self) -> None:
        r"""
        Perform a single NGD step (no closure).
        You must have called `loss.backward()` externally and computed the FIM
        (storing it in `fish_comp.last_fisher`, for example).
        """
        # 1) Gather all parameters that have gradients
        param_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.grad is not None:
                    param_list.append(p)

        # 2) Flatten the gradient into a single vector
        grad_vec = self._gather_grad_vec(param_list)

        # 3) Retrieve or compute the FIM (already stored by user code)
        F = None
        if self.fish_comp and hasattr(self.fish_comp, "last_fisher"):
            F = self.fish_comp.last_fisher
        else:
            # If user didn't compute or doesn't want to use it, fallback to identity
            # (which is just standard gradient descent).
            F = torch.eye(self._total_params, device=grad_vec.device)

        # 4) Add damping for stability
        damping = self.defaults["damping"]
        F_damped = F + damping * torch.eye(F.shape[0], device=F.device)

        # 5) Solve for the natural gradient: (F + dampingI)*delta = grad
        delta = torch.linalg.solve(F_damped, grad_vec)

        # 6) Update parameters: theta <- theta - lr*delta
        lr = self.defaults["lr"]
        self._distribute_update(param_list, -lr * delta)

    def _gather_grad_vec(self, param_list) -> torch.Tensor:
        # Flatten each param grad -> cat
        grads = []
        for p in param_list:
            grads.append(p.grad.view(-1))
        return torch.cat(grads, dim=0)

    def _distribute_update(self, param_list, update_vec: torch.Tensor):
        # Opposite of gather: reshape slices of update into each param's shape
        offset = 0
        for p in param_list:
            numel = p.numel()
            g_slice = update_vec[offset : offset + numel].view(p.shape)
            p.add_(g_slice)
            offset += numel
