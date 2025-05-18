import cudaq
import torch
import torch.nn as nn
import torchquantum.quantum as qu
import torchquantum.quantum.feature_maps as qfm
import torchquantum.quantum.variational_forms as qvf
from torch.autograd import Function
from typing import List, Any, Tuple
from math import pi
from cudaq import spin


class QuantumFunction(Function):
    r"""Custom autograd function for evaluating parameterized quantum circuits.

    This function wraps the evaluation of a quantum circuit defined by a feature map and
    a variational form. Gradients are computed via a finite-difference (parameter shift) rule.
    Fully compatible with functorch transforms via forward-mode setup_context.

    Attributes:
        in_features (int): Number of input features (qubits).
        out_features (int): Number of output features (qubits to measure).
        feature_map (qfm.FeatureMap): The feature map object.
        var_form (qvf.VariationalForm): The variational form object.
        backend (Any): Quantum backend for circuit evaluation.
        shots (int): Number of shots for circuit evaluation.
        reupload (bool): Whether to use reuploading strategy.
        _kernel (cudaq.Kernel): The pre-built combined circuit kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        feature_map: qfm.FeatureMap,
        var_form: qvf.VariationalForm,
        backend: Any,
        shots: int,
        reupload: bool = False,
        shift: float = pi / 2,
    ) -> None:
        r"""Initialize QuantumFunction with circuit components.

        Args:
            in_features (int): Number of input qubits.
            out_features (int): Number of qubits to measure.
            feature_map (FeatureMap): Data encoding circuit.
            var_form (VariationalForm): Parameterized ansatz.
            backend (Any): Quantum execution backend.
            shots (int): Number of measurement shots.
            reupload (bool): If True, interleave feature map each layer. Defaults to False.
            shift (float): Finite-difference shift for parameter shift rule. Defaults to pi/2.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.feature_map = feature_map
        self.var_form = var_form
        self.backend = backend
        self.shots = shots
        self.reupload = reupload
        self.pooling = qu.PoolingBlock(in_features, out_features, "cx") if in_features > out_features else None
        self._observables: List[cudaq.SpinOperator] = [
            spin.z(i) for i in range(out_features)
        ]
        self._kernel: cudaq.Kernel = self.build_kernel()
        self.shift = shift

    @property
    def kernel(self) -> cudaq.Kernel:
        r"""Return the pre-built quantum circuit kernel."""
        return self._kernel

    def build_kernel(self) -> cudaq.Kernel:
        r"""Construct the quantum circuit kernel by combining feature map and variational form.

        Depending on whether reuploading is enabled, the circuit applies the feature map
        once or repeatedly for each layer.

        Returns:
            cudaq.Kernel: The constructed quantum circuit.
        """
        feature_params: int = self.feature_map.get_parameter_count()
        var_form_params: int = self.var_form.get_parameter_count()
        total_params: int = feature_params + var_form_params

        kernel, *params = cudaq.make_kernel(*([float] * total_params))
        x, thetas = params[:feature_params], params[feature_params:]
        qvec = kernel.qalloc(max(self.in_features, self.out_features))
        if not self.reupload:
            kernel.apply_call(self.feature_map.kernel, qvec[: self.in_features], *x)
            kernel.apply_call(self.var_form.kernel, qvec, *thetas)
        else:
            ptr: int = 0
            # Apply the feature map and layer-specific variational form in a loop.
            for _, layer in self.var_form.layers.items():
                kernel.apply_call(self.feature_map.kernel, qvec[: self.in_features], *x)
                count: int = layer.get_parameter_count()
                kernel.apply_call(layer.kernel, qvec, *thetas[ptr : ptr + count])
                ptr += count
        
        if self.pooling:
            kernel.apply_call(self.pooling.kernel, qvec)
        return kernel

    def run(self, theta_vals: List[nn.Parameter], x: torch.Tensor) -> torch.Tensor:
        r"""Execute the quantum circuit for a batch of inputs.

        For each input sample, the circuit is executed and the expectation value of the Z
        observable on each qubit is measured.

        Args:
            theta_vals (List[nn.Parameter]): List of circuit parameters.
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Tensor of expectation values with shape (batch_size, in_features).
        """
        batch_size: int = x.shape[0]
        results: List[List[float]] = []
        theta_list: List[float] = theta_vals.detach().tolist()
        for i in range(batch_size):
            feature_list: List[float] = x[i].detach().tolist()
            params: List[float] = feature_list + theta_list
            observe_results = cudaq.observe(
                self.kernel, self._observables, *params, shots_count=self.shots
            )
            yi = [result.expectation() for result in observe_results]
            results.append(yi)
        device = theta_vals.device
        return torch.tensor(results, device=device, dtype=torch.float32)

    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        r"""
        Save everything we need for backward.

        This is called *after* forward() returned, with exactly the
        inputs to forward and the single Tensor output.
        """
        theta, x, quantum_circuit = inputs
        # we don’t need grads for these non-tensors
        ctx.set_materialize_grads(False)
        ctx.quantum_circuit = quantum_circuit
        ctx.shift = quantum_circuit.shift
        # stash tensors for backward:
        ctx.save_for_backward(theta, x, output)

    @staticmethod
    def forward(
        theta: torch.Tensor, x: torch.Tensor, quantum_circuit: "QuantumFunction"
    ) -> torch.Tensor:
        r"""
        Execute the circuit to get expectation values.

        Args:
            theta (Tensor): 1-D tensor of circuit parameters.
            x (Tensor):     shape (batch, in_features), the data.
            quantum_circuit (QuantumFunction): instance carrying the kernel.
            shift (float):  finite-difference shift.

        Returns:
            Tensor of shape (batch, out_features) with expectations.
        """
        return quantum_circuit.run(theta, x)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        r"""Backward pass: compute gradients via parameter-shift.

        Args:
            ctx: Autograd context containing saved tensors.
            grad_output (torch.Tensor): Gradient of loss w.r.t. output.

        Returns:
            Tuple[torch.Tensor, None, None, None]: Gradient w.r.t. theta, Nones.
        """
        theta, x, _ = ctx.saved_tensors
        device = theta.device
        gradients: torch.Tensor = torch.zeros_like(theta, device=device)
        for i in range(theta.numel()):
            theta_plus: torch.Tensor = theta.clone()
            theta_plus[i] += ctx.shift
            y_plus: torch.Tensor = ctx.quantum_circuit.run(theta_plus, x)
            theta_minus: torch.Tensor = theta.clone()
            theta_minus[i] -= ctx.shift
            y_minus: torch.Tensor = ctx.quantum_circuit.run(theta_minus, x)
            diff: torch.Tensor = (y_plus - y_minus) / (2 * ctx.shift)
            gradients[i] = torch.sum(grad_output * diff)
        return gradients, None, None


# Alias for nn.Modules
quantum_fn = QuantumFunction.apply
