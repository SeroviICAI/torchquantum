from typing import List
import torchquantum.quantum as qu


class PoolingBlock(qu.EntanglementBlock):
    r"""Pooling block ansatz for qubit fan-in entanglement.

    Funnels information from any *surplus* qubits (those with index
    ``>= out_features``) into the first ``out_features`` qubits by applying a
    two-qubit entangling gate between each surplus **control** qubit and a
    corresponding **target** qubit chosen modulo ``out_features``.

    Circuit Diagram Example (``gate="cx"``, ``num_qubits=6``, ``out_features=3``):

    .. code-block:: text

         q2 ●───┐
         q5 │   │
         q0 X───●── q0 (measured)
            │   │
         q3 ●───┤
         q4 X───┘

    Args:
        num_qubits (int): Total number of qubits in the register.
        out_features (int): Number of qubits to be measured (must satisfy
            ``out_features <= num_qubits``).
        gate (str, optional): Entanglement gate to use (e.g. ``"cx"``,
            ``"cnot"``, ``"cz"``, or ``"cphase"``). Defaults to ``"cx"``.

    Raises:
        ValueError: If ``out_features`` exceeds ``num_qubits``.

    Notes:
        This block is **parameter-free**—:py:meth:`get_parameter_count`
        always returns ``0``—so adding pooling does not introduce additional
        trainable parameters.
    """

    def __init__(self, num_qubits: int, out_features: int, gate: str = "cx") -> None:
        if out_features > num_qubits:
            raise ValueError(
                f"out_features ({out_features}) must not exceed num_qubits "
                f"({num_qubits})."
            )

        pairs: List[List[int]] = [
            [ctrl, ctrl % out_features] for ctrl in range(out_features, num_qubits)
        ]
        super().__init__(pairs, gate)

        self.out_features = out_features
        self.gate = gate.lower()
