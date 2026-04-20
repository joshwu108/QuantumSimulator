"""Internal circuit representation.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gate:
    """A quantum gate applied to specific qubits.

    Attributes:
        name: Gate identifier (e.g. "h", "cx", "rx").
        qubits: Ordered list of qubit indices this gate acts on.
                 For controlled gates, control qubits come first.
        params: Rotation parameters in radians (empty for non-parametric gates).
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        # Validate qubit count matches gate expectations
        expected = _GATE_QUBIT_COUNTS.get(self.name)
        if expected is not None and len(self.qubits) != expected:
            raise ValueError(
                f"Gate '{self.name}' expects {expected} qubit(s), "
                f"got {len(self.qubits)}"
            )


@dataclass(frozen=True)
class Measurement:
    """A measurement operation: measure qubit -> classical bit.

    Attributes:
        qubit: Index of the qubit to measure.
        clbit: Index of the classical bit to store the result.
    """
    qubit: int
    clbit: int


@dataclass
class Circuit:
    """Complete quantum circuit ready for simulation.

    Attributes:
        num_qubits: Total number of qubits in the circuit.
        num_clbits: Total number of classical bits.
        gates: Ordered list of gates to apply.
        measurements: List of measurement operations.
    """

    num_qubits: int
    num_clbits: int
    gates: list[Gate] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)

    def add_gate(self, gate: Gate) -> None:
        """Append a gate to the circuit."""
        for q in gate.qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {self.num_qubits - 1}]"
                )
        self.gates.append(gate)

    def add_measurement(self, measurement: Measurement) -> None:
        """Append a measurement to the circuit."""
        if measurement.qubit < 0 or measurement.qubit >= self.num_qubits:
            raise ValueError(
                f"Qubit index {measurement.qubit} out of range "
                f"[0, {self.num_qubits - 1}]"
            )
        if measurement.clbit < 0 or measurement.clbit >= self.num_clbits:
            raise ValueError(
                f"Classical bit index {measurement.clbit} out of range "
                f"[0, {self.num_clbits - 1}]"
            )
        self.measurements.append(measurement)


# Expected qubit counts for supported gates
_GATE_QUBIT_COUNTS: dict[str, int] = {
    # Single-qubit gates
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "u1": 1,
    "u2": 1,
    "u3": 1,
    "id": 1,
    # Two-qubit gates
    "cx": 2,
    "cz": 2,
    "swap": 2,
    # Three-qubit gates
    "ccx": 3,
}

# Expected parameter counts for supported gates
GATE_PARAM_COUNTS: dict[str, int] = {
    "x": 0,
    "y": 0,
    "z": 0,
    "h": 0,
    "s": 0,
    "sdg": 0,
    "t": 0,
    "tdg": 0,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "u1": 1,
    "u2": 2,
    "u3": 3,
    "id": 0,
    "cx": 0,
    "cz": 0,
    "swap": 0,
    "ccx": 0,
}
