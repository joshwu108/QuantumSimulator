"""Shared helpers for the interactive quantum workbench UI.

This module keeps the Streamlit app thin by collecting the core logic for:
  - editable circuit operations
  - simulation dispatch
  - OpenQASM export
  - simple circuit diagram rendering
  - compact state summaries for display
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np

from simulator.circuit import Circuit, GATE_PARAM_COUNTS, Gate, Measurement
from simulator.engine import (
    DensityMatrixResult,
    DensityMatrixSimulator,
    SimulationResult,
    StatevectorSimulator,
)
from simulator.noise import (
    AmplitudeDampingChannel,
    DepolarizingChannel,
    NoiseModel,
)
from simulator.parser import QASMParseError, parse_qasm
from simulator.visualization import state_to_density_matrix

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any


APP_SINGLE_QUBIT_GATES: tuple[str, ...] = (
    "id",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "u1",
    "u2",
    "u3",
)
APP_CONTROLLED_GATES: tuple[str, ...] = ("cx", "cz")
APP_SUPPORTED_GATES: tuple[str, ...] = APP_SINGLE_QUBIT_GATES + APP_CONTROLLED_GATES

PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "u1": ("lambda",),
    "u2": ("phi", "lambda"),
    "u3": ("theta", "phi", "lambda"),
}

NOISE_OPTIONS: tuple[str, ...] = ("none", "depolarizing", "amplitude_damping")


@dataclass(frozen=True)
class OperationRow:
    """Editable operation record used by the frontend."""

    gate: str
    target: int
    control: int | None = None
    param_1: float = 0.0
    param_2: float = 0.0
    param_3: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class PresetSpec:
    """Small curated starter circuit for the workbench."""

    name: str
    description: str
    num_qubits: int
    measured_qubits: tuple[int, ...]
    operations: tuple[OperationRow, ...]


@dataclass
class SimulationSnapshot:
    """Bundle of live simulation outputs for the frontend."""

    circuit: Circuit
    result: SimulationResult | DensityMatrixResult
    counts: dict[str, int]
    qasm: str
    mode_label: str
    purity: float


PRESETS: dict[str, PresetSpec] = {
    "Blank": PresetSpec(
        name="Blank",
        description="Start from an empty workbench and add your own gates.",
        num_qubits=2,
        measured_qubits=(0, 1),
        operations=(),
    ),
    "Superposition": PresetSpec(
        name="Superposition",
        description="A single Hadamard puts one qubit into a 50/50 superposition.",
        num_qubits=1,
        measured_qubits=(0,),
        operations=(OperationRow(gate="h", target=0),),
    ),
    "Bell State": PresetSpec(
        name="Bell State",
        description="Creates entanglement with H on q0 followed by CX(q0, q1).",
        num_qubits=2,
        measured_qubits=(0, 1),
        operations=(
            OperationRow(gate="h", target=0),
            OperationRow(gate="cx", target=1, control=0),
        ),
    ),
    "GHZ State": PresetSpec(
        name="GHZ State",
        description="Extends the Bell-state pattern to three qubits.",
        num_qubits=3,
        measured_qubits=(0, 1, 2),
        operations=(
            OperationRow(gate="h", target=0),
            OperationRow(gate="cx", target=1, control=0),
            OperationRow(gate="cx", target=2, control=0),
        ),
    ),
    "Rotation Lab": PresetSpec(
        name="Rotation Lab",
        description="A compact single-qubit circuit for exploring parametric gates.",
        num_qubits=1,
        measured_qubits=(0,),
        operations=(
            OperationRow(gate="ry", target=0, param_1=pi / 3),
            OperationRow(gate="rz", target=0, param_1=pi / 4),
            OperationRow(gate="u3", target=0, param_1=pi / 2, param_2=0.3, param_3=-0.6),
        ),
    ),
}


def blank_operation(num_qubits: int) -> OperationRow:
    """Return a sensible starter row for the current qubit count."""
    target = 0 if num_qubits <= 1 else min(1, num_qubits - 1)
    return OperationRow(gate="h", target=target, control=0 if num_qubits > 1 else None)


def preset_names() -> list[str]:
    """Return preset names in display order."""
    return list(PRESETS.keys())


def get_preset(name: str) -> PresetSpec:
    """Return a named preset."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset '{name}'.") from exc


def sanitize_operation_rows(
    rows: list[OperationRow],
    num_qubits: int,
) -> list[OperationRow]:
    """Clamp qubit indices and normalize operation settings for the UI."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")

    sanitized: list[OperationRow] = []
    for row in rows:
        gate = row.gate if row.gate in APP_SUPPORTED_GATES else "h"
        target = int(np.clip(row.target, 0, num_qubits - 1))
        control = row.control

        if gate in APP_CONTROLLED_GATES:
            if num_qubits < 2:
                gate = "h"
                control = None
            else:
                if control is None:
                    control = 0 if target != 0 else 1
                control = int(np.clip(control, 0, num_qubits - 1))
                if control == target:
                    control = (target + 1) % num_qubits
        else:
            control = None

        sanitized.append(
            OperationRow(
                gate=gate,
                target=target,
                control=control,
                param_1=float(row.param_1),
                param_2=float(row.param_2),
                param_3=float(row.param_3),
                enabled=bool(row.enabled),
            )
        )

    return sanitized


def build_circuit_from_operation_rows(
    num_qubits: int,
    measured_qubits: list[int] | tuple[int, ...],
    rows: list[OperationRow],
) -> Circuit:
    """Build a Circuit from editable workbench rows."""
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive.")

    clean_rows = list(rows)
    measured = sorted({int(qubit) for qubit in measured_qubits})
    for qubit in measured:
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(
                f"Measured qubit {qubit} out of range [0, {num_qubits - 1}]"
            )

    circuit = Circuit(num_qubits=num_qubits, num_clbits=len(measured))

    for row in clean_rows:
        if not row.enabled:
            continue
        if row.gate not in APP_SUPPORTED_GATES:
            raise ValueError(f"Unsupported gate '{row.gate}'.")
        if row.target < 0 or row.target >= num_qubits:
            raise ValueError(
                f"Target qubit {row.target} out of range [0, {num_qubits - 1}]"
            )

        param_count = GATE_PARAM_COUNTS[row.gate]
        params = (row.param_1, row.param_2, row.param_3)[:param_count]

        if row.gate in APP_CONTROLLED_GATES:
            if row.control is None:
                raise ValueError(f"Gate '{row.gate}' requires a control qubit.")
            if row.control < 0 or row.control >= num_qubits:
                raise ValueError(
                    f"Control qubit {row.control} out of range [0, {num_qubits - 1}]"
                )
            if row.control == row.target:
                raise ValueError("Control and target qubits must be different.")
            gate = Gate(name=row.gate, qubits=(row.control, row.target), params=params)
        else:
            gate = Gate(name=row.gate, qubits=(row.target,), params=params)

        circuit.add_gate(gate)

    for clbit, qubit in enumerate(measured):
        circuit.add_measurement(Measurement(qubit=qubit, clbit=clbit))

    return circuit


def operation_rows_from_circuit(circuit: Circuit) -> list[OperationRow]:
    """Convert a Circuit into editable rows for the frontend."""
    rows: list[OperationRow] = []
    for gate in circuit.gates:
        params = list(gate.params) + [0.0, 0.0, 0.0]
        if gate.name in APP_CONTROLLED_GATES:
            control, target = gate.qubits
            rows.append(
                OperationRow(
                    gate=gate.name,
                    target=target,
                    control=control,
                    param_1=float(params[0]),
                    param_2=float(params[1]),
                    param_3=float(params[2]),
                )
            )
        elif gate.name in APP_SINGLE_QUBIT_GATES:
            rows.append(
                OperationRow(
                    gate=gate.name,
                    target=gate.qubits[0],
                    param_1=float(params[0]),
                    param_2=float(params[1]),
                    param_3=float(params[2]),
                )
            )
        else:
            raise ValueError(
                f"Workbench import does not support gate '{gate.name}'. "
                "Use the supported gate set shown in the app sidebar."
            )

    return rows


def import_qasm_program(source: str) -> tuple[int, list[int], list[OperationRow]]:
    """Parse a QASM program and convert it to workbench state."""
    circuit = parse_qasm(source)
    measured_qubits = [measurement.qubit for measurement in circuit.measurements]
    return circuit.num_qubits, measured_qubits, operation_rows_from_circuit(circuit)


def export_openqasm(
    num_qubits: int,
    measured_qubits: list[int] | tuple[int, ...],
    rows: list[OperationRow],
) -> str:
    """Serialize the current workbench state into OpenQASM 2.0."""
    circuit = build_circuit_from_operation_rows(num_qubits, measured_qubits, rows)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.num_qubits}];",
    ]
    if circuit.num_clbits:
        lines.append(f"creg c[{circuit.num_clbits}];")
    lines.append("")

    for gate in circuit.gates:
        if gate.params:
            params = ", ".join(_format_float(value) for value in gate.params)
            gate_head = f"{gate.name}({params})"
        else:
            gate_head = gate.name

        if len(gate.qubits) == 1:
            lines.append(f"{gate_head} q[{gate.qubits[0]}];")
        else:
            qubit_args = ", ".join(f"q[{qubit}]" for qubit in gate.qubits)
            lines.append(f"{gate_head} {qubit_args};")

    if circuit.gates:
        lines.append("")

    for measurement in circuit.measurements:
        lines.append(
            f"measure q[{measurement.qubit}] -> c[{measurement.clbit}];"
        )

    return "\n".join(lines)


def build_noise_model(noise_mode: str, strength: float) -> NoiseModel | None:
    """Create a global noise model for the selected mode."""
    if noise_mode == "none":
        return None

    model = NoiseModel()
    if noise_mode == "depolarizing":
        model.add_all_gates_noise(DepolarizingChannel(p=float(strength)))
    elif noise_mode == "amplitude_damping":
        model.add_all_gates_noise(AmplitudeDampingChannel(gamma=float(strength)))
    else:
        raise ValueError(f"Unknown noise mode '{noise_mode}'.")

    return model


def simulate_operation_rows(
    num_qubits: int,
    measured_qubits: list[int] | tuple[int, ...],
    rows: list[OperationRow],
    shots: int,
    noise_mode: str = "none",
    noise_strength: float = 0.0,
    seed: int | None = 7,
) -> SimulationSnapshot:
    """Run the circuit editor state through the appropriate simulator."""
    circuit = build_circuit_from_operation_rows(num_qubits, measured_qubits, rows)
    qasm = export_openqasm(num_qubits, measured_qubits, rows)

    if noise_mode == "none":
        result: SimulationResult | DensityMatrixResult = StatevectorSimulator(circuit).run()
        mode_label = "Noiseless statevector"
    else:
        noise_model = build_noise_model(noise_mode, noise_strength)
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        mode_label = f"Noisy density matrix ({noise_mode.replace('_', ' ')})"

    counts = result.get_counts(shots=shots, seed=seed) if circuit.measurements else {}
    density_matrix = state_to_density_matrix(result)
    purity = float(np.real(np.trace(density_matrix @ density_matrix)))

    return SimulationSnapshot(
        circuit=circuit,
        result=result,
        counts=counts,
        qasm=qasm,
        mode_label=mode_label,
        purity=purity,
    )


def state_table_rows(
    result: SimulationResult | DensityMatrixResult,
    max_rows: int = 16,
) -> list[dict[str, str]]:
    """Create a compact, display-friendly state summary."""
    probabilities = np.asarray(result.probabilities, dtype=float)
    num_qubits = _infer_num_qubits(len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:max_rows]

    rows: list[dict[str, str]] = []
    if isinstance(result, SimulationResult):
        amplitudes = result.statevector
        for index in top_indices:
            rows.append(
                {
                    "basis_state": f"|{format(index, f'0{num_qubits}b')}>",
                    "probability": f"{probabilities[index]:.6f}",
                    "amplitude": _format_complex(amplitudes[index]),
                }
            )
    else:
        diagonal = np.diag(result.density_matrix)
        for index in top_indices:
            rows.append(
                {
                    "basis_state": f"|{format(index, f'0{num_qubits}b')}>",
                    "probability": f"{probabilities[index]:.6f}",
                    "rho_ii": _format_complex(diagonal[index]),
                }
            )

    return rows


def plot_circuit_diagram(
    num_qubits: int,
    measured_qubits: list[int] | tuple[int, ...],
    rows: list[OperationRow],
    *,
    title: str = "Live Circuit",
) -> tuple[Figure, Axes]:
    """Render a simple composer-style circuit preview with Matplotlib."""
    plt, patches = _require_matplotlib()

    measured = sorted({int(qubit) for qubit in measured_qubits})
    clean_rows = sanitize_operation_rows(rows, num_qubits)
    active_rows = [row for row in clean_rows if row.enabled]
    total_columns = max(len(active_rows), 1) + (1 if measured else 0)

    fig_width = max(9.0, 2.0 + total_columns * 1.45)
    fig_height = max(3.6, 1.4 + num_qubits * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#0c141b")
    ax.set_facecolor("#0c141b")

    y_positions = {qubit: num_qubits - 1 - qubit for qubit in range(num_qubits)}
    for qubit in range(num_qubits):
        y = y_positions[qubit]
        ax.plot(
            [0.3, total_columns + 0.8],
            [y, y],
            color="#506170",
            linewidth=1.6,
            solid_capstyle="round",
            zorder=1,
        )
        ax.text(
            0.08,
            y,
            f"q{qubit}",
            color="#D4DEE6",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    if not active_rows:
        ax.text(
            (total_columns + 1.0) / 2,
            (num_qubits - 1) / 2 if num_qubits > 1 else 0.0,
            "Add a gate to start exploring",
            color="#AABBC8",
            ha="center",
            va="center",
            fontsize=14,
        )

    color_map = {
        "id": "#7E8B95",
        "h": "#74B9B1",
        "x": "#C9986B",
        "y": "#C9986B",
        "z": "#C9986B",
        "s": "#8B7FB2",
        "sdg": "#8B7FB2",
        "t": "#9A7AA6",
        "tdg": "#9A7AA6",
        "rx": "#7FA88F",
        "ry": "#7FA88F",
        "rz": "#7FA88F",
        "u1": "#7C9FA0",
        "u2": "#7C9FA0",
        "u3": "#7C9FA0",
        "cx": "#B97A7A",
        "cz": "#B97A7A",
    }

    for step, row in enumerate(active_rows, start=1):
        if row.gate in APP_CONTROLLED_GATES and row.control is not None:
            control_y = y_positions[row.control]
            target_y = y_positions[row.target]
            ax.plot(
                [step, step],
                [control_y, target_y],
                color="#DCE5EC",
                linewidth=1.6,
                zorder=2,
            )
            ax.scatter(
                [step],
                [control_y],
                s=90,
                color="#DCE5EC",
                zorder=4,
            )
            _draw_gate_box(
                ax,
                patches,
                step,
                target_y,
                row.gate[-1].upper(),
                fill=color_map[row.gate],
            )
        else:
            target_y = y_positions[row.target]
            _draw_gate_box(
                ax,
                patches,
                step,
                target_y,
                row.gate.upper(),
                fill=color_map[row.gate],
            )

    if measured:
        measurement_x = len(active_rows) + 1
        for qubit in measured:
            _draw_gate_box(
                ax,
                patches,
                measurement_x,
                y_positions[qubit],
                "M",
                fill="#D1B06B",
            )

    ax.set_title(title, color="#F2F6FA", fontsize=17, fontweight="bold", pad=16)
    ax.text(
        0.3,
        -0.72,
        "Little-endian ordering: q0 is the least-significant qubit in state labels.",
        color="#91A2AF",
        fontsize=10,
        ha="left",
    )
    ax.set_xlim(0.0, total_columns + 1.0)
    ax.set_ylim(-1.1, num_qubits - 0.2)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def parameter_help_text(gate: str) -> str:
    """Return compact UI help for gate parameters."""
    names = PARAMETER_NAMES.get(gate)
    if not names:
        return "No parameters"
    return ", ".join(names)


def noise_help_text(noise_mode: str) -> str:
    """Small explanatory label for the selected noise model."""
    if noise_mode == "depolarizing":
        return "Applies a random Pauli error after each gate."
    if noise_mode == "amplitude_damping":
        return "Models relaxation from |1> toward |0> after each gate."
    return "Runs the exact noiseless statevector simulator."


def _draw_gate_box(
    ax: Axes,
    patches,
    x: float,
    y: float,
    label: str,
    *,
    fill: str,
) -> None:
    """Draw a rounded gate box centered on a qubit wire."""
    rect = patches.FancyBboxPatch(
        (x - 0.35, y - 0.28),
        0.7,
        0.56,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#DCE5EC",
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#0b131a",
        zorder=4,
    )


def _require_matplotlib():
    """Import Matplotlib only when the UI needs it."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for circuit rendering. Install dependencies "
            "from requirements.txt before launching the app."
        ) from exc
    return plt, patches


def _format_float(value: float) -> str:
    """Format a float compactly for generated QASM."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_complex(value: complex) -> str:
    """Format a complex number for a compact state table."""
    real = f"{value.real:.4f}"
    imag = f"{abs(value.imag):.4f}"
    sign = "+" if value.imag >= 0 else "-"
    return f"{real}{sign}{imag}j"


def _infer_num_qubits(dimension: int) -> int:
    """Infer n from a Hilbert-space dimension 2^n."""
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError(
            f"State dimension must be a positive power of two, got {dimension}."
        )
    return dimension.bit_length() - 1


__all__ = [
    "APP_CONTROLLED_GATES",
    "APP_SINGLE_QUBIT_GATES",
    "APP_SUPPORTED_GATES",
    "NOISE_OPTIONS",
    "OperationRow",
    "PresetSpec",
    "SimulationSnapshot",
    "PRESETS",
    "blank_operation",
    "build_circuit_from_operation_rows",
    "build_noise_model",
    "export_openqasm",
    "get_preset",
    "import_qasm_program",
    "noise_help_text",
    "operation_rows_from_circuit",
    "parameter_help_text",
    "plot_circuit_diagram",
    "preset_names",
    "sanitize_operation_rows",
    "simulate_operation_rows",
    "state_table_rows",
]
