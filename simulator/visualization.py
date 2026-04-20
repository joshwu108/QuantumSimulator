"""Visualization helpers for simulator outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from simulator.engine import DensityMatrixResult, SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any


_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def state_to_density_matrix(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Return a density matrix for a statevector, density matrix, or result.

    Args:
        state: One of the following:
            - `SimulationResult`
            - `DensityMatrixResult`
            - a 1D statevector of length 2^n
            - a square density matrix of shape (2^n, 2^n)

    Returns:
        A complex density matrix of shape (2^n, 2^n).
    """
    if isinstance(state, SimulationResult):
        state = state.statevector
    elif isinstance(state, DensityMatrixResult):
        state = state.density_matrix

    array = np.asarray(state, dtype=complex)
    if array.ndim == 1:
        _infer_num_qubits(array.size)
        return np.outer(array, np.conjugate(array))
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        _infer_num_qubits(array.shape[0])
        return array

    raise ValueError(
        "State must be a statevector, density matrix, SimulationResult, "
        "or DensityMatrixResult."
    )


def single_qubit_reduced_density_matrix(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
    qubit: int = 0,
) -> np.ndarray:
    """Return the reduced 2x2 density matrix for one qubit.

    For a multi-qubit state this performs a partial trace over every other
    qubit. For a single-qubit state it simply returns the full density matrix.
    """
    rho = state_to_density_matrix(state)
    num_qubits = _infer_num_qubits(rho.shape[0])

    if qubit < 0 or qubit >= num_qubits:
        raise ValueError(
            f"Qubit index {qubit} out of range [0, {num_qubits - 1}]"
        )

    if num_qubits == 1:
        return rho

    rho_tensor = rho.reshape([2] * (2 * num_qubits))
    row_axis = num_qubits - 1 - qubit
    col_axis = (2 * num_qubits) - 1 - qubit

    # Move the target qubit's row/column axes to the front so the remaining
    # qubits become one combined "environment" that can be traced out.
    permutation = (
        [row_axis]
        + [axis for axis in range(num_qubits) if axis != row_axis]
        + [col_axis]
        + [
            axis
            for axis in range(num_qubits, 2 * num_qubits)
            if axis != col_axis
        ]
    )
    rho_tensor = np.transpose(rho_tensor, axes=permutation)

    environment_dim = 2 ** (num_qubits - 1)
    rho_tensor = rho_tensor.reshape(2, environment_dim, 2, environment_dim)
    return np.trace(rho_tensor, axis1=1, axis2=3)


def bloch_vector_from_density_matrix(density_matrix: np.ndarray) -> np.ndarray:
    """Compute the Bloch vector r = (Tr(ρX), Tr(ρY), Tr(ρZ)) for one qubit."""
    rho = np.asarray(density_matrix, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError(
            "Bloch vectors require a single-qubit 2x2 density matrix."
        )

    vector = np.array(
        [
            np.trace(rho @ _PAULI_X),
            np.trace(rho @ _PAULI_Y),
            np.trace(rho @ _PAULI_Z),
        ],
        dtype=complex,
    )
    return np.real(vector).astype(float)


def bloch_vector_from_state(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
    qubit: int = 0,
) -> np.ndarray:
    """Return the Bloch vector for a one-qubit state or reduced qubit state."""
    reduced_rho = single_qubit_reduced_density_matrix(state, qubit=qubit)
    return bloch_vector_from_density_matrix(reduced_rho)


def plot_counts(
    counts: Mapping[str, int],
    *,
    ax: Axes | None = None,
    title: str = "Measurement Counts",
    color: str = "#2A6F97",
) -> tuple[Figure, Axes]:
    """Plot a histogram of measured bitstring counts."""
    if not counts:
        raise ValueError("Counts dictionary must not be empty.")

    labels = _sorted_labels(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = _get_2d_axes(ax)
    bars = ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.set_ylim(0, max(values) * 1.15 if max(values) else 1.0)

    if len(values) <= 16:
        _annotate_bars(ax, bars, [str(value) for value in values])

    fig.tight_layout()
    return fig, ax


def plot_probabilities(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = "#E9C46A",
    min_probability: float = 0.0,
) -> tuple[Figure, Axes]:
    """Plot basis-state probabilities from a statevector or density matrix.

    Labels are shown in the usual ket order `|q[n-1] ... q[0]>`, while the
    simulator still keeps its efficient little-endian internal indexing.
    """
    probabilities, num_qubits = _probabilities_from_state(state)
    basis_labels = [f"|{label}>" for label in _basis_labels(num_qubits)]

    if min_probability > 0.0:
        filtered = [
            (label, probability)
            for label, probability in zip(basis_labels, probabilities)
            if probability >= min_probability
        ]
        if not filtered:
            max_index = int(np.argmax(probabilities))
            filtered = [(basis_labels[max_index], float(probabilities[max_index]))]
        basis_labels = [label for label, _ in filtered]
        probabilities = np.array([probability for _, probability in filtered])

    fig, ax = _get_2d_axes(ax)
    bars = ax.bar(
        basis_labels,
        probabilities,
        color=color,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title(title or "Basis-State Probabilities")
    ax.set_xlabel("Basis state")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, max(probabilities) * 1.15 if len(probabilities) else 1.0)

    if len(basis_labels) > 8:
        ax.tick_params(axis="x", labelrotation=45)

    if len(probabilities) <= 16:
        _annotate_bars(ax, bars, [f"{value:.3f}" for value in probabilities])

    fig.tight_layout()
    return fig, ax


def plot_bloch_from_state(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
    *,
    qubit: int = 0,
    ax: Axes | None = None,
    title: str | None = None,
    vector_color: str = "#D1495B",
) -> tuple[Figure, Axes]:
    """Plot a Bloch sphere for a single qubit or reduced single-qubit state."""
    reduced_rho = single_qubit_reduced_density_matrix(state, qubit=qubit)
    bloch_vector = bloch_vector_from_density_matrix(reduced_rho)

    fig, ax = _get_3d_axes(ax)
    _draw_bloch_sphere(ax)

    ax.quiver(
        0.0,
        0.0,
        0.0,
        bloch_vector[0],
        bloch_vector[1],
        bloch_vector[2],
        color=vector_color,
        linewidth=2.5,
        arrow_length_ratio=0.12,
    )
    ax.scatter(
        [bloch_vector[0]],
        [bloch_vector[1]],
        [bloch_vector[2]],
        color=vector_color,
        s=60,
    )

    purity = float(np.real(np.trace(reduced_rho @ reduced_rho)))
    vector_length = float(np.linalg.norm(bloch_vector))
    ax.set_title(title or f"Bloch Sphere for Qubit {qubit}")
    ax.text2D(
        0.02,
        0.95,
        f"|r| = {vector_length:.3f}\nTr($\\rho^2$) = {purity:.3f}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()
    return fig, ax


def _probabilities_from_state(
    state: SimulationResult | DensityMatrixResult | Sequence[complex] | np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return a normalized probability vector and the matching qubit count."""
    if isinstance(state, SimulationResult):
        probabilities = np.asarray(state.probabilities, dtype=float)
        num_qubits = _infer_num_qubits(state.statevector.size)
    elif isinstance(state, DensityMatrixResult):
        probabilities = np.asarray(state.probabilities, dtype=float)
        num_qubits = _infer_num_qubits(state.density_matrix.shape[0])
    else:
        array = np.asarray(state, dtype=complex)
        if array.ndim == 1:
            num_qubits = _infer_num_qubits(array.size)
            probabilities = np.abs(array) ** 2
        elif array.ndim == 2 and array.shape[0] == array.shape[1]:
            num_qubits = _infer_num_qubits(array.shape[0])
            probabilities = np.real(np.diag(array))
        else:
            raise ValueError(
                "State must be a statevector, density matrix, SimulationResult, "
                "or DensityMatrixResult."
            )

    probabilities = np.real(probabilities).astype(float)
    probabilities = np.clip(probabilities, 0.0, None)

    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("State probabilities must sum to a positive value.")
    return probabilities / total, num_qubits


def _infer_num_qubits(dimension: int) -> int:
    """Infer n from a Hilbert-space dimension 2^n."""
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError(
            f"State dimension must be a positive power of two, got {dimension}."
        )
    return dimension.bit_length() - 1


def _basis_labels(num_qubits: int) -> list[str]:
    """Return basis labels in ket order |q[n-1] ... q[0]>."""
    return [format(index, f"0{num_qubits}b") for index in range(2 ** num_qubits)]


def _sorted_labels(labels: Sequence[str]) -> list[str]:
    """Sort bitstrings numerically when possible, otherwise lexicographically."""
    if all(label and set(label) <= {"0", "1"} for label in labels):
        return sorted(labels, key=lambda label: (len(label), int(label, 2)))
    return sorted(labels)


def _annotate_bars(ax: Axes, bars, labels: Sequence[str]) -> None:
    """Write compact value labels above a short bar chart."""
    offset = 0.02 * max((bar.get_height() for bar in bars), default=1.0)
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _get_2d_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Create or reuse a 2D Matplotlib axes."""
    if ax is not None:
        return ax.figure, ax
    plt = _require_matplotlib()
    fig, new_ax = plt.subplots(figsize=(7, 4))
    return fig, new_ax


def _get_3d_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Create or reuse a 3D Matplotlib axes."""
    if ax is not None:
        return ax.figure, ax

    plt = _require_matplotlib()
    fig = plt.figure(figsize=(6, 6))
    new_ax = fig.add_subplot(111, projection="3d")
    return fig, new_ax


def _draw_bloch_sphere(ax: Axes) -> None:
    """Draw a lightweight Bloch sphere scaffold."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 60)
    polar = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))

    ax.plot_wireframe(x, y, z, color="#C8D5B9", linewidth=0.6, alpha=0.35)

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    zeros = np.zeros_like(theta)
    ax.plot(np.cos(theta), np.sin(theta), zeros, color="#999999", alpha=0.5)
    ax.plot(np.cos(theta), zeros, np.sin(theta), color="#999999", alpha=0.25)
    ax.plot(zeros, np.cos(theta), np.sin(theta), color="#999999", alpha=0.25)

    ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=1.0, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=1.0, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=1.0, alpha=0.5)

    ax.text(1.08, 0.0, 0.0, "|+>")
    ax.text(-1.18, 0.0, 0.0, "|->")
    ax.text(0.0, 1.08, 0.0, "|+i>")
    ax.text(0.0, -1.18, 0.0, "|-i>")
    ax.text(0.0, 0.0, 1.1, "|0>")
    ax.text(0.0, 0.0, -1.18, "|1>")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.view_init(elev=20, azim=35)
    ax.grid(False)


def _require_matplotlib():
    """Import Matplotlib only when a plot is actually requested."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install it with "
            "'pip install matplotlib' or from requirements.txt."
        ) from exc
    return plt


__all__ = [
    "state_to_density_matrix",
    "single_qubit_reduced_density_matrix",
    "bloch_vector_from_density_matrix",
    "bloch_vector_from_state",
    "plot_counts",
    "plot_probabilities",
    "plot_bloch_from_state",
]
