"""Tests for plotting helpers and Bloch-vector math."""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from simulator.circuit import Circuit, Gate
from simulator.engine import DensityMatrixSimulator, StatevectorSimulator
from simulator.visualization import (
    bloch_vector_from_density_matrix,
    bloch_vector_from_state,
    plot_bloch_from_state,
    plot_counts,
    plot_probabilities,
    single_qubit_reduced_density_matrix,
    state_to_density_matrix,
)


def _make_circuit(num_qubits: int) -> Circuit:
    return Circuit(num_qubits=num_qubits, num_clbits=0)


def test_state_to_density_matrix_from_statevector():
    """A pure statevector should become |psi><psi|."""
    statevector = np.array([1.0, 1.0j], dtype=complex) / math.sqrt(2)
    density_matrix = state_to_density_matrix(statevector)
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)
    np.testing.assert_allclose(density_matrix, expected)


def test_single_qubit_reduction_of_bell_state_is_maximally_mixed():
    """Tracing out either qubit of a Bell state leaves I/2."""
    circuit = _make_circuit(2)
    circuit.add_gate(Gate(name="h", qubits=(0,)))
    circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
    result = StatevectorSimulator(circuit).run()

    reduced = single_qubit_reduced_density_matrix(result.statevector, qubit=0)
    np.testing.assert_allclose(reduced, np.eye(2, dtype=complex) / 2)


def test_bloch_vector_for_plus_state_points_along_x():
    """|+> should land on the +X axis of the Bloch sphere."""
    statevector = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
    bloch = bloch_vector_from_state(statevector)
    np.testing.assert_allclose(bloch, np.array([1.0, 0.0, 0.0]))


def test_bloch_vector_from_density_matrix_for_zero_state():
    """|0><0| should map to the north pole."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    bloch = bloch_vector_from_density_matrix(rho)
    np.testing.assert_allclose(bloch, np.array([0.0, 0.0, 1.0]))


def test_plot_counts_returns_axes_with_expected_bars():
    """Counts histogram should preserve one bar per measured outcome."""
    fig, ax = plot_counts({"00": 10, "11": 14})
    try:
        assert len(ax.patches) == 2
        assert ax.get_ylabel() == "Counts"
    finally:
        plt.close(fig)


def test_plot_probabilities_accepts_density_matrix_result():
    """Probability plotting should work for noisy density-matrix outputs too."""
    circuit = _make_circuit(1)
    circuit.add_gate(Gate(name="x", qubits=(0,)))
    result = DensityMatrixSimulator(circuit).run()

    fig, ax = plot_probabilities(result)
    try:
        heights = [patch.get_height() for patch in ax.patches]
        np.testing.assert_allclose(heights, [0.0, 1.0])
    finally:
        plt.close(fig)


def test_plot_bloch_from_state_accepts_reduced_multi_qubit_state():
    """Multi-qubit pure states should reduce cleanly to a single-qubit Bloch plot."""
    circuit = _make_circuit(2)
    circuit.add_gate(Gate(name="h", qubits=(0,)))
    circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
    result = StatevectorSimulator(circuit).run()

    fig, ax = plot_bloch_from_state(result, qubit=1)
    try:
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_zlabel() == "Z"
    finally:
        plt.close(fig)
