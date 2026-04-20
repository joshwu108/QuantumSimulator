"""Unit tests for the noiseless statevector simulator engine.
"""

import math

import numpy as np
import pytest

from simulator.circuit import Circuit, Gate, Measurement
from simulator.engine import (
    StatevectorSimulator,
    SimulationResult,
    apply_single_qubit_gate,
    apply_controlled_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circuit(n_qubits: int, n_clbits: int = 0) -> Circuit:
    return Circuit(num_qubits=n_qubits, num_clbits=n_clbits)


# ---------------------------------------------------------------------------
# 1. Statevector initialization
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_initial_state_is_all_zeros_ground_state(self):
        """Fresh simulator starts in |0...0>: amplitude 1 at index 0."""
        circuit = make_circuit(3)
        result = StatevectorSimulator(circuit).run()
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_statevector_size_is_2_to_n(self):
        for n in [1, 2, 3, 4]:
            circuit = make_circuit(n)
            result = StatevectorSimulator(circuit).run()
            assert len(result.statevector) == 2 ** n

    def test_initial_probabilities_sum_to_one(self):
        circuit = make_circuit(4)
        result = StatevectorSimulator(circuit).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# 2. Hadamard gate
# ---------------------------------------------------------------------------
class TestHadamardGate:
    def test_h_on_zero_produces_equal_superposition(self):
        """H|0> = (|0> + |1>) / sqrt(2)."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        expected = np.array([1, 1], dtype=complex) / math.sqrt(2)
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_h_on_one_produces_minus_superposition(self):
        """H|1> = (|0> - |1>) / sqrt(2)."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        expected = np.array([1, -1], dtype=complex) / math.sqrt(2)
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_h_squared_is_identity(self):
        """H² = I: applying Hadamard twice returns to |0>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [1, 0])

    def test_h_on_second_qubit_does_not_affect_first(self):
        """H applied only to qubit 1 leaves qubit 0 unchanged."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="h", qubits=(1,)))
        result = StatevectorSimulator(circuit).run()
        # |0> ⊗ (|0>+|1>)/√2 = (|00> + |10>)/√2
        # Little-endian indices: |q1=0,q0=0>=0, |q1=1,q0=0>=2
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / math.sqrt(2)  # |q1=0, q0=0>
        expected[2] = 1 / math.sqrt(2)  # |q1=1, q0=0>
        np.testing.assert_array_almost_equal(result.statevector, expected)


# ---------------------------------------------------------------------------
# 3. Bell state creation
# ---------------------------------------------------------------------------
class TestBellState:
    def test_bell_state_statevector(self):
        """Bell state |Φ+> = (|00> + |11>) / sqrt(2).

        Qubit ordering (little-endian / Qiskit convention):
          - Statevector index i encodes qubit k as bit k: state_k = (i >> k) & 1
          - |00> = index 0, |11> = index 3 (binary 11 = q0=1, q1=1)
        """
        circuit = make_circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        result = StatevectorSimulator(circuit).run()

        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / math.sqrt(2)  # |q1=0, q0=0> = |00>
        expected[3] = 1 / math.sqrt(2)  # |q1=1, q0=1> = |11>
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_bell_state_probabilities_are_fifty_fifty(self):
        """Bell state: P(|00>) = P(|11>) = 0.5, all others zero."""
        circuit = make_circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        result = StatevectorSimulator(circuit).run()

        probs = result.probabilities
        assert math.isclose(probs[0], 0.5, abs_tol=1e-10)  # |00>
        assert math.isclose(probs[3], 0.5, abs_tol=1e-10)  # |11>
        assert math.isclose(probs[1], 0.0, abs_tol=1e-10)  # |01>
        assert math.isclose(probs[2], 0.0, abs_tol=1e-10)  # |10>

    def test_bell_state_counts_are_balanced(self):
        """Bell state shot counts: only '00' and '11' appear, roughly 50/50.

        Bitstring format: c[n-1]...c[1]c[0] (MSB of classical register first).
        For 2 classical bits with c0 = meas(q0), c1 = meas(q1): key = f'{c1}{c0}'.
        """
        circuit = make_circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        result = StatevectorSimulator(circuit).run()

        counts = result.get_counts(shots=1000, seed=42)
        assert set(counts.keys()).issubset({"00", "11"})
        assert sum(counts.values()) == 1000
        assert abs(counts.get("00", 0) - 500) < 100
        assert abs(counts.get("11", 0) - 500) < 100


# ---------------------------------------------------------------------------
# 4. Deterministic basis-state outputs (X, Y, Z)
# ---------------------------------------------------------------------------

class TestDeterministicGates:
    def test_x_gate_flips_zero_to_one(self):
        """X|0> = |1>: amplitude is fully at index 1."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1])

    def test_x_gate_always_measures_one(self):
        """X|0> is deterministically |1>: 100 shots all produce '1'."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=100, seed=0)
        assert counts == {"1": 100}

    def test_x_x_is_identity(self):
        """X²|0> = |0>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [1, 0])

    def test_z_gate_on_zero_is_no_op(self):
        """Z|0> = |0>: Z only adds phase to |1>, leaves |0> unchanged."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="z", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [1, 0])

    def test_z_gate_flips_phase_of_one(self):
        """Z|1> = -|1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="z", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, -1])

    def test_y_gate_on_zero(self):
        """Y|0> = i|1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="y", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1j])

    def test_zero_state_always_measures_zero(self):
        """Unmodified |0> always measures 0."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=100, seed=0)
        assert counts == {"0": 100}


# ---------------------------------------------------------------------------
# 5. Parametric gates (Rx, Ry, Rz, U1, U2, U3)
# ---------------------------------------------------------------------------

class TestParametricGates:
    def test_rx_pi_rotates_zero_to_one(self):
        """Rx(π)|0> = -i|1>: full population transfer, no amplitude at |0>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="rx", qubits=(0,), params=(math.pi,)))
        result = StatevectorSimulator(circuit).run()
        assert abs(result.statevector[0]) < 1e-10
        assert math.isclose(abs(result.statevector[1]), 1.0, abs_tol=1e-10)

    def test_ry_pi_rotates_zero_to_one_real(self):
        """Ry(π)|0> = |1> exactly (no imaginary phase)."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="ry", qubits=(0,), params=(math.pi,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1])

    def test_rz_does_not_change_measurement_probabilities(self):
        """Rz only adds a relative phase; measurement probs remain unchanged."""
        circuit1 = make_circuit(1)
        circuit1.add_gate(Gate(name="h", qubits=(0,)))
        probs_before = StatevectorSimulator(circuit1).run().probabilities

        circuit2 = make_circuit(1)
        circuit2.add_gate(Gate(name="h", qubits=(0,)))
        circuit2.add_gate(Gate(name="rz", qubits=(0,), params=(math.pi / 4,)))
        probs_after = StatevectorSimulator(circuit2).run().probabilities

        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_u3_half_pi_zero_zero_gives_equal_superposition(self):
        """u3(π/2, 0, 0)|0> yields equal superposition like Hadamard."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="u3", qubits=(0,), params=(math.pi / 2, 0.0, 0.0)))
        probs = StatevectorSimulator(circuit).run().probabilities
        assert math.isclose(probs[0], 0.5, abs_tol=1e-10)
        assert math.isclose(probs[1], 0.5, abs_tol=1e-10)

    def test_u3_matches_hadamard_exactly(self):
        """u3(π/2, 0, π) == H exactly (standard OpenQASM 2.0 decomposition)."""
        circuit_h = make_circuit(1)
        circuit_h.add_gate(Gate(name="h", qubits=(0,)))
        sv_h = StatevectorSimulator(circuit_h).run().statevector

        circuit_u3 = make_circuit(1)
        circuit_u3.add_gate(Gate(name="u3", qubits=(0,), params=(math.pi / 2, 0.0, math.pi)))
        sv_u3 = StatevectorSimulator(circuit_u3).run().statevector

        np.testing.assert_array_almost_equal(sv_h, sv_u3)

    def test_u1_is_phase_only(self):
        """u1(π) applied to |0> leaves it unchanged (phase gate only)."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="u1", qubits=(0,), params=(math.pi,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [1, 0])

    def test_u1_pi_on_one_flips_phase(self):
        """u1(π)|1> = -|1>: equivalent to Z gate on |1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="u1", qubits=(0,), params=(math.pi,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, -1])

    def test_u2_zero_pi_matches_hadamard(self):
        """u2(0, π)|0> == H|0>."""
        circuit_h = make_circuit(1)
        circuit_h.add_gate(Gate(name="h", qubits=(0,)))
        sv_h = StatevectorSimulator(circuit_h).run().statevector

        circuit_u2 = make_circuit(1)
        circuit_u2.add_gate(Gate(name="u2", qubits=(0,), params=(0.0, math.pi)))
        sv_u2 = StatevectorSimulator(circuit_u2).run().statevector

        np.testing.assert_array_almost_equal(sv_h, sv_u2)


# ---------------------------------------------------------------------------
# 6. Phase gates (S, Sdg, T, Tdg)
# ---------------------------------------------------------------------------

class TestPhaseGates:
    def test_s_gate_on_zero_is_no_op(self):
        """S|0> = |0>: phase gate doesn't affect |0>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="s", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [1, 0])

    def test_s_gate_adds_i_phase_to_one(self):
        """S|1> = i|1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="s", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1j])

    def test_sdg_gate_adds_minus_i_phase_to_one(self):
        """Sdg|1> = -i|1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="sdg", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, -1j])

    def test_s_sdg_cancel(self):
        """S then Sdg returns to original state."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="s", qubits=(0,)))
        circuit.add_gate(Gate(name="sdg", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1])

    def test_t_gate_adds_pi_over_4_phase_to_one(self):
        """T|1> = e^(iπ/4)|1>."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="t", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        expected = np.array([0, np.exp(1j * math.pi / 4)], dtype=complex)
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_t_tdg_cancel(self):
        """T then Tdg cancels out."""
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_gate(Gate(name="t", qubits=(0,)))
        circuit.add_gate(Gate(name="tdg", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        np.testing.assert_array_almost_equal(result.statevector, [0, 1])

    def test_s_squared_is_z(self):
        """S² = Z: two S gates equivalent to Z gate."""
        circuit_ss = make_circuit(1)
        circuit_ss.add_gate(Gate(name="x", qubits=(0,)))
        circuit_ss.add_gate(Gate(name="s", qubits=(0,)))
        circuit_ss.add_gate(Gate(name="s", qubits=(0,)))
        sv_ss = StatevectorSimulator(circuit_ss).run().statevector

        circuit_z = make_circuit(1)
        circuit_z.add_gate(Gate(name="x", qubits=(0,)))
        circuit_z.add_gate(Gate(name="z", qubits=(0,)))
        sv_z = StatevectorSimulator(circuit_z).run().statevector

        np.testing.assert_array_almost_equal(sv_ss, sv_z)


# ---------------------------------------------------------------------------
# 7. Multi-qubit gates: CX, CZ, GHZ
# ---------------------------------------------------------------------------

class TestMultiQubitGates:
    def test_cx_flips_target_when_control_is_one(self):
        """CX: control=|1> causes target to flip. |10> -> |11> (little-endian)."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="x", qubits=(0,)))   # |10>: q0=1, q1=0
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))  # ctrl=q0, target=q1
        result = StatevectorSimulator(circuit).run()
        # |11> = index 3 (q0=1 → bit 0, q1=1 → bit 1, index = 1+2 = 3)
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1.0
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_cx_no_flip_when_control_is_zero(self):
        """CX: control=|0> leaves target unchanged."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        result = StatevectorSimulator(circuit).run()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0  # |00> unchanged
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_cz_flips_phase_of_11(self):
        """CZ|11> = -|11>: applies a phase flip to the |11> component."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="x", qubits=(0,)))   # q0 = |1>
        circuit.add_gate(Gate(name="x", qubits=(1,)))   # q1 = |1>
        circuit.add_gate(Gate(name="cz", qubits=(0, 1)))
        result = StatevectorSimulator(circuit).run()
        expected = np.zeros(4, dtype=complex)
        expected[3] = -1.0  # |11> with phase flip
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_cz_no_phase_on_non_11_states(self):
        """CZ leaves |01> unchanged (no phase flip when ctrl=|0>)."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="x", qubits=(1,)))   # q1 = |1>, q0 = |0>
        circuit.add_gate(Gate(name="cz", qubits=(0, 1)))
        result = StatevectorSimulator(circuit).run()
        # |01> = q0=0, q1=1 → index 2 (bit 1 = 1, bit 0 = 0)
        expected = np.zeros(4, dtype=complex)
        expected[2] = 1.0  # |01> unchanged
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_ghz_state_three_qubits(self):
        """GHZ = H q0; CX q0->q1; CX q1->q2 = (|000> + |111>) / sqrt(2)."""
        circuit = make_circuit(3, n_clbits=3)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_gate(Gate(name="cx", qubits=(1, 2)))
        result = StatevectorSimulator(circuit).run()

        expected = np.zeros(8, dtype=complex)
        # |000> = index 0, |111> = index 7 (q0=1 q1=1 q2=1 → 1+2+4 = 7)
        expected[0] = 1 / math.sqrt(2)
        expected[7] = 1 / math.sqrt(2)
        np.testing.assert_array_almost_equal(result.statevector, expected)

    def test_cx_reversed_control_target(self):
        """CX with reversed roles: ctrl=q1, target=q0."""
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="x", qubits=(1,)))    # q1 = |1>, q0 = |0>
        circuit.add_gate(Gate(name="cx", qubits=(1, 0)))  # ctrl=q1, target=q0
        result = StatevectorSimulator(circuit).run()
        # |01> (q0=0, q1=1) → target q0 flips → |11> = index 3
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1.0
        np.testing.assert_array_almost_equal(result.statevector, expected)


# ---------------------------------------------------------------------------
# 8. Probability normalization
# ---------------------------------------------------------------------------

class TestProbabilityNormalization:
    def test_probabilities_sum_to_one_after_single_gate(self):
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        result = StatevectorSimulator(circuit).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_probabilities_sum_to_one_after_entanglement(self):
        circuit = make_circuit(3)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_gate(Gate(name="cx", qubits=(1, 2)))
        result = StatevectorSimulator(circuit).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_probabilities_sum_to_one_after_parametric_gate(self):
        circuit = make_circuit(1)
        circuit.add_gate(Gate(name="rx", qubits=(0,), params=(1.23,)))
        result = StatevectorSimulator(circuit).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_probabilities_are_non_negative(self):
        circuit = make_circuit(2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        probs = StatevectorSimulator(circuit).run().probabilities
        assert np.all(probs >= 0)


# ---------------------------------------------------------------------------
# 9. Measurement counts and sampling
# ---------------------------------------------------------------------------

class TestMeasurementCounts:
    def test_hadamard_counts_within_tolerance(self):
        """H gate → ~50% |0>, ~50% |1> over 10 000 shots (< 3% error)."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=10_000, seed=42)

        total = sum(counts.values())
        assert total == 10_000
        assert abs(counts.get("0", 0) / total - 0.5) < 0.03
        assert abs(counts.get("1", 0) / total - 0.5) < 0.03

    def test_counts_sum_equals_shots(self):
        circuit = make_circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=500, seed=7)
        assert sum(counts.values()) == 500

    def test_x_gate_deterministic_one_output(self):
        """X|0> measured 50 times always returns '1'."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=50)
        assert counts.get("1", 0) == 50
        assert "0" not in counts

    def test_zero_state_deterministic_zero_output(self):
        """Unmodified |0> measured 50 times always returns '0'."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=50)
        assert counts == {"0": 50}

    def test_bell_state_only_correlated_outcomes(self):
        """Bell state: only '00' and '11' appear, never '01' or '10'."""
        circuit = make_circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=500, seed=99)
        assert "01" not in counts
        assert "10" not in counts

    def test_seed_reproducibility(self):
        """Same seed produces identical counts."""
        circuit = make_circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        result = StatevectorSimulator(circuit).run()
        counts1 = result.get_counts(shots=200, seed=123)
        counts2 = result.get_counts(shots=200, seed=123)
        assert counts1 == counts2

    def test_bitstring_length_equals_num_clbits(self):
        """Bitstring keys always have length == num_clbits."""
        circuit = make_circuit(3, n_clbits=3)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))
        circuit.add_measurement(Measurement(qubit=2, clbit=2))
        result = StatevectorSimulator(circuit).run()
        counts = result.get_counts(shots=100, seed=1)
        for key in counts:
            assert len(key) == 3


# ---------------------------------------------------------------------------
# 10. Helper function unit tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_apply_single_qubit_identity_unchanged(self):
        """Identity matrix leaves statevector unchanged."""
        n = 2
        sv = np.array([0.5, 0.5j, 0.5, -0.5j], dtype=complex)
        result = apply_single_qubit_gate(sv.copy(), np.eye(2, dtype=complex), qubit=0, n=n)
        np.testing.assert_array_almost_equal(result, sv)

    def test_apply_single_qubit_x_on_qubit_0(self):
        """X on qubit 0: |00> -> |10> (index 0 -> index 1)."""
        n = 2
        sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_single_qubit_gate(sv, x, qubit=0, n=n)
        expected = np.array([0, 1, 0, 0], dtype=complex)  # |10> = index 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_single_qubit_x_on_qubit_1(self):
        """X on qubit 1: |00> -> |01> (index 0 -> index 2)."""
        n = 2
        sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_single_qubit_gate(sv, x, qubit=1, n=n)
        expected = np.array([0, 0, 1, 0], dtype=complex)  # |01> = index 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_controlled_gate_ctrl_zero_no_change(self):
        """Controlled gate doesn't apply when control qubit is |0>."""
        n = 2
        sv = np.array([1, 0, 0, 0], dtype=complex)  # |00>, ctrl q0=0
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_controlled_gate(sv.copy(), x, ctrl=0, target=1, n=n)
        np.testing.assert_array_almost_equal(result, sv)

    def test_apply_controlled_gate_ctrl_one_flips_target(self):
        """CX flips target when control is |1>."""
        n = 2
        sv = np.array([0, 1, 0, 0], dtype=complex)  # |10>: q0=1, q1=0
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_controlled_gate(sv, x, ctrl=0, target=1, n=n)
        expected = np.array([0, 0, 0, 1], dtype=complex)  # |11> = index 3
        np.testing.assert_array_almost_equal(result, expected)
