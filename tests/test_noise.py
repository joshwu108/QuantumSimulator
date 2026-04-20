"""Tests for noisy quantum simulation using the density matrix formalism.
"""

import math

import numpy as np
import pytest

from simulator.circuit import Circuit, Gate, Measurement
from simulator.noise import (AmplitudeDampingChannel, DepolarizingChannel, NoiseModel)
from simulator.engine import (DensityMatrixResult, DensityMatrixSimulator, StatevectorSimulator, apply_kraus_to_dm, apply_op_to_dm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _circuit(n_qubits: int, n_clbits: int = 0) -> Circuit:
    return Circuit(num_qubits=n_qubits, num_clbits=n_clbits)


# ---------------------------------------------------------------------------
# 1. Kraus operator properties
# ---------------------------------------------------------------------------
class TestKrausOperators:
    def test_depolarizing_kraus_completeness_small_p(self):
        """For p=0.1: Σ K_i† K_i must equal I (2×2)."""
        channel = DepolarizingChannel(p=0.1)
        ops = channel.kraus_operators()
        total = sum(K.conj().T @ K for K in ops)
        np.testing.assert_array_almost_equal(total, np.eye(2))

    def test_depolarizing_kraus_completeness_max_p(self):
        """For p=0.75 (maximally depolarizing): Σ K_i† K_i = I."""
        channel = DepolarizingChannel(p=0.75)
        ops = channel.kraus_operators()
        total = sum(K.conj().T @ K for K in ops)
        np.testing.assert_array_almost_equal(total, np.eye(2))

    def test_amplitude_damping_kraus_completeness(self):
        """For gamma=0.5: Σ K_i† K_i = I."""
        channel = AmplitudeDampingChannel(gamma=0.5)
        ops = channel.kraus_operators()
        total = sum(K.conj().T @ K for K in ops)
        np.testing.assert_array_almost_equal(total, np.eye(2))

    def test_amplitude_damping_kraus_completeness_full(self):
        """For gamma=1.0: Σ K_i† K_i = I."""
        channel = AmplitudeDampingChannel(gamma=1.0)
        ops = channel.kraus_operators()
        total = sum(K.conj().T @ K for K in ops)
        np.testing.assert_array_almost_equal(total, np.eye(2))

    def test_depolarizing_four_kraus_operators(self):
        """Depolarizing channel has exactly 4 Kraus operators (I, X, Y, Z)."""
        channel = DepolarizingChannel(p=0.2)
        assert len(channel.kraus_operators()) == 4

    def test_amplitude_damping_two_kraus_operators(self):
        """Amplitude damping channel has exactly 2 Kraus operators (K0, K1)."""
        channel = AmplitudeDampingChannel(gamma=0.3)
        assert len(channel.kraus_operators()) == 2

    def test_depolarizing_p_zero_equivalent_to_identity(self):
        """p=0 depolarizing: E(ρ) = ρ for any ρ.

        The channel simplifies to K0=I, K1=K2=K3=0.
        Applying it should leave any density matrix unchanged.
        """
        channel = DepolarizingChannel(p=0.0)
        rho = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        np.testing.assert_array_almost_equal(result, rho)

    def test_amplitude_damping_gamma_zero_equivalent_to_identity(self):
        """gamma=0 amplitude damping: E(ρ) = ρ."""
        channel = AmplitudeDampingChannel(gamma=0.0)
        rho = np.array([[0.3, 0.4], [0.4, 0.7]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        np.testing.assert_array_almost_equal(result, rho)

    def test_depolarizing_invalid_p_too_large(self):
        """p > 0.75 is unphysical and must raise ValueError."""
        with pytest.raises(ValueError):
            DepolarizingChannel(p=0.8)

    def test_depolarizing_invalid_p_negative(self):
        """Negative p is unphysical."""
        with pytest.raises(ValueError):
            DepolarizingChannel(p=-0.1)

    def test_amplitude_damping_invalid_gamma_too_large(self):
        """gamma > 1.0 is unphysical."""
        with pytest.raises(ValueError):
            AmplitudeDampingChannel(gamma=1.1)

    def test_amplitude_damping_invalid_gamma_negative(self):
        """Negative gamma is unphysical."""
        with pytest.raises(ValueError):
            AmplitudeDampingChannel(gamma=-0.01)


# ---------------------------------------------------------------------------
# 2. NoiseModel configuration
# ---------------------------------------------------------------------------

class TestNoiseModel:
    """Tests for NoiseModel configuration API."""

    def test_empty_model_is_noiseless(self):
        """A freshly created NoiseModel has no channels configured."""
        model = NoiseModel()
        assert model.is_noiseless()

    def test_add_global_noise_makes_model_noisy(self):
        """After add_all_gates_noise, model is no longer noiseless."""
        model = NoiseModel()
        model.add_all_gates_noise(DepolarizingChannel(p=0.01))
        assert not model.is_noiseless()

    def test_add_gate_specific_noise_makes_model_noisy(self):
        """After add_gate_noise, model is no longer noiseless."""
        model = NoiseModel()
        model.add_gate_noise("x", qubit=0, channel=DepolarizingChannel(p=0.05))
        assert not model.is_noiseless()

    def test_global_noise_returned_for_any_gate(self):
        """Global noise channel appears for every gate/qubit query."""
        ch = DepolarizingChannel(p=0.02)
        model = NoiseModel()
        model.add_all_gates_noise(ch)
        assert ch in model.get_channels_for_gate("x", qubit=0)
        assert ch in model.get_channels_for_gate("h", qubit=1)
        assert ch in model.get_channels_for_gate("cx", qubit=0)

    def test_gate_specific_noise_only_for_matching_gate(self):
        """Gate-specific noise only appears when gate name and qubit match."""
        ch = DepolarizingChannel(p=0.05)
        model = NoiseModel()
        model.add_gate_noise("x", qubit=0, channel=ch)

        # Should appear for (x, 0)
        assert ch in model.get_channels_for_gate("x", qubit=0)
        # Should NOT appear for (h, 0) or (x, 1)
        assert ch not in model.get_channels_for_gate("h", qubit=0)
        assert ch not in model.get_channels_for_gate("x", qubit=1)

    def test_no_channels_for_unregistered_gate(self):
        """A gate with no registered noise returns an empty channel list."""
        model = NoiseModel()
        model.add_gate_noise("x", qubit=0, channel=DepolarizingChannel(p=0.1))
        assert model.get_channels_for_gate("h", qubit=0) == []

    def test_global_and_gate_specific_combined(self):
        """Both global and gate-specific channels appear together."""
        global_ch = DepolarizingChannel(p=0.01)
        specific_ch = AmplitudeDampingChannel(gamma=0.05)
        model = NoiseModel()
        model.add_all_gates_noise(global_ch)
        model.add_gate_noise("x", qubit=0, channel=specific_ch)

        channels = model.get_channels_for_gate("x", qubit=0)
        assert global_ch in channels
        assert specific_ch in channels

    def test_multiple_global_channels_accumulated(self):
        """Multiple add_all_gates_noise calls all accumulate."""
        ch1 = DepolarizingChannel(p=0.01)
        ch2 = AmplitudeDampingChannel(gamma=0.02)
        model = NoiseModel()
        model.add_all_gates_noise(ch1)
        model.add_all_gates_noise(ch2)
        channels = model.get_channels_for_gate("h", qubit=0)
        assert ch1 in channels
        assert ch2 in channels


# ---------------------------------------------------------------------------
# 3. Density matrix helper functions
# ---------------------------------------------------------------------------

class TestApplyOpToDm:
    """Unit tests for apply_op_to_dm and apply_kraus_to_dm."""

    def test_identity_op_leaves_dm_unchanged(self):
        """Applying identity M to a density matrix returns the same matrix."""
        rho = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
        I = np.eye(2, dtype=complex)
        result = apply_op_to_dm(rho, I, qubit=0, n=1)
        np.testing.assert_array_almost_equal(result, rho)

    def test_x_op_flips_excited_state(self):
        """Applying X: |1><1| → X|1><1|X† = |0><0|."""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)   # |1><1|
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_op_to_dm(rho, X, qubit=0, n=1)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_op_preserves_trace(self):
        """Any unitary applied via apply_op_to_dm preserves the trace."""
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        result = apply_op_to_dm(rho, H, qubit=0, n=1)
        assert math.isclose(np.trace(result).real, 1.0, abs_tol=1e-10)

    def test_apply_op_to_qubit_0_in_two_qubit_system(self):
        """apply_op_to_dm on qubit 0 of a 2-qubit system: X flips |00><00| to |10><10|.

        |00> has index 0 (qubit 0=0, qubit 1=0).
        After X on qubit 0: |10> has index 1 (qubit 0=1, qubit 1=0).
        """
        n = 2
        # rho = |00><00|
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_op_to_dm(rho, X, qubit=0, n=n)
        # Expected: |10><10| = rho[1,1] = 1
        expected = np.zeros((4, 4), dtype=complex)
        expected[1, 1] = 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_op_to_qubit_1_in_two_qubit_system(self):
        """apply_op_to_dm on qubit 1 of a 2-qubit system: X flips |00><00| to |01><01|.

        After X on qubit 1: |01> has index 2 (qubit 0=0, qubit 1=1).
        """
        n = 2
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        result = apply_op_to_dm(rho, X, qubit=1, n=n)
        expected = np.zeros((4, 4), dtype=complex)
        expected[2, 2] = 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_kraus_trace_preserving(self):
        """Noise channel E(ρ) = Σ K_i ρ K_i† must preserve trace."""
        channel = DepolarizingChannel(p=0.3)
        rho = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        assert math.isclose(np.trace(result).real, 1.0, abs_tol=1e-10)

    def test_kraus_full_depolarizing_on_excited_state_gives_mixed(self):
        """Depolarizing p=0.75 on |1><1| gives the maximally mixed state I/2."""
        channel = DepolarizingChannel(p=0.75)
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_kraus_full_depolarizing_on_ground_state_gives_mixed(self):
        """Depolarizing p=0.75 on |0><0| also gives the maximally mixed state I/2."""
        channel = DepolarizingChannel(p=0.75)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_amplitude_damping_gamma_one_collapses_to_ground(self):
        """Amplitude damping γ=1 on |1><1| completely damps to |0><0|."""
        channel = AmplitudeDampingChannel(gamma=1.0)
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)

    def test_amplitude_damping_half_gamma_on_excited_state(self):
        """Amplitude damping γ=0.5 on |1><1|: ρ' = [[0.5,0],[0,0.5]]."""
        channel = AmplitudeDampingChannel(gamma=0.5)
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected)

    def test_amplitude_damping_does_not_affect_ground_state(self):
        """Amplitude damping on |0><0| leaves it unchanged (|0> is steady state)."""
        channel = AmplitudeDampingChannel(gamma=0.8)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit=0, n=1)
        np.testing.assert_array_almost_equal(result, rho)


# ---------------------------------------------------------------------------
# 4. DensityMatrixSimulator — noiseless mode
# ---------------------------------------------------------------------------

class TestDensityMatrixSimulatorNoiseless:
    """Without noise, DensityMatrixSimulator must match StatevectorSimulator."""

    def test_initial_state_is_pure_ground_density_matrix(self):
        """Initial ρ = |0><0|: only ρ[0,0] = 1, all others zero."""
        circuit = _circuit(2)
        result = DensityMatrixSimulator(circuit).run()
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_dm_trace_is_one(self):
        """Trace of initial density matrix equals 1."""
        circuit = _circuit(3)
        result = DensityMatrixSimulator(circuit).run()
        assert math.isclose(np.trace(result.density_matrix).real, 1.0, abs_tol=1e-10)

    def test_noiseless_x_gate_gives_excited_state(self):
        """X|0> → ρ = |1><1|: ρ[1,1] = 1."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        result = DensityMatrixSimulator(circuit).run()
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_noiseless_h_gate_gives_superposition(self):
        """H|0> → ρ = |+><+| = [[0.5, 0.5], [0.5, 0.5]]."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        result = DensityMatrixSimulator(circuit).run()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_probabilities_match_statevector_simulator(self):
        """DM simulator probabilities match statevector simulator for Bell state."""
        circuit = _circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))

        sv_probs = StatevectorSimulator(circuit).run().probabilities
        dm_probs = DensityMatrixSimulator(circuit).run().probabilities

        np.testing.assert_array_almost_equal(dm_probs, sv_probs)

    def test_probabilities_sum_to_one(self):
        """DM simulator probabilities always sum to 1."""
        circuit = _circuit(2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        result = DensityMatrixSimulator(circuit).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_noiseless_dm_counts_match_statevector_for_bell_state(self):
        """Bell state counts from DM simulator contain only '00' and '11'."""
        circuit = _circuit(2, n_clbits=2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        circuit.add_measurement(Measurement(qubit=1, clbit=1))

        counts = DensityMatrixSimulator(circuit).run().get_counts(shots=1000, seed=42)
        assert set(counts.keys()).issubset({"00", "11"})
        assert sum(counts.values()) == 1000

    def test_x_gate_noiseless_always_measures_one(self):
        """Without noise, X|0> always measures '1'."""
        circuit = _circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))
        counts = DensityMatrixSimulator(circuit).run().get_counts(shots=100)
        assert counts.get("1", 0) == 100
        assert "0" not in counts

    def test_none_noise_model_behaves_like_noiseless(self):
        """Passing noise_model=None is identical to no noise model."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        r1 = DensityMatrixSimulator(circuit, noise_model=None).run()
        r2 = DensityMatrixSimulator(circuit).run()
        np.testing.assert_array_almost_equal(r1.density_matrix, r2.density_matrix)

    def test_empty_noise_model_behaves_like_noiseless(self):
        """An empty NoiseModel (no channels added) must not change results."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        r_noiseless = DensityMatrixSimulator(circuit).run()
        r_empty = DensityMatrixSimulator(circuit, noise_model=NoiseModel()).run()
        np.testing.assert_array_almost_equal(
            r_noiseless.density_matrix, r_empty.density_matrix
        )


# ---------------------------------------------------------------------------
# 5. Depolarizing noise effects
# ---------------------------------------------------------------------------

class TestDepolarizingNoise:
    """Tests that depolarizing noise makes outcomes more mixed."""

    def test_zero_p_is_same_as_noiseless(self):
        """DepolarizingChannel(p=0) must produce identical results to noiseless."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.0))

        rho_noisy = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        rho_clean = DensityMatrixSimulator(circuit).run()
        np.testing.assert_array_almost_equal(
            rho_noisy.density_matrix, rho_clean.density_matrix
        )

    def test_max_depolarizing_on_excited_state_gives_fifty_fifty(self):
        """X followed by p=0.75 depolarizing: P(0)=P(1)=0.5.

        Mathematical derivation: ρ_initial = |1><1|
        E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ) = I/2 when p=0.75.
        """
        circuit = _circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.75))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        probs = result.probabilities
        assert math.isclose(probs[0], 0.5, abs_tol=1e-10)  # P(|0>)
        assert math.isclose(probs[1], 0.5, abs_tol=1e-10)  # P(|1>)

    def test_depolarizing_reduces_excited_state_probability(self):
        """Depolarizing p=0.3 on |1>: P(1) = 1 - 2p/3 ≈ 0.8 (not 1.0)."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.3))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        probs = result.probabilities
        # P(0) = 2p/3 = 0.2, P(1) = 1 - 2p/3 = 0.8
        assert math.isclose(probs[0], 2 * 0.3 / 3, abs_tol=1e-10)
        assert math.isclose(probs[1], 1 - 2 * 0.3 / 3, abs_tol=1e-10)

    def test_depolarizing_makes_counts_more_mixed(self):
        """Strong depolarizing noise causes |1> to sometimes measure as |0>.

        Without noise: X|0> is deterministically |1>.
        With p=0.75: ~50% zeros appear in counts.
        """
        circuit = _circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))

        # Noiseless: always "1"
        noiseless_counts = DensityMatrixSimulator(circuit).run().get_counts(
            shots=1000, seed=42
        )
        assert noiseless_counts.get("0", 0) == 0

        # Noisy: "0" should appear frequently
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.75))
        noisy_counts = DensityMatrixSimulator(circuit, noise_model=noise_model).run().get_counts(
            shots=1000, seed=42
        )
        assert noisy_counts.get("0", 0) > 200  # at least 20% zeros

    def test_depolarizing_probabilities_sum_to_one(self):
        """Probabilities after depolarizing noise still sum to 1."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.2))
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_depolarizing_trace_preserved(self):
        """Trace of density matrix remains 1 after depolarizing noise."""
        circuit = _circuit(2)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.05))
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        assert math.isclose(np.trace(result.density_matrix).real, 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# 6. Amplitude damping effects
# ---------------------------------------------------------------------------

class TestAmplitudeDampingNoise:
    """Tests that amplitude damping pushes |1> toward |0>."""

    def test_zero_gamma_is_same_as_noiseless(self):
        """AmplitudeDampingChannel(gamma=0) produces identical results to noiseless."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=0.0))

        rho_noisy = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        rho_clean = DensityMatrixSimulator(circuit).run()
        np.testing.assert_array_almost_equal(
            rho_noisy.density_matrix, rho_clean.density_matrix
        )

    def test_full_damping_collapses_one_to_zero(self):
        """gamma=1.0 after X gate: |1> fully damps to |0>; P(0)=1, P(1)=0."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=1.0))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        probs = result.probabilities
        assert math.isclose(probs[1], 0.0, abs_tol=1e-10)  # |1> gone
        assert math.isclose(probs[0], 1.0, abs_tol=1e-10)  # all in |0>

    def test_partial_damping_reduces_excited_state_probability(self):
        """gamma=0.5 after X gate: P(1) = 1-gamma = 0.5, P(0) = gamma = 0.5."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=0.5))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        probs = result.probabilities
        assert math.isclose(probs[0], 0.5, abs_tol=1e-10)  # P(|0>) = gamma
        assert math.isclose(probs[1], 0.5, abs_tol=1e-10)  # P(|1>) = 1-gamma

    def test_amplitude_damping_does_not_affect_ground_state(self):
        """|0> is the steady state: amplitude damping on |0> leaves P(0)=1."""
        circuit = _circuit(1, n_clbits=1)
        # Start in |0> (no X gate), apply damping
        circuit.add_measurement(Measurement(qubit=0, clbit=0))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=0.9))

        # No gates, so noise model doesn't fire (no gate to trigger noise after)
        # The DM result should still be |0><0|
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        assert math.isclose(result.probabilities[0], 1.0, abs_tol=1e-10)

    def test_amplitude_damping_pushes_one_toward_zero_in_counts(self):
        """With gamma=1.0, all 1000 shots after X should measure '0', not '1'.

        This directly validates the requirement: amplitude damping pushes |1> → |0>.
        """
        circuit = _circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))

        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=1.0))

        counts = DensityMatrixSimulator(circuit, noise_model=noise_model).run().get_counts(
            shots=1000, seed=42
        )
        assert counts.get("0", 0) == 1000
        assert counts.get("1", 0) == 0

    def test_amplitude_damping_probabilities_sum_to_one(self):
        """Probabilities after amplitude damping noise still sum to 1."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=0.3))
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        assert math.isclose(result.probabilities.sum(), 1.0, abs_tol=1e-10)

    def test_amplitude_damping_partial_gamma_matches_formula(self):
        """gamma=0.7 after X: P(1) = 1-0.7 = 0.3, P(0) = 0.7 (exact formula)."""
        gamma = 0.7
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(AmplitudeDampingChannel(gamma=gamma))
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        probs = result.probabilities
        assert math.isclose(probs[0], gamma, abs_tol=1e-10)
        assert math.isclose(probs[1], 1 - gamma, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# 7. Noise insertion configuration
# ---------------------------------------------------------------------------

class TestNoiseInsertion:
    """Tests for configuring where noise is inserted."""

    def test_global_noise_applied_after_every_gate(self):
        """add_all_gates_noise applies depolarizing after H and after X.

        Circuit: H on q0 then X on q0.
        Without noise: H X|0> = H|1> = |-> = [[0.5,-0.5],[-0.5,0.5]].
        With strong global noise: output is more mixed.
        """
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        # Noise after every gate (applied twice: after H, after X)
        noise_model = NoiseModel()
        noise_model.add_all_gates_noise(DepolarizingChannel(p=0.75))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        # Applying p=0.75 depolarizing twice:
        # First application turns any state into I/2.
        # Second application on I/2 keeps I/2.
        # Expected: I/2 = [[0.5, 0], [0, 0.5]]
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_gate_specific_noise_only_on_matching_gates(self):
        """add_gate_noise("x", ...) only adds noise after X gates, not H gates.

        Circuit: H on q0 then X on q0.
        Noise only on X: only the second gate triggers noise.
        """
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))  # no noise after this
        circuit.add_gate(Gate(name="x", qubits=(0,)))  # noise after this

        noise_model = NoiseModel()
        noise_model.add_gate_noise("x", qubit=0, channel=DepolarizingChannel(p=0.75))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        # After H: |+> (pure, no noise)
        # After X on |+>: |-> = [[0.5,-0.5],[-0.5,0.5]]
        # After depolarizing p=0.75 on |-><-|: I/2
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_noise_only_on_h_gate_not_x_gate(self):
        """add_gate_noise("h", ...) applies noise after H but not after X.

        Circuit: H then X.
        After H + noise: I/2.
        After X (no noise): X (I/2) X† = I/2 (mixed state unchanged by X).
        Result: still I/2.
        """
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="h", qubits=(0,)))
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        noise_model = NoiseModel()
        noise_model.add_gate_noise("h", qubit=0, channel=DepolarizingChannel(p=0.75))

        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()
        # H noise makes I/2, then X rotates but I/2 is invariant under unitaries
        # (X (I/2) X† = I/2 since I/2 commutes with everything)
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_array_almost_equal(result.density_matrix, expected)

    def test_noiseless_dm_sim_without_noise_model(self):
        """DensityMatrixSimulator() without noise_model is fully noiseless."""
        circuit = _circuit(1, n_clbits=1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))
        circuit.add_measurement(Measurement(qubit=0, clbit=0))

        counts = DensityMatrixSimulator(circuit).run().get_counts(shots=100)
        # Noiseless X|0>: always measures "1"
        assert counts == {"1": 100}

    def test_two_noise_models_can_be_compared(self):
        """Switching noise_model between runs should produce different results."""
        circuit = _circuit(1)
        circuit.add_gate(Gate(name="x", qubits=(0,)))

        nm_depol = NoiseModel()
        nm_depol.add_all_gates_noise(DepolarizingChannel(p=0.5))

        nm_damp = NoiseModel()
        nm_damp.add_all_gates_noise(AmplitudeDampingChannel(gamma=0.5))

        r_depol = DensityMatrixSimulator(circuit, noise_model=nm_depol).run()
        r_damp = DensityMatrixSimulator(circuit, noise_model=nm_damp).run()

        # Both should differ from noiseless
        r_clean = DensityMatrixSimulator(circuit).run()
        assert not np.allclose(r_depol.density_matrix, r_clean.density_matrix)
        assert not np.allclose(r_damp.density_matrix, r_clean.density_matrix)
