"""Statevector and density matrix quantum circuit simulators.

Qubit ordering convention (little-endian / Qiskit-compatible):
  - Statevector index i encodes qubit k's state as bit k of i:
      state_of_qubit_k = (i >> k) & 1
  - Qubit 0 is the least-significant bit (LSB / rightmost).
  - |0...0> (all qubits ground) maps to index 0.
  - |1, 0, ..., 0> (only qubit 0 excited) maps to index 1.

  Example for 2 qubits:
    index 0 → |q1=0, q0=0> = |00>
    index 1 → |q1=0, q0=1> = |10>   (qubit 0 excited)
    index 2 → |q1=1, q0=0> = |01>   (qubit 1 excited)
    index 3 → |q1=1, q0=1> = |11>

Efficiency design:
  - Gates are applied without ever constructing a 2^n × 2^n unitary matrix.
  - Single-qubit gates: O(2^n) via tensor contraction on the reshaped statevector.
  - Controlled gates: O(2^n) via direct slice indexing into the tensor.
  - Memory: O(2^n) for the statevector (doubles when using density matrices for noise).

Extending to noisy simulation:
  - Replace the pure-state statevector with a density matrix rho (2^n × 2^n).
  - After each gate U, apply rho -> U rho U†.
  - Apply Kraus operators {K_i} for noise channels: rho -> sum_i K_i rho K_i†.
  - Measurement probabilities become Tr(rho |i><i|) instead of |<i|psi>|².
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulator.circuit import Circuit, Gate
from simulator.noise import NoiseModel


# ---------------------------------------------------------------------------
# Static gate matrices (2x2 complex unitary matrices in {|0>, |1>} basis)
# Gate convention: matrix[row, col] = <row|U|col>
# ---------------------------------------------------------------------------

_I   = np.eye(2, dtype=complex)

_X   = np.array([[0, 1],   [1,  0]],  dtype=complex)
_Y   = np.array([[0, -1j], [1j, 0]],  dtype=complex)
_Z   = np.array([[1, 0],   [0, -1]],  dtype=complex)

# H = (X + Z) / sqrt(2): equal superposition gate
_H   = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

# S = sqrt(Z): adds i phase to |1>
_S   = np.array([[1, 0], [0,  1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)   # S†

# T = sqrt(S) = 4th root of Z: adds e^(iπ/4) phase to |1>
_T   = np.array([[1, 0], [0, np.exp( 1j * math.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)   # T†


# ---------------------------------------------------------------------------
# Parametric gate constructors
# ---------------------------------------------------------------------------

def _rx(theta: float) -> np.ndarray:
    """Rotation around the X-axis by angle theta.
    Rx(θ) = exp(-iθX/2) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]].
    """
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    """Rotation around the Y-axis by angle theta.
    Ry(θ) = exp(-iθY/2) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]].
    """
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """Rotation around the Z-axis by angle theta.
    Rz(θ) = exp(-iθZ/2) = diag(e^(-iθ/2), e^(iθ/2)).
    Note: only adds a relative phase; measurement probabilities are unchanged.
    """
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


def _u1(lam: float) -> np.ndarray:
    """Phase gate u1(λ) = diag(1, e^(iλ)) per OpenQASM 2.0.
    Equivalent to Rz up to a global phase.
    """
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


def _u2(phi: float, lam: float) -> np.ndarray:
    """Single-pulse gate u2(φ, λ) per OpenQASM 2.0.
    u2(φ, λ) = 1/sqrt(2) * [[1, -e^(iλ)], [e^(iφ), e^(i(φ+λ))]].
    H = u2(0, π).
    """
    return np.array(
        [[1, -np.exp(1j * lam)], [np.exp(1j * phi), np.exp(1j * (phi + lam))]],
        dtype=complex,
    ) / math.sqrt(2)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary u3(θ, φ, λ) per OpenQASM 2.0.
    u3(θ, φ, λ) = [[cos(θ/2), -e^(iλ)·sin(θ/2)],
                    [e^(iφ)·sin(θ/2), e^(i(φ+λ))·cos(θ/2)]].
    H = u3(π/2, 0, π). Any single-qubit unitary is expressible as u3.
    """
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c,                      -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s,   np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


# Dispatch table for single-qubit gate names to matrix factories.
# Callable entries take `gate.params` as arguments.
_SINGLE_QUBIT_GATES: dict = {
    "x":   lambda _: _X,
    "y":   lambda _: _Y,
    "z":   lambda _: _Z,
    "h":   lambda _: _H,
    "s":   lambda _: _S,
    "sdg": lambda _: _SDG,
    "t":   lambda _: _T,
    "tdg": lambda _: _TDG,
    "id":  lambda _: _I,
    "rx":  lambda p: _rx(p[0]),
    "ry":  lambda p: _ry(p[0]),
    "rz":  lambda p: _rz(p[0]),
    "u1":  lambda p: _u1(p[0]),
    "u2":  lambda p: _u2(p[0], p[1]),
    "u3":  lambda p: _u3(p[0], p[1], p[2]),
}


def _gate_matrix(gate: Gate) -> np.ndarray:
    """Return the 2×2 unitary matrix for a single-qubit gate."""
    factory = _SINGLE_QUBIT_GATES.get(gate.name)
    if factory is None:
        raise ValueError(f"Unknown single-qubit gate: '{gate.name}'")
    return factory(gate.params)


# ---------------------------------------------------------------------------
# Statevector gate application (core computational routines)
# ---------------------------------------------------------------------------

def apply_single_qubit_gate(
    sv: np.ndarray, matrix: np.ndarray, qubit: int, n: int
) -> np.ndarray:
    """Apply a 2×2 unitary to one qubit of an n-qubit statevector.

    Algorithm (tensor contraction, no full 2^n×2^n matrix):
      1. Reshape sv from (2^n,) to (2, 2, ..., 2) [n axes].
      2. In C-order reshape, axis k corresponds to qubit (n-1-k).
         So qubit q is on axis (n-1-q).
      3. Contract gate[out, in] with sv along the axis for `qubit`.
      4. Move the output axis back to the correct position.
      5. Reshape back to (2^n,).

    Complexity: O(2^n) time and O(2^n) space — no 4^n matrix ever built.

    Args:
        sv:     Statevector of shape (2^n,).
        matrix: 2×2 unitary gate matrix.
        qubit:  Target qubit index (0 = LSB).
        n:      Total number of qubits.

    Returns:
        New statevector of shape (2^n,).
    """
    sv = sv.reshape([2] * n)
    # C-order (row-major) reshape: axis 0 is MSB = qubit (n-1),
    # last axis is LSB = qubit 0.  Qubit q → axis (n-1-q).
    axis = n - 1 - qubit
    # tensordot contracts matrix's axis 1 ("input" index) with sv's `axis`
    sv = np.tensordot(matrix, sv, axes=([1], [axis]))
    # tensordot places the new "output" axis at position 0; move it back.
    sv = np.moveaxis(sv, 0, axis)
    return sv.reshape(-1)


def apply_controlled_gate(
    sv: np.ndarray, matrix: np.ndarray, ctrl: int, target: int, n: int
) -> np.ndarray:
    """Apply a 2×2 unitary to `target` conditioned on `ctrl` being |1>.

    Algorithm (direct slice indexing, no 4^n controlled-unitary matrix):
      1. Reshape sv to tensor form (2, ..., 2).
      2. Index the ctrl axis at value 1 to extract the sub-tensor where
         ctrl=|1>.
      3. Apply the gate to the target axis of that sub-tensor only.
      4. Write the modified sub-tensor back into the original tensor.
      5. Reshape back to (2^n,).

    The sub-tensor has shape (2,)^(n-1); the target axis shifts by -1
    if target > ctrl (because the ctrl axis was removed).

    Complexity: O(2^n) time and O(2^n) space.

    Args:
        sv:     Statevector of shape (2^n,).
        matrix: 2×2 unitary gate matrix to apply to target qubit.
        ctrl:   Control qubit index (gate applies when ctrl = |1>).
        target: Target qubit index.
        n:      Total number of qubits.

    Returns:
        New statevector of shape (2^n,).
    """
    sv = sv.reshape([2] * n)

    # Map qubit indices to tensor axes (C-order: qubit q → axis n-1-q)
    axis_ctrl   = n - 1 - ctrl
    axis_target = n - 1 - target

    # Build an index tuple selecting ctrl axis = 1 (ctrl qubit in state |1>)
    idx = [slice(None)] * n
    idx[axis_ctrl] = 1
    idx = tuple(idx)

    # In the sub-tensor (ctrl axis removed), target axis position:
    #   - If axis_target < axis_ctrl: same position (axis_ctrl removed above it)
    #   - If axis_target > axis_ctrl: shifts left by 1
    target_in_sub = axis_target if axis_target < axis_ctrl else axis_target - 1

    sub = sv[idx]  # shape: (2,)^(n-1)
    sub = np.tensordot(matrix, sub, axes=([1], [target_in_sub]))
    sub = np.moveaxis(sub, 0, target_in_sub)
    sv[idx] = sub

    return sv.reshape(-1)


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a statevector simulation.

    Attributes:
        statevector: Final pre-measurement complex amplitude array, length 2^n.
                     This is the state *before* any measurement collapse.
        circuit:     The circuit that was simulated (needed for measurement map).
    """

    statevector: np.ndarray
    circuit: Circuit

    @property
    def probabilities(self) -> np.ndarray:
        """Born-rule probability for each computational basis state.

        P(i) = |<i|ψ>|² = |statevector[i]|²
        """
        return np.abs(self.statevector) ** 2

    def get_counts(self, shots: int, seed: Optional[int] = None) -> dict[str, int]:
        """Sample the statevector probability distribution to produce counts.

        Each shot:
          1. Sample a computational basis index i with probability P(i).
          2. Extract each measured qubit's state from i: state_k = (i >> k) & 1.
          3. Store into the corresponding classical bit per circuit.measurements.
          4. Build a bitstring key with classical bits in MSB-first order:
               key = c[m-1] c[m-2] ... c[1] c[0]   (m = num_clbits)

        Unmeasured classical bits remain 0 in the output.

        Bitstring format example (2 clbits, c0=meas(q0), c1=meas(q1)):
          "00" → c1=0, c0=0
          "11" → c1=1, c0=1

        Args:
            shots: Number of measurement shots.
            seed:  Optional integer seed for reproducibility.

        Returns:
            Dict mapping bitstrings (length num_clbits) to shot counts.
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities
        # Renormalize to guard against floating-point drift
        probs = probs / probs.sum()

        num_clbits = self.circuit.num_clbits
        # Map: qubit index → classical bit index for all measurements
        meas_map: dict[int, int] = {m.qubit: m.clbit for m in self.circuit.measurements}

        # Sample `shots` basis state indices from the probability distribution
        sampled_indices = rng.choice(len(probs), size=shots, p=probs)

        counts: dict[str, int] = {}
        for idx in sampled_indices:
            clbits = [0] * num_clbits
            for qubit, clbit in meas_map.items():
                # Extract qubit's state from the sampled index (little-endian)
                clbits[clbit] = (int(idx) >> qubit) & 1

            # Format: MSB (highest classical bit index) first
            key = "".join(str(clbits[i]) for i in range(num_clbits - 1, -1, -1))
            counts[key] = counts.get(key, 0) + 1

        return counts


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class StatevectorSimulator:
    """Noiseless quantum circuit simulator using the statevector formalism.

    State representation:
      - A complex numpy array of length 2^n, where n = num_qubits.
      - Initialised to |0...0>: amplitude 1 at index 0, 0 everywhere else.

    Gate application strategy:
      - Single-qubit gates: tensor contraction (no 4^n matrices).
      - Controlled gates: direct slice indexing on the reshaped tensor.
      - Each gate costs O(2^n) time and O(1) extra space beyond the statevector.

    Tradeoffs vs full-unitary approach:
      PRO  Gate cost is O(2^n) not O(4^n); practical up to ~30 qubits on RAM.
      PRO  Memory is O(2^n) not O(4^n); no full circuit unitary is stored.
      CON  Cannot symbolically compose circuits or easily extract the full unitary.
      CON  Mid-circuit measurements require collapsing the state (not yet supported).

    Extending to noisy simulation (future):
      Replace the pure-state vector with a density matrix rho ∈ C^{2^n × 2^n}:
        - Gate application: rho ← U rho U†
        - Depolarizing noise after gate g: rho ← (1-p) U rho U† + p/3 * Σ_i P_i rho P_i†
        - Amplitude damping: Kraus operators K0, K1.
      Measurement probabilities become Tr(rho Π_i) where Π_i = |i><i|.
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.n = circuit.num_qubits

    def run(self) -> SimulationResult:
        """Execute the circuit on the |0...0> initial state.

        Applies all gates in declaration order and captures the final
        statevector *before* any measurement collapse.

        Returns:
            SimulationResult containing the statevector and the circuit.
        """
        sv = self._initialize_statevector()
        sv = self._apply_all_gates(sv)
        return SimulationResult(statevector=sv, circuit=self.circuit)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _initialize_statevector(self) -> np.ndarray:
        """Return the |0...0> state: unit amplitude at index 0."""
        sv = np.zeros(2 ** self.n, dtype=complex)
        sv[0] = 1.0 + 0j
        return sv

    def _apply_all_gates(self, sv: np.ndarray) -> np.ndarray:
        """Apply circuit gates in order, returning the final statevector."""
        for gate in self.circuit.gates:
            sv = self._apply_gate(sv, gate)
        return sv

    def _apply_gate(self, sv: np.ndarray, gate: Gate) -> np.ndarray:
        """Dispatch a single gate to the appropriate application routine."""
        if gate.name == "cx":
            ctrl, target = gate.qubits
            return apply_controlled_gate(sv, _X, ctrl, target, self.n)
        if gate.name == "cz":
            ctrl, target = gate.qubits
            return apply_controlled_gate(sv, _Z, ctrl, target, self.n)
        # All remaining supported gates are single-qubit
        matrix = _gate_matrix(gate)
        return apply_single_qubit_gate(sv, matrix, gate.qubits[0], self.n)


# ---------------------------------------------------------------------------
# Density matrix helpers
# ---------------------------------------------------------------------------

def apply_op_to_dm(
    rho: np.ndarray,
    M: np.ndarray,
    qubit: int,
    n: int,
) -> np.ndarray:
    """Apply operator M to qubit in density matrix: returns M_q ρ M_q†.

    M_q denotes M ⊗ I ⊗ ... ⊗ I with M acting only on `qubit`.

    Algorithm (tensor contraction — no 4^n operator ever built):
      1. Reshape ρ from (2^n, 2^n) to (2, 2, ..., 2) with 2n axes.
         In C-order: row indices are axes 0..n-1 (MSB first), so qubit q
         corresponds to row axis (n-1-q) and col axis (2n-1-q).
      2. Left-multiply by M on the row axis for qubit q:
           tensordot(M, ρ_tensor, axes=([1], [row_axis])) then moveaxis.
      3. Right-multiply by M† on the col axis for qubit q:
           tensordot(ρ_tensor, M†, axes=([col_axis], [0])) then moveaxis.
      4. Reshape back to (2^n, 2^n).

    Complexity: O(4^n) time and space — same order as the density matrix itself.

    Args:
        rho:   Density matrix of shape (2^n, 2^n).
        M:     2×2 complex operator matrix.
        qubit: Target qubit index (0 = LSB, little-endian convention).
        n:     Total number of qubits.

    Returns:
        New density matrix of shape (2^n, 2^n).
    """
    dim = 2 ** n
    rho_t = rho.reshape([2] * (2 * n))

    # In C-order reshape: qubit q → row axis (n-1-q), col axis (2n-1-q)
    row_axis = n - 1 - qubit
    col_axis = 2 * n - 1 - qubit

    # Left: rho_t' = M @ rho_t  along row_axis
    # tensordot(M, rho_t, axes=([1], [row_axis])) contracts M's "in" index (axis 1)
    # with rho_t's row_axis.  New M output axis lands at position 0; move it back.
    rho_t = np.tensordot(M, rho_t, axes=([1], [row_axis]))
    rho_t = np.moveaxis(rho_t, 0, row_axis)

    # Right: rho_t' = rho_t @ M†  along col_axis
    # tensordot(rho_t, M†, axes=([col_axis], [0])) contracts rho_t's col_axis with
    # M†'s "in" index (axis 0).  New M† output axis is appended at the end; move back.
    M_dag = M.conj().T
    rho_t = np.tensordot(rho_t, M_dag, axes=([col_axis], [0]))
    rho_t = np.moveaxis(rho_t, -1, col_axis)

    return rho_t.reshape(dim, dim)


def apply_kraus_to_dm(
    rho: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit: int,
    n: int,
) -> np.ndarray:
    """Apply a Kraus noise channel to qubit in density matrix.

    Computes E(ρ) = Σ_i K_i ρ K_i†  where {K_i} are the Kraus operators.

    Args:
        rho:       Density matrix of shape (2^n, 2^n).
        kraus_ops: List of 2×2 Kraus operator matrices satisfying Σ K_i† K_i = I.
        qubit:     Target qubit index.
        n:         Total number of qubits.

    Returns:
        New density matrix after applying the noise channel.
    """
    rho_new = np.zeros_like(rho)
    for K in kraus_ops:
        rho_new = rho_new + apply_op_to_dm(rho, K, qubit, n)
    return rho_new


# ---------------------------------------------------------------------------
# Density matrix simulation result
# ---------------------------------------------------------------------------

@dataclass
class DensityMatrixResult:
    """Result of a density matrix simulation.

    Attributes:
        density_matrix: Final density matrix ρ of shape (2^n, 2^n).
        circuit:        The simulated circuit (needed for measurement map).

    Representation vs. StatevectorSimulator:
      - Memory: O(4^n) vs O(2^n)  — density matrix is 2^n times larger.
      - Probabilities: P(i) = ρ[i,i]  vs P(i) = |sv[i]|²  (same numerically
        for pure states).
      - Handles mixed states; StatevectorSimulator cannot represent noise.
    """

    density_matrix: np.ndarray
    circuit: Circuit

    @property
    def probabilities(self) -> np.ndarray:
        """Born-rule probabilities from the diagonal of the density matrix.

        P(i) = ⟨i|ρ|i⟩ = ρ[i, i]

        For a pure state ρ = |ψ⟩⟨ψ| this equals |ψ[i]|², matching
        StatevectorSimulator.  For mixed states this gives the correct
        classical mixture probabilities.
        """
        return np.real(np.diag(self.density_matrix))

    def get_counts(self, shots: int, seed: Optional[int] = None) -> dict[str, int]:
        """Sample the density matrix probability distribution to produce counts.

        Identical sampling logic to SimulationResult.get_counts — both draw
        shot samples from the marginal probability vector P(i) = ρ[i,i].

        Args:
            shots: Number of measurement shots.
            seed:  Optional integer seed for reproducibility.

        Returns:
            Dict mapping bitstrings (length num_clbits) to shot counts.
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities
        probs = probs / probs.sum()  # renormalize against floating-point drift

        num_clbits = self.circuit.num_clbits
        meas_map: dict[int, int] = {m.qubit: m.clbit for m in self.circuit.measurements}

        sampled_indices = rng.choice(len(probs), size=shots, p=probs)

        counts: dict[str, int] = {}
        for idx in sampled_indices:
            clbits = [0] * num_clbits
            for qubit, clbit in meas_map.items():
                clbits[clbit] = (int(idx) >> qubit) & 1
            key = "".join(str(clbits[i]) for i in range(num_clbits - 1, -1, -1))
            counts[key] = counts.get(key, 0) + 1

        return counts


# ---------------------------------------------------------------------------
# Density matrix simulator
# ---------------------------------------------------------------------------

class DensityMatrixSimulator:
    """Noisy quantum circuit simulator using the density matrix formalism.

    State representation
    --------------------
    A Hermitian positive-semidefinite matrix ρ ∈ ℂ^{2^n × 2^n} with Tr(ρ)=1.
    Initialised to the pure ground state |0...0⟩⟨0...0|.

    Gate application
    ----------------
    For each gate U:  ρ ← U_q ρ U_q†  (tensor contraction, O(4^n) per gate).

    Noise application (after each gate)
    ------------------------------------
    For each Kraus operator K_i in the channel for that gate/qubit:
        ρ ← Σ_i K_i ρ K_i†  (O(4^n) per Kraus operator)

    Runtime cost vs. StatevectorSimulator
    --------------------------------------
    Memory:   O(4^n) vs O(2^n)   — 2^n times more memory.
    Per gate: O(4^n) vs O(2^n)   — 2^n times more work per gate.
    Practical limit: ~12-15 qubits (vs ~30 for statevector).

    This is the recommended approach for a student project because:
      - The math is fully deterministic: same circuit → same ρ.
      - Kraus operators make the physics transparent.
      - No per-shot randomness complicates debugging.

    Switching between modes
    -----------------------
    # Noiseless:
    sim = DensityMatrixSimulator(circuit)

    # Depolarizing after every gate:
    from simulator.noise import NoiseModel, DepolarizingChannel
    nm = NoiseModel()
    nm.add_all_gates_noise(DepolarizingChannel(p=0.01))
    sim = DensityMatrixSimulator(circuit, noise_model=nm)

    # Amplitude damping only after specific gate:
    nm = NoiseModel()
    nm.add_gate_noise("cx", qubit=0, channel=AmplitudeDampingChannel(gamma=0.05))
    sim = DensityMatrixSimulator(circuit, noise_model=nm)
    """

    def __init__(
        self,
        circuit: Circuit,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self.circuit = circuit
        self.n = circuit.num_qubits
        self.noise_model = noise_model

    def run(self) -> DensityMatrixResult:
        """Execute the circuit, optionally applying noise after each gate.

        Returns:
            DensityMatrixResult with the final density matrix and circuit.
        """
        rho = self._initialize_dm()
        for gate in self.circuit.gates:
            rho = self._apply_gate_dm(rho, gate)
        return DensityMatrixResult(density_matrix=rho, circuit=self.circuit)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _initialize_dm(self) -> np.ndarray:
        """Return ρ = |0...0⟩⟨0...0|: only ρ[0,0] = 1, all others 0."""
        dim = 2 ** self.n
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _apply_gate_dm(self, rho: np.ndarray, gate: Gate) -> np.ndarray:
        """Apply gate unitary then optional noise to density matrix."""
        rho = self._apply_unitary_dm(rho, gate)
        if self.noise_model is not None:
            rho = self._apply_noise_dm(rho, gate)
        return rho

    def _apply_unitary_dm(self, rho: np.ndarray, gate: Gate) -> np.ndarray:
        """Apply the gate's unitary U: ρ ← U_full ρ U_full†."""
        if gate.name == "cx":
            ctrl, target = gate.qubits
            U = _build_two_qubit_unitary(ctrl, target, _X, self.n)
            return U @ rho @ U.conj().T
        if gate.name == "cz":
            ctrl, target = gate.qubits
            U = _build_two_qubit_unitary(ctrl, target, _Z, self.n)
            return U @ rho @ U.conj().T
        # Single-qubit gate
        matrix = _gate_matrix(gate)
        return apply_op_to_dm(rho, matrix, gate.qubits[0], self.n)

    def _apply_noise_dm(self, rho: np.ndarray, gate: Gate) -> np.ndarray:
        """Apply registered noise channels to each qubit the gate acts on."""
        for qubit in gate.qubits:
            channels = self.noise_model.get_channels_for_gate(gate.name, qubit)
            for channel in channels:
                rho = apply_kraus_to_dm(rho, channel.kraus_operators(), qubit, self.n)
        return rho


# ---------------------------------------------------------------------------
# Two-qubit unitary builder (for controlled gates in DM simulator)
# ---------------------------------------------------------------------------

def _build_two_qubit_unitary(
    ctrl: int, target: int, gate_2x2: np.ndarray, n: int
) -> np.ndarray:
    """Build the 2^n × 2^n unitary for a controlled single-qubit gate.

    For each basis state |i⟩:
      - If ctrl qubit of i is 0: output = i  (identity on target)
      - If ctrl qubit of i is 1: output = i with target bit flipped through gate_2x2

    For CX: gate_2x2 = X  → |ctrl=1, target⟩ → |ctrl=1, X·target⟩
    For CZ: gate_2x2 = Z  → phase flip on |11⟩ component

    Args:
        ctrl:      Control qubit index.
        target:    Target qubit index.
        gate_2x2:  2×2 unitary applied to target when ctrl=|1⟩.
        n:         Total number of qubits.

    Returns:
        (2^n, 2^n) unitary matrix.
    """
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)

    for j in range(dim):  # input basis state
        ctrl_val = (j >> ctrl) & 1
        if ctrl_val == 0:
            # ctrl=0: identity on whole system
            U[j, j] += 1.0
        else:
            # ctrl=1: apply gate_2x2 to target qubit
            target_val = (j >> target) & 1
            for out_t in range(2):
                # gate_2x2[out_t, target_val] is the amplitude for target → out_t
                amp = gate_2x2[out_t, target_val]
                if amp == 0:
                    continue
                # Build output index: same as j but with target bit = out_t
                i = (j & ~(1 << target)) | (out_t << target)
                U[i, j] += amp

    return U
