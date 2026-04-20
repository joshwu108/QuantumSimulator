"""Noise channels and noise model for density matrix simulation.

Mathematical background
-----------------------
A noise channel is a completely positive, trace-preserving (CPTP) map on
density matrices.  The Kraus representation writes it as:

    E(ρ) = Σ_i  K_i  ρ  K_i†

where the Kraus operators {K_i} satisfy the completeness relation:

    Σ_i K_i† K_i = I                              (trace-preserving)

This ensures Tr[E(ρ)] = Tr[ρ] = 1 for all valid density matrices ρ.

Implemented channels
--------------------
DepolarizingChannel(p)
    Models random Pauli errors.  With probability p/3 each, an X, Y, or Z
    error is applied; with probability 1-p the qubit is left untouched.

    E(ρ) = (1-p) ρ  +  (p/3)(X ρ X  +  Y ρ Y  +  Z ρ Z)

    Kraus operators:
        K0 = sqrt(1-p) · I
        K1 = sqrt(p/3) · X
        K2 = sqrt(p/3) · Y
        K3 = sqrt(p/3) · Z

    Physical range: 0 ≤ p ≤ 3/4.
      p = 0    → identity (no noise)
      p = 3/4  → maximally depolarizing: any ρ → I/2

AmplitudeDampingChannel(gamma)
    Models energy relaxation (T1 decay): the excited state |1⟩ spontaneously
    decays to the ground state |0⟩ with probability γ.

    Kraus operators:
        K0 = [[1,         0      ],   "no-jump"  (qubit stays in ground state
              [0, sqrt(1-γ)      ]]    or excited state shrinks)

        K1 = [[0, sqrt(γ)],           "jump"  (|1⟩ → |0⟩ with amplitude √γ)
              [0,    0   ]]

    Effect on density matrix ρ = [[ρ00, ρ01], [ρ10, ρ11]]:
        ρ00' = ρ00 + γ · ρ11         (ground state grows)
        ρ11' = (1-γ) · ρ11           (excited state decays)
        ρ01' = sqrt(1-γ) · ρ01       (off-diagonal coherences damp)
        ρ10' = sqrt(1-γ) · ρ10

    Physical range: 0 ≤ γ ≤ 1.
      γ = 0  → identity (no noise)
      γ = 1  → complete damping: any ρ → |0><0|

NoiseModel
----------
Configuration object that maps (gate_name, qubit_index) → [NoiseChannel, ...].

    # Noise after every gate on every qubit:
    nm = NoiseModel()
    nm.add_all_gates_noise(DepolarizingChannel(p=0.01))

    # Noise only after specific gate/qubit combinations:
    nm = NoiseModel()
    nm.add_gate_noise("h", qubit=0, channel=AmplitudeDampingChannel(gamma=0.02))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NoiseChannel:
    """Abstract base class for a single-qubit noise channel.

    Subclasses must implement kraus_operators().
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Return a list of 2×2 complex Kraus operator matrices {K_i}.

        The operators must satisfy the completeness condition:
            Σ_i K_i† K_i = I
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Depolarizing channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepolarizingChannel(NoiseChannel):
    """Single-qubit depolarizing noise channel with error probability p.

    E(ρ) = (1-p) ρ + (p/3)(X ρ X + Y ρ Y + Z ρ Z)

    Kraus operators (4 total):
        K0 = sqrt(1-p) · I      (no error)
        K1 = sqrt(p/3) · X      (bit-flip error)
        K2 = sqrt(p/3) · Y      (bit-and-phase-flip error)
        K3 = sqrt(p/3) · Z      (phase-flip error)

    Valid range: 0 ≤ p ≤ 0.75.
    """

    p: float  # error probability

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 0.75):
            raise ValueError(
                f"Depolarizing p must be in [0, 0.75], got {self.p}"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        p = self.p
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1],   [1,  0]],  dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]],  dtype=complex)
        Z = np.array([[1, 0],   [0, -1]],  dtype=complex)
        return [
            math.sqrt(1 - p)       * I,  # K0: no error
            math.sqrt(p / 3)       * X,  # K1: X error
            math.sqrt(p / 3)       * Y,  # K2: Y error
            math.sqrt(p / 3)       * Z,  # K3: Z error
        ]


# ---------------------------------------------------------------------------
# Amplitude damping channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AmplitudeDampingChannel(NoiseChannel):
    """Single-qubit amplitude damping channel with damping parameter gamma.

    Models spontaneous emission: |1⟩ decays to |0⟩ with probability γ.

    Kraus operators (2 total):
        K0 = [[1,          0      ]]   ("no-jump" operator)
             [[0, sqrt(1-γ)      ]]

        K1 = [[0, sqrt(γ)]]            ("jump" operator: |1⟩ → |0⟩)
             [[0,    0   ]]

    Valid range: 0 ≤ γ ≤ 1.
    """

    gamma: float  # damping parameter

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(
                f"Amplitude damping gamma must be in [0, 1], got {self.gamma}"
            )

    def kraus_operators(self) -> list[np.ndarray]:
        g = self.gamma
        K0 = np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex)
        K1 = np.array([[0, math.sqrt(g)], [0, 0]],     dtype=complex)
        return [K0, K1]


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    """Maps (gate_name, qubit) pairs to lists of noise channels.

    Noise can be added in two ways:

    1. Global noise — applied after every gate on every qubit it touches:
           model.add_all_gates_noise(DepolarizingChannel(p=0.01))

    2. Gate-specific noise — applied only after a particular gate on a
       particular qubit:
           model.add_gate_noise("cx", qubit=0, channel=AmplitudeDampingChannel(gamma=0.05))

    Both styles can be combined; get_channels_for_gate returns the union.

    Example (all three modes):
        # Noiseless:
        model = NoiseModel()

        # Depolarizing after every gate:
        model = NoiseModel()
        model.add_all_gates_noise(DepolarizingChannel(p=0.01))

        # Amplitude damping only on specific gates:
        model = NoiseModel()
        model.add_gate_noise("h", qubit=0, channel=AmplitudeDampingChannel(gamma=0.02))
    """

    _global_channels: list[NoiseChannel] = field(default_factory=list)
    # key: (gate_name, qubit_index) → list of channels
    _gate_channels: dict[tuple[str, int], list[NoiseChannel]] = field(
        default_factory=dict
    )

    def add_all_gates_noise(self, channel: NoiseChannel) -> None:
        """Register a channel to fire after every gate on every touched qubit."""
        self._global_channels.append(channel)

    def add_gate_noise(
        self,
        gate_name: str,
        qubit: int,
        channel: NoiseChannel,
    ) -> None:
        """Register a channel to fire after `gate_name` on `qubit` only."""
        key = (gate_name, qubit)
        if key not in self._gate_channels:
            self._gate_channels[key] = []
        self._gate_channels[key].append(channel)

    def get_channels_for_gate(
        self, gate_name: str, qubit: int
    ) -> list[NoiseChannel]:
        """Return all channels that should fire after `gate_name` on `qubit`.

        Combines global channels + gate-specific channels in insertion order.
        """
        channels: list[NoiseChannel] = list(self._global_channels)
        specific = self._gate_channels.get((gate_name, qubit), [])
        channels.extend(specific)
        return channels

    def is_noiseless(self) -> bool:
        """Return True when no channels have been registered."""
        return not self._global_channels and not self._gate_channels
