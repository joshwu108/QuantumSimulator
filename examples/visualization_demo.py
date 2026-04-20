"""Small visualization demo for the quantum simulator project.

Run from the repository root:
    python examples/visualization_demo.py
"""

from __future__ import annotations

from pathlib import Path

from simulator import (
    Circuit,
    Gate,
    Measurement,
    StatevectorSimulator,
    plot_bloch_from_state,
    plot_counts,
    plot_probabilities,
)
from simulator.engine import DensityMatrixSimulator
from simulator.noise import DepolarizingChannel, NoiseModel


def build_superposition_circuit() -> Circuit:
    """Hadamard on |0> gives a clean one-qubit superposition."""
    circuit = Circuit(num_qubits=1, num_clbits=1)
    circuit.add_gate(Gate(name="h", qubits=(0,)))
    circuit.add_measurement(Measurement(qubit=0, clbit=0))
    return circuit


def build_bell_circuit() -> Circuit:
    """Bell-state circuit used to show entanglement in counts."""
    circuit = Circuit(num_qubits=2, num_clbits=2)
    circuit.add_gate(Gate(name="h", qubits=(0,)))
    circuit.add_gate(Gate(name="cx", qubits=(0, 1)))
    circuit.add_measurement(Measurement(qubit=0, clbit=0))
    circuit.add_measurement(Measurement(qubit=1, clbit=1))
    return circuit


def main() -> None:
    output_dir = Path("examples/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Superposition: the probability plot shows the 50/50 split and the
    #    Bloch sphere shows that H|0> moves the state to the +X direction.
    superposition = StatevectorSimulator(build_superposition_circuit()).run()
    fig, _ = plot_probabilities(
        superposition.statevector,
        title="Hadamard on |0>: basis-state probabilities",
    )
    fig.savefig(
        output_dir / "superposition_probabilities.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig, _ = plot_bloch_from_state(
        superposition.statevector,
        title="Hadamard on |0>: Bloch sphere",
    )
    fig.savefig(output_dir / "superposition_bloch.png", dpi=200, bbox_inches="tight")

    # 2. Entanglement: Bell-state counts produce only 00 and 11, which is a
    #    very intuitive visual when you are explaining correlated outcomes.
    bell = StatevectorSimulator(build_bell_circuit()).run()
    fig, _ = plot_counts(
        bell.get_counts(shots=1000, seed=7),
        title="Bell State: measurement counts",
    )
    fig.savefig(output_dir / "bell_counts.png", dpi=200, bbox_inches="tight")

    # 3. Noise: depolarizing noise shrinks the reduced Bloch vector, making the
    #    loss of purity visible even before you talk through the equations.
    noisy_model = NoiseModel()
    noisy_model.add_all_gates_noise(DepolarizingChannel(p=0.20))
    noisy_superposition = DensityMatrixSimulator(
        build_superposition_circuit(),
        noise_model=noisy_model,
    ).run()
    fig, _ = plot_bloch_from_state(
        noisy_superposition,
        title="Depolarizing Noise: reduced Bloch sphere",
    )
    fig.savefig(output_dir / "noisy_bloch.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
