"""Command-line runner for OpenQASM 2.0 programs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from simulator import (
    DensityMatrixSimulator,
    StatevectorSimulator,
    parse_qasm_file,
    plot_bloch_from_state,
    plot_counts,
    plot_probabilities,
)
from simulator.engine import DensityMatrixResult, SimulationResult
from simulator.noise import AmplitudeDampingChannel, DepolarizingChannel, NoiseModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenQASM 2.0 file through the quantum simulator and print "
            "the pre-measurement state plus sampled counts."
        )
    )
    parser.add_argument("qasm_file", help="Path to the OpenQASM 2.0 file.")
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots to sample. Default: 1024.",
    )
    parser.add_argument(
        "--noise",
        choices=("none", "depolarizing", "amplitude_damping"),
        default="none",
        help="Optional noise model. Default: none.",
    )
    parser.add_argument(
        "--param",
        type=float,
        default=0.0,
        help=(
            "Noise parameter: depolarizing p in [0, 0.75] or amplitude damping "
            "gamma in [0, 1]. Ignored when --noise none."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for measurement sampling. Default: 7.",
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Optional directory where counts/probability/Bloch plots are saved.",
    )
    parser.add_argument(
        "--bloch-qubit",
        type=int,
        default=0,
        help="Qubit index for the Bloch sphere when saving plots. Default: 0.",
    )
    return parser.parse_args()


def build_noise_model(noise_name: str, param: float) -> NoiseModel | None:
    """Build the requested global noise model."""
    if noise_name == "none":
        return None

    model = NoiseModel()
    if noise_name == "depolarizing":
        model.add_all_gates_noise(DepolarizingChannel(p=param))
    elif noise_name == "amplitude_damping":
        model.add_all_gates_noise(AmplitudeDampingChannel(gamma=param))
    else:
        raise ValueError(f"Unknown noise model '{noise_name}'.")
    return model


def run_simulation(args: argparse.Namespace) -> tuple[SimulationResult | DensityMatrixResult, dict[str, int]]:
    """Run the requested file through the appropriate simulator."""
    circuit = parse_qasm_file(args.qasm_file)
    noise_model = build_noise_model(args.noise, args.param)

    if noise_model is None:
        result: SimulationResult | DensityMatrixResult = StatevectorSimulator(circuit).run()
    else:
        result = DensityMatrixSimulator(circuit, noise_model=noise_model).run()

    counts = result.get_counts(shots=args.shots, seed=args.seed)
    return result, counts


def save_plots(
    result: SimulationResult | DensityMatrixResult,
    counts: dict[str, int],
    output_dir: Path,
    bloch_qubit: int,
) -> list[Path]:
    """Save the available visualization outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    counts_fig, _ = plot_counts(counts, title="Measurement Counts")
    counts_path = output_dir / "counts.png"
    counts_fig.savefig(counts_path, dpi=200, bbox_inches="tight")
    counts_fig.clear()
    saved_files.append(counts_path)

    prob_fig, _ = plot_probabilities(result, title="Basis-State Probabilities")
    prob_path = output_dir / "probabilities.png"
    prob_fig.savefig(prob_path, dpi=200, bbox_inches="tight")
    prob_fig.clear()
    saved_files.append(prob_path)

    try:
        bloch_fig, _ = plot_bloch_from_state(result, qubit=bloch_qubit)
    except ValueError:
        return saved_files

    bloch_path = output_dir / f"bloch_q{bloch_qubit}.png"
    bloch_fig.savefig(bloch_path, dpi=200, bbox_inches="tight")
    bloch_fig.clear()
    saved_files.append(bloch_path)
    return saved_files


def format_state(result: SimulationResult | DensityMatrixResult) -> str:
    """Return the full pre-measurement state as a printable string."""
    if isinstance(result, SimulationResult):
        return np.array2string(
            result.statevector,
            precision=6,
            suppress_small=False,
            threshold=sys.maxsize,
        )

    return np.array2string(
        result.density_matrix,
        precision=6,
        suppress_small=False,
        threshold=sys.maxsize,
    )


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    try:
        result, counts = run_simulation(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"QASM file: {args.qasm_file}")
    print(f"Shots: {args.shots}")
    if isinstance(result, SimulationResult):
        print("Mode: noiseless statevector")
        print("\nPre-measurement statevector:")
    else:
        print(f"Mode: noisy density matrix ({args.noise})")
        print("\nPre-measurement density matrix:")
    print(format_state(result))

    print("\nCounts:")
    print(counts)

    if args.save_plots is not None:
        try:
            saved_files = save_plots(
                result=result,
                counts=counts,
                output_dir=args.save_plots,
                bloch_qubit=args.bloch_qubit,
            )
        except Exception as exc:
            print(f"\nPlot save failed: {exc}", file=sys.stderr)
            return 1
        if saved_files:
            print("\nSaved plots:")
            for path in saved_files:
                print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
