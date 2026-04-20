# Quantum Computer Simulator

A Python-based quantum computer simulator that reads OpenQASM 2.0 circuits and produces statevector output and measurement statistics. Supports noiseless simulation, noisy channels (depolarizing, amplitude damping), and visualization.

## Features

- **QASM Parsing**: Reads OpenQASM 2.0 files (IBM Quantum Experience compatible)
- **Statevector Simulation**: Exact noiseless simulation via unitary gate application
- **Noisy Simulation**: Density matrix simulation with depolarizing and amplitude damping channels
- **Measurement Sampling**: Simulates repeated measurements with configurable shot count
- **Visualization**: Measurement histograms and Bloch sphere plots

## Usage

```bash
# Noiseless simulation
python main.py examples/bell_state.qasm --shots 1000

# Noisy simulation with depolarizing channel
python main.py examples/bell_state.qasm --shots 1000 --noise depolarizing --param 0.01

# Noisy simulation with amplitude damping
python main.py examples/bell_state.qasm --shots 1000 --noise amplitude_damping --param 0.05

# With visualization
python main.py examples/bell_state.qasm --shots 1000 --plot
```

### Visualization API

The plotting helpers live in `simulator.visualization` and return Matplotlib
figure/axes objects so they are easy to save into a report or slide deck.

```python
from simulator import (
    plot_bloch_from_state,
    plot_counts,
    plot_probabilities,
)

fig, ax = plot_counts(counts)
fig.savefig("counts.png", dpi=200, bbox_inches="tight")

fig, ax = plot_probabilities(result.statevector)
fig.savefig("probabilities.png", dpi=200, bbox_inches="tight")

fig, ax = plot_bloch_from_state(result.statevector, qubit=0)
fig.savefig("bloch.png", dpi=200, bbox_inches="tight")
```

For a complete demo that saves presentation-ready images, see
`examples/visualization_demo.py`.

## Output

1. **Statevector**: The quantum state immediately before measurement, displayed as a vector of complex amplitudes
2. **Counts**: A dictionary mapping each measured bitstring to its observed count across all shots

Example output:
```
Statevector:
  |00⟩: 0.7071+0.0000j
  |11⟩: 0.7071+0.0000j

Counts (1000 shots):
  00: 503
  11: 497
```

## Supported OpenQASM 2.0 Subset

| Category | Supported |
|----------|-----------|
| Header | `OPENQASM 2.0;`, `include "qelib1.inc";` |
| Registers | `qreg`, `creg` |
| Single-qubit gates | `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg` |
| Parametric gates | `rx`, `ry`, `rz`, `u1`, `u2`, `u3` |
| Multi-qubit gates | `cx`, `ccx`, `swap` |
| Measurement | `measure q[i] -> c[i];` |
| Barrier | Parsed but ignored |

### Not Supported

- Classical control flow (`if`)
- Custom gate definitions
- Mid-circuit measurement
- Reset operations

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────┐
│  QASM Parser │────▶│ Circuit (IR) │────▶│ Simulator Engine │────▶│  Results   │
└─────────────┘     └──────────────┘     └─────────────────┘     └────────────┘
                                                │                        │
                                          ┌─────┴─────┐          ┌──────┴──────┐
                                          │  Noise    │          │ Visualizer  │
                                          │  Model    │          │             │
                                          └───────────┘          └─────────────┘
```

### Modules

| Module | Responsibility |
|--------|---------------|
| `simulator/parser.py` | Tokenize and parse QASM into Circuit IR |
| `simulator/circuit.py` | Gate and Circuit dataclasses |
| `simulator/engine.py` | Statevector and density-matrix simulation engines |
| `simulator/noise.py` | Depolarizing and amplitude damping channel definitions |
| `simulator/visualization.py` | Histogram, probability, and Bloch-sphere plotting |
| `main.py` | CLI entry point and argument parsing |

## Design Decisions

### Qubit Ordering

Uses **little-endian** convention (same as Qiskit): `q[0]` is the least-significant (rightmost) bit in output bitstrings. For example, if `q[0]` is measured as 1 and `q[1]` as 0, the output string is `"01"`.

### State Representation

- **Noiseless**: Statevector as a 1D NumPy array of `complex128` with 2^n entries
- **Noisy**: Density matrix as a 2D NumPy array of `complex128` with shape (2^n, 2^n)

### Gate Application

Single-qubit gates are applied using the reshape/index trick for O(2^n) performance rather than constructing full 2^n x 2^n matrices via Kronecker products.

### Noise Model

Noise is applied as a quantum channel after each gate using Kraus operator representation:

- **Depolarizing**: ρ' = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
- **Amplitude damping**: ρ' = E₀ρE₀† + E₁ρE₁† where E₀ = [[1,0],[0,√(1-γ)]], E₁ = [[0,√γ],[0,0]]

## Project Structure

```
QuantumSimulator/
├── README.md
├── requirements.txt
├── main.py
├── simulator/
│   ├── __init__.py
│   ├── parser.py
│   ├── circuit.py
│   ├── statevector.py
│   ├── density_matrix.py
│   ├── noise.py
│   ├── gates.py
│   └── visualizer.py
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_statevector.py
│   ├── test_density_matrix.py
│   ├── test_noise.py
│   ├── test_gates.py
│   └── test_integration.py
└── examples/
    ├── bell_state.qasm
    ├── ghz_state.qasm
    └── grover_2qubit.qasm
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (for visualization)
- pytest (for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v --cov=simulator
```

## Limitations

- Practical qubit limit: ~16 qubits (noiseless), ~10 qubits (noisy) due to memory
- Terminal measurement only (all measurements must be at the end of the circuit)
- No classical control flow or mid-circuit measurement
- No GPU acceleration

## References

- [OpenQASM 2.0 Specification](https://arxiv.org/abs/1707.03429)
- [Qiskit Textbook](https://qiskit.org/textbook/)
- Nielsen & Chuang, *Quantum Computation and Quantum Information*
