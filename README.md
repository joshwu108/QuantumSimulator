# Quantum Computer Simulator

A Python-based quantum computer simulator that reads OpenQASM 2.0 circuits and produces statevector output and measurement statistics. Supports noiseless simulation, noisy channels (depolarizing, amplitude damping), and visualization.

## Features

- **QASM Parsing**: Reads OpenQASM 2.0 files (IBM Quantum Experience compatible)
- **Statevector Simulation**: Exact noiseless simulation via unitary gate application
- **Noisy Simulation**: Density matrix simulation with depolarizing and amplitude damping channels
- **Measurement Sampling**: Simulates repeated measurements with configurable shot count
- **Visualization**: Measurement histograms and Bloch sphere plots
- **Interactive Frontend**: Streamlit workbench for live circuit editing and results
- **CLI Runner**: Run a `.qasm` file directly with shots and optional noise

## Usage

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

From the repository root, there are two main ways to use the simulator:

- the CLI, which is best for running a `.qasm` file directly
- the Streamlit app, which is best for live demos and interactive exploration

### Quick Start

```bash
source venv/bin/activate

# Run a QASM file from the CLI
python run_qasm.py examples/bell_state.qasm --shots 1024

# Launch the interactive Streamlit app
streamlit run streamlit_app.py

# Save plots from the CLI
python run_qasm.py examples/bell_state.qasm --shots 1024 --save-plots demo_plots
```

### Streamlit Workbench

The Streamlit frontend gives you a live circuit editor that feels much closer
to a small quantum composer:

- upload a local `.qasm` file
- paste OpenQASM text directly
- change the number of qubits
- add or remove gates in a live operation table
- switch between noiseless and noisy simulation
- inspect measurement counts, basis-state probabilities, Bloch spheres, the
  full pre-measurement state, and generated OpenQASM

Run it from the repository root:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

How to use the Streamlit app:

1. Open the app in your browser after running the command.
2. Load a preset circuit, upload a `.qasm` file, or paste OpenQASM text.
3. Choose the number of qubits and the number of measurement shots.
4. Optionally enable `depolarizing` or `amplitude_damping` noise.
5. Inspect the counts plot, probability plot, Bloch sphere, and full pre-measurement state.
6. Download the generated QASM if you want to export the edited circuit.

If you want the app to import the local `simulator` package explicitly:

```bash
source venv/bin/activate
PYTHONPATH=. streamlit run streamlit_app.py
```

### CLI Runner

If you want a direct assignment-ready command, use `run_qasm.py`. The CLI
takes a `.qasm` file and a shot count, runs the circuit, prints the
pre-measurement state, and prints the measurement counts.

Basic usage:

```bash
source venv/bin/activate
python run_qasm.py examples/bell_state.qasm --shots 1024
```

This prints:

- the QASM filename
- the simulation mode
- the pre-measurement statevector or density matrix
- the counts dictionary

Noisy with depolarizing noise:

```bash
source venv/bin/activate
python run_qasm.py examples/bell_state.qasm --shots 1024 --noise depolarizing --param 0.02
```

Noisy with amplitude damping:

```bash
source venv/bin/activate
python run_qasm.py examples/bell_state.qasm --shots 1024 --noise amplitude_damping --param 0.05
```

Save visualization files from the CLI:

```bash
source venv/bin/activate
python run_qasm.py examples/bell_state.qasm --shots 1024 --save-plots demo_plots
```

That command saves:

- `demo_plots/counts.png`
- `demo_plots/probabilities.png`
- `demo_plots/bloch_q0.png`

To save the Bloch sphere for a different qubit:

```bash
source venv/bin/activate
python run_qasm.py examples/ghz_state.qasm --shots 1024 --save-plots demo_plots --bloch-qubit 1
```

If Matplotlib complains about its cache directory:

```bash
source venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib python run_qasm.py examples/bell_state.qasm --shots 1024 --save-plots demo_plots
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
| Multi-qubit gates | `cx`, `cz` |
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
| `simulator/workbench.py` | Frontend-facing circuit editing, simulation, and rendering helpers |
| `run_qasm.py` | Terminal entry point for running `.qasm` files directly |
| `streamlit_app.py` | Interactive Streamlit frontend |

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
├── script.md
├── requirements.txt
├── run_qasm.py
├── streamlit_app.py
├── .streamlit/
│   └── config.toml
├── simulator/
│   ├── __init__.py
│   ├── parser.py
│   ├── circuit.py
│   ├── engine.py
│   ├── noise.py
│   ├── visualization.py
│   └── workbench.py
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_engine.py
│   ├── test_noise.py
│   ├── test_visualization.py
│   └── test_workbench.py
└── examples/
    ├── bell_state.qasm
    ├── ghz_state.qasm
    ├── parametric_gates.qasm
    └── visualization_demo.py
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (for visualization)
- Streamlit (for the interactive frontend)
- pytest (for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v --cov=simulator
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
