"""Tests for the interactive workbench helpers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from simulator.circuit import Circuit, Gate
from simulator.engine import DensityMatrixResult, SimulationResult
from simulator.workbench import (
    OperationRow,
    build_circuit_from_operation_rows,
    export_openqasm,
    import_qasm_program,
    operation_rows_from_circuit,
    plot_circuit_diagram,
    simulate_operation_rows,
    state_table_rows,
)


def test_build_circuit_from_rows_creates_expected_bell_state():
    """H on q0 then CX(q0, q1) should create a Bell state."""
    rows = [
        OperationRow(gate="h", target=0),
        OperationRow(gate="cx", target=1, control=0),
    ]
    snapshot = simulate_operation_rows(
        num_qubits=2,
        measured_qubits=[0, 1],
        rows=rows,
        shots=256,
        noise_mode="none",
        seed=5,
    )

    assert isinstance(snapshot.result, SimulationResult)
    np.testing.assert_allclose(snapshot.result.probabilities, [0.5, 0.0, 0.0, 0.5])
    assert sum(snapshot.counts.values()) == 256


def test_export_openqasm_includes_gate_lines_and_measurements():
    """The QASM export should serialize gates and terminal measurements."""
    qasm = export_openqasm(
        num_qubits=2,
        measured_qubits=[0, 1],
        rows=[
            OperationRow(gate="h", target=0),
            OperationRow(gate="cz", target=1, control=0),
        ],
    )

    assert 'include "qelib1.inc";' in qasm
    assert "h q[0];" in qasm
    assert "cz q[0], q[1];" in qasm
    assert "measure q[0] -> c[0];" in qasm
    assert "measure q[1] -> c[1];" in qasm


def test_import_qasm_program_returns_editable_rows():
    """QASM import should preserve qubit count, measurements, and gates."""
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """

    num_qubits, measured_qubits, rows = import_qasm_program(source)

    assert num_qubits == 2
    assert measured_qubits == [0, 1]
    assert rows[0].gate == "h"
    assert rows[1].gate == "cx"
    assert rows[1].control == 0
    assert rows[1].target == 1


def test_simulate_operation_rows_with_noise_returns_density_matrix_result():
    """Noisy mode should switch to density-matrix simulation."""
    snapshot = simulate_operation_rows(
        num_qubits=1,
        measured_qubits=[0],
        rows=[OperationRow(gate="h", target=0)],
        shots=128,
        noise_mode="depolarizing",
        noise_strength=0.2,
        seed=11,
    )

    assert isinstance(snapshot.result, DensityMatrixResult)
    assert snapshot.purity < 1.0
    assert sum(snapshot.counts.values()) == 128


def test_state_table_rows_are_sorted_by_probability():
    """The state summary should show the most likely basis states first."""
    snapshot = simulate_operation_rows(
        num_qubits=1,
        measured_qubits=[0],
        rows=[OperationRow(gate="x", target=0)],
        shots=32,
        noise_mode="none",
        seed=3,
    )

    rows = state_table_rows(snapshot.result, max_rows=2)
    assert rows[0]["basis_state"] == "|1>"
    assert rows[0]["probability"] == "1.000000"


def test_plot_circuit_diagram_returns_matplotlib_objects():
    """The circuit preview helper should render a figure without errors."""
    fig, ax = plot_circuit_diagram(
        num_qubits=2,
        measured_qubits=[0, 1],
        rows=[
            OperationRow(gate="h", target=0),
            OperationRow(gate="cx", target=1, control=0),
        ],
    )
    try:
        assert ax.get_title() == "Live Circuit"
    finally:
        fig.clf()


def test_build_circuit_rejects_same_control_and_target():
    """Controlled gates should fail fast when control equals target."""
    try:
        build_circuit_from_operation_rows(
            num_qubits=2,
            measured_qubits=[0, 1],
            rows=[OperationRow(gate="cx", target=0, control=0)],
        )
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for invalid controlled gate.")


def test_operation_rows_from_circuit_rejects_unsupported_gate():
    """Workbench import should fail loudly for gates the UI cannot represent."""
    circuit = Circuit(num_qubits=2, num_clbits=0)
    circuit.add_gate(Gate(name="swap", qubits=(0, 1)))

    try:
        operation_rows_from_circuit(circuit)
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("Expected unsupported gate import to raise ValueError.")
