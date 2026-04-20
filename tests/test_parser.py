"""Unit tests for the QASM parser.

Tests are organized by feature:
1. Header and includes
2. Register declarations
3. Gate parsing (non-parametric and parametric)
4. Measurement parsing
5. Multi-register circuits
6. Error handling
7. Integration tests with example files
"""

import math
from pathlib import Path

import pytest

from simulator.circuit import Gate, Measurement, Circuit
from simulator.parser import parse_qasm, parse_qasm_file, QASMParseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# 1. Header and includes
# ---------------------------------------------------------------------------

class TestHeader:
    def test_valid_header(self):
        qasm = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];"
        circuit = parse_qasm(qasm)
        assert circuit.num_qubits == 1

    def test_missing_header(self):
        with pytest.raises(QASMParseError, match="Missing.*header"):
            parse_qasm("qreg q[1];\ncreg c[1];")

    def test_wrong_version(self):
        with pytest.raises(QASMParseError, match="Only OpenQASM 2.0"):
            parse_qasm("OPENQASM 3.0;\nqreg q[1];")

    def test_include_qelib(self):
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];'
        circuit = parse_qasm(qasm)
        assert circuit.num_qubits == 1

    def test_unsupported_include(self):
        qasm = 'OPENQASM 2.0;\ninclude "custom.inc";\nqreg q[1];'
        with pytest.raises(QASMParseError, match="Unsupported include"):
            parse_qasm(qasm)


# ---------------------------------------------------------------------------
# 2. Register declarations
# ---------------------------------------------------------------------------

class TestRegisters:
    def test_single_qreg(self):
        qasm = "OPENQASM 2.0;\nqreg q[3];\ncreg c[3];"
        circuit = parse_qasm(qasm)
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3

    def test_multiple_qregs(self):
        qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\ncreg c[5];"
        circuit = parse_qasm(qasm)
        assert circuit.num_qubits == 5
        assert circuit.num_clbits == 5

    def test_no_qreg(self):
        with pytest.raises(QASMParseError, match="No qreg declared"):
            parse_qasm("OPENQASM 2.0;\ncreg c[1];")

    def test_duplicate_qreg(self):
        qasm = "OPENQASM 2.0;\nqreg q[2];\nqreg q[3];"
        with pytest.raises(QASMParseError, match="Duplicate qreg"):
            parse_qasm(qasm)

    def test_zero_size_register(self):
        qasm = "OPENQASM 2.0;\nqreg q[0];"
        with pytest.raises(QASMParseError, match="Invalid qreg size"):
            parse_qasm(qasm)


# ---------------------------------------------------------------------------
# 3. Gate parsing
# ---------------------------------------------------------------------------

class TestGates:
    def test_single_qubit_gates(self):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        x q[0];
        y q[0];
        z q[0];
        s q[0];
        t q[0];
        sdg q[0];
        tdg q[0];
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.gates) == 8
        assert circuit.gates[0] == Gate(name="h", qubits=(0,))
        assert circuit.gates[1] == Gate(name="x", qubits=(0,))
        assert circuit.gates[6] == Gate(name="sdg", qubits=(0,))

    def test_cx_gate(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        cx q[0], q[1];
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.gates) == 1
        assert circuit.gates[0] == Gate(name="cx", qubits=(0, 1))

    def test_cz_gate(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        cz q[0], q[1];
        """
        circuit = parse_qasm(qasm)
        assert circuit.gates[0] == Gate(name="cz", qubits=(0, 1))

    def test_rx_with_pi(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(pi) q[0];
        """
        circuit = parse_qasm(qasm)
        assert circuit.gates[0].name == "rx"
        assert math.isclose(circuit.gates[0].params[0], math.pi)

    def test_rx_with_pi_fraction(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(pi/2) q[0];
        """
        circuit = parse_qasm(qasm)
        assert math.isclose(circuit.gates[0].params[0], math.pi / 2)

    def test_negative_parameter(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rz(-pi/4) q[0];
        """
        circuit = parse_qasm(qasm)
        assert math.isclose(circuit.gates[0].params[0], -math.pi / 4)

    def test_u3_three_params(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        u3(pi/2, 0, pi) q[0];
        """
        circuit = parse_qasm(qasm)
        gate = circuit.gates[0]
        assert gate.name == "u3"
        assert len(gate.params) == 3
        assert math.isclose(gate.params[0], math.pi / 2)
        assert math.isclose(gate.params[1], 0.0)
        assert math.isclose(gate.params[2], math.pi)

    def test_u2_two_params(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        u2(0, pi) q[0];
        """
        circuit = parse_qasm(qasm)
        gate = circuit.gates[0]
        assert gate.name == "u2"
        assert len(gate.params) == 2

    def test_unknown_gate(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        foo q[0];
        """
        with pytest.raises(QASMParseError, match="Unknown gate.*foo"):
            parse_qasm(qasm)

    def test_wrong_param_count(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(pi/2, pi) q[0];
        """
        with pytest.raises(QASMParseError, match="expects 1 parameter"):
            parse_qasm(qasm)

    def test_qubit_out_of_range(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[5];
        """
        with pytest.raises(QASMParseError, match="out of range"):
            parse_qasm(qasm)


# ---------------------------------------------------------------------------
# 4. Measurement parsing
# ---------------------------------------------------------------------------

class TestMeasurement:
    def test_single_measurement(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.measurements) == 1
        assert circuit.measurements[0] == Measurement(qubit=0, clbit=0)

    def test_multiple_measurements(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.measurements) == 3

    def test_measurement_undefined_qreg(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        measure x[0] -> c[0];
        """
        with pytest.raises(QASMParseError, match="Undefined qreg"):
            parse_qasm(qasm)

    def test_measurement_undefined_creg(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        measure q[0] -> x[0];
        """
        with pytest.raises(QASMParseError, match="Undefined creg"):
            parse_qasm(qasm)


# ---------------------------------------------------------------------------
# 5. Multi-register circuits
# ---------------------------------------------------------------------------

class TestMultiRegister:
    def test_cross_register_gate(self):
        qasm = """
        OPENQASM 2.0;
        qreg a[2];
        qreg b[1];
        cx a[1], b[0];
        """
        circuit = parse_qasm(qasm)
        # a[1] -> absolute index 1, b[0] -> absolute index 2
        assert circuit.gates[0] == Gate(name="cx", qubits=(1, 2))
        assert circuit.num_qubits == 3

    def test_multi_creg_measurement(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        creg ca[1];
        creg cb[1];
        measure q[0] -> ca[0];
        measure q[1] -> cb[0];
        """
        circuit = parse_qasm(qasm)
        assert circuit.measurements[0] == Measurement(qubit=0, clbit=0)
        assert circuit.measurements[1] == Measurement(qubit=1, clbit=1)


# ---------------------------------------------------------------------------
# 6. Comments and whitespace
# ---------------------------------------------------------------------------

class TestCommentsWhitespace:
    def test_inline_comments(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1]; // quantum register
        creg c[1]; // classical register
        h q[0]; // Hadamard
        measure q[0] -> c[0]; // measure
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.gates) == 1
        assert len(circuit.measurements) == 1

    def test_blank_lines(self):
        qasm = """

        OPENQASM 2.0;

        qreg q[1];

        h q[0];

        """
        circuit = parse_qasm(qasm)
        assert len(circuit.gates) == 1

    def test_barrier_ignored(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        barrier q[0], q[1];
        cx q[0], q[1];
        """
        circuit = parse_qasm(qasm)
        assert len(circuit.gates) == 2


# ---------------------------------------------------------------------------
# 7. Integration tests with example files
# ---------------------------------------------------------------------------

class TestExampleFiles:
    def test_bell_state(self):
        circuit = parse_qasm_file(EXAMPLES_DIR / "bell_state.qasm")
        assert circuit.num_qubits == 2
        assert circuit.num_clbits == 2
        assert len(circuit.gates) == 2
        assert circuit.gates[0] == Gate(name="h", qubits=(0,))
        assert circuit.gates[1] == Gate(name="cx", qubits=(0, 1))
        assert len(circuit.measurements) == 2

    def test_ghz_state(self):
        circuit = parse_qasm_file(EXAMPLES_DIR / "ghz_state.qasm")
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3
        assert len(circuit.gates) == 3

    def test_parametric_gates(self):
        circuit = parse_qasm_file(EXAMPLES_DIR / "parametric_gates.qasm")
        assert circuit.num_qubits == 1
        assert len(circuit.gates) == 2
        assert math.isclose(circuit.gates[0].params[0], math.pi / 2)
        assert math.isclose(circuit.gates[1].params[0], math.pi / 4)

    def test_multi_register(self):
        circuit = parse_qasm_file(EXAMPLES_DIR / "multi_register.qasm")
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3
        assert len(circuit.gates) == 4

    def test_u3_gate(self):
        circuit = parse_qasm_file(EXAMPLES_DIR / "u3_gate.qasm")
        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 2
        assert circuit.gates[0].name == "u3"
        assert len(circuit.gates[0].params) == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_qasm_file("nonexistent.qasm")


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_numeric_param(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(1.5707963) q[0];
        """
        circuit = parse_qasm(qasm)
        assert math.isclose(circuit.gates[0].params[0], 1.5707963)

    def test_arithmetic_expression(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        rx(2*pi/3) q[0];
        """
        circuit = parse_qasm(qasm)
        assert math.isclose(circuit.gates[0].params[0], 2 * math.pi / 3)

    def test_no_space_in_qubit_arg(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        cx q[0],q[1];
        """
        circuit = parse_qasm(qasm)
        assert circuit.gates[0] == Gate(name="cx", qubits=(0, 1))

    def test_extra_spaces(self):
        qasm = """
        OPENQASM 2.0;
        qreg  q [ 2 ];
        cx  q[ 0 ] , q[ 1 ] ;
        """
        circuit = parse_qasm(qasm)
        assert circuit.gates[0] == Gate(name="cx", qubits=(0, 1))

    def test_gate_order_preserved(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        x q[1];
        cx q[0], q[1];
        z q[0];
        """
        circuit = parse_qasm(qasm)
        names = [g.name for g in circuit.gates]
        assert names == ["h", "x", "cx", "z"]
