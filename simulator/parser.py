"""OpenQASM 2.0 parser.

Parses a practical subset of OpenQASM 2.0 into the internal Circuit representation.
Uses a hand-written, line-oriented approach with regex patterns.

Processing strategy:
1. Strip comments and blank lines
2. Concatenate multi-line statements (split on semicolons)
3. Classify each statement by regex match
4. Build Circuit incrementally
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from simulator.circuit import Circuit, Gate, Measurement, GATE_PARAM_COUNTS


class QASMParseError(Exception):
    """Raised when QASM input is malformed."""
    def __init__(self, message: str, line_number: int | None = None) -> None:
        self.line_number = line_number
        prefix = f"Line {line_number}: " if line_number is not None else ""
        super().__init__(f"{prefix}{message}")


def parse_qasm_file(path: str | Path) -> Circuit:
    """Parse a QASM file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QASM file not found: {path}")
    return parse_qasm(path.read_text())


def parse_qasm(source: str) -> Circuit:
    """Parse QASM source string into a Circuit.

    Processing pipeline:
    1. Remove comments (// to end of line)
    2. Join lines and split on semicolons to get statements
    3. Process each statement in order
    4. Return completed Circuit

    Args:
        source: OpenQASM 2.0 source code as a string.

    Returns:
        Circuit object ready for simulation.

    Raises:
        QASMParseError: If the input is malformed or uses unsupported features.
    """
    statements = _tokenize(source)
    return _parse_statements(statements)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(source: str) -> list[tuple[str, int]]:
    """Split source into (statement, line_number) pairs.
    Handles:
    - Single-line comments (//)
    - Multi-statement lines (a; b;)
    - Statements spanning multiple lines (rare in QASM but possible)
    """
    # Remove comments
    lines = source.split("\n")
    cleaned_lines: list[tuple[str, int]] = []
    for i, line in enumerate(lines, start=1):
        # Strip // comments (but not inside strings)
        comment_pos = line.find("//")
        if comment_pos != -1:
            line = line[:comment_pos]
        stripped = line.strip()
        if stripped:
            cleaned_lines.append((stripped, i))

    # Join all text and split on semicolons, tracking line numbers
    statements: list[tuple[str, int]] = []
    buffer = ""
    current_line = 0

    for text, line_num in cleaned_lines:
        if not buffer:
            current_line = line_num
        buffer += " " + text

        # Split on semicolons
        while ";" in buffer:
            stmt, buffer = buffer.split(";", 1)
            stmt = stmt.strip()
            if stmt:
                statements.append((stmt, current_line))
            current_line = line_num

    # Anything left without a semicolon
    remaining = buffer.strip()
    if remaining:
        # Some statements like OPENQASM header might not need semicolons
        # but in standard QASM they do — treat as error
        if not remaining.upper().startswith("OPENQASM"):
            statements.append((remaining, current_line))

    return statements


# ---------------------------------------------------------------------------
# Statement parsing
# ---------------------------------------------------------------------------

# Regex patterns for each statement type
_RE_OPENQASM = re.compile(r"^OPENQASM\s+(\d+\.\d+)$", re.IGNORECASE)
_RE_INCLUDE = re.compile(r'^include\s+"([^"]+)"$')
_RE_QREG = re.compile(r"^qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_RE_CREG = re.compile(r"^creg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_RE_BARRIER = re.compile(r"^barrier\b")
_RE_RESET = re.compile(r"^reset\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_RE_MEASURE = re.compile(
    r"^measure\s+(\w+)\s*\[\s*(\d+)\s*\]\s*->\s*(\w+)\s*\[\s*(\d+)\s*\]$"
)
# Gate pattern: name(params) qubit_args
# Examples: h q[0]    cx q[0],q[1]    rx(pi/2) q[0]
_RE_GATE = re.compile(
    r"^(\w+)"  # gate name
    r"(?:\(([^)]*)\))?"  # optional parameters in parentheses
    r"\s+"  # whitespace
    r"(.+)$"  # qubit arguments
)


def _parse_statements(statements: list[tuple[str, int]]) -> Circuit:
    """Process all statements and build a Circuit."""
    # State tracking
    qregs: dict[str, tuple[int, int]] = {}  # name -> (start_index, size)
    cregs: dict[str, tuple[int, int]] = {}  # name -> (start_index, size)
    total_qubits = 0
    total_clbits = 0
    gates: list[Gate] = []
    measurements: list[Measurement] = []
    header_seen = False

    for stmt, line_num in statements:
        try:
            # --- OPENQASM header ---
            m = _RE_OPENQASM.match(stmt)
            if m:
                version = m.group(1)
                if version != "2.0":
                    raise QASMParseError(
                        f"Only OpenQASM 2.0 is supported, got {version}",
                        line_num,
                    )
                header_seen = True
                continue

            # --- include ---
            m = _RE_INCLUDE.match(stmt)
            if m:
                # We accept "qelib1.inc" silently (gates are built-in)
                # Other includes are warned but not fatal
                filename = m.group(1)
                if filename != "qelib1.inc":
                    raise QASMParseError(
                        f'Unsupported include: "{filename}". '
                        f'Only "qelib1.inc" is supported.',
                        line_num,
                    )
                continue

            # --- qreg ---
            m = _RE_QREG.match(stmt)
            if m:
                name = m.group(1)
                size = int(m.group(2))
                if name in qregs:
                    raise QASMParseError(
                        f"Duplicate qreg declaration: {name}", line_num
                    )
                if size <= 0:
                    raise QASMParseError(
                        f"Invalid qreg size: {size}", line_num
                    )
                qregs[name] = (total_qubits, size)
                total_qubits += size
                continue

            # --- creg ---
            m = _RE_CREG.match(stmt)
            if m:
                name = m.group(1)
                size = int(m.group(2))
                if name in cregs:
                    raise QASMParseError(
                        f"Duplicate creg declaration: {name}", line_num
                    )
                if size <= 0:
                    raise QASMParseError(
                        f"Invalid creg size: {size}", line_num
                    )
                cregs[name] = (total_clbits, size)
                total_clbits += size
                continue

            # --- barrier (parse and ignore) ---
            if _RE_BARRIER.match(stmt):
                continue

            # --- reset ---
            m = _RE_RESET.match(stmt)
            if m:
                # We parse it but warn — not supported in simulation yet
                raise QASMParseError(
                    "reset is parsed but not supported in simulation",
                    line_num,
                )

            # --- measure ---
            m = _RE_MEASURE.match(stmt)
            if m:
                qreg_name = m.group(1)
                qubit_idx = int(m.group(2))
                creg_name = m.group(3)
                clbit_idx = int(m.group(4))

                abs_qubit = _resolve_qubit(qreg_name, qubit_idx, qregs, line_num)
                abs_clbit = _resolve_clbit(creg_name, clbit_idx, cregs, line_num)
                measurements.append(Measurement(qubit=abs_qubit, clbit=abs_clbit))
                continue

            # --- gate application ---
            m = _RE_GATE.match(stmt)
            if m:
                gate_name = m.group(1).lower()
                param_str = m.group(2)  # may be None
                qubit_str = m.group(3)

                # Validate gate is known
                if gate_name not in GATE_PARAM_COUNTS:
                    raise QASMParseError(
                        f"Unknown gate: '{gate_name}'", line_num
                    )

                # Parse parameters
                params = _parse_params(param_str, line_num)
                expected_params = GATE_PARAM_COUNTS[gate_name]
                if len(params) != expected_params:
                    raise QASMParseError(
                        f"Gate '{gate_name}' expects {expected_params} "
                        f"parameter(s), got {len(params)}",
                        line_num,
                    )

                # Parse qubit arguments
                qubits = _parse_qubit_args(qubit_str, qregs, line_num)

                gate = Gate(
                    name=gate_name,
                    qubits=tuple(qubits),
                    params=tuple(params),
                )
                gates.append(gate)
                continue

            # --- unrecognized ---
            raise QASMParseError(f"Unrecognized statement: '{stmt}'", line_num)

        except QASMParseError:
            raise
        except Exception as e:
            raise QASMParseError(f"Unexpected error: {e}", line_num) from e

    # Validation
    if not header_seen:
        raise QASMParseError("Missing 'OPENQASM 2.0;' header")
    if total_qubits == 0:
        raise QASMParseError("No qreg declared — circuit has no qubits")

    # Build circuit
    circuit = Circuit(
        num_qubits=total_qubits,
        num_clbits=total_clbits,
        gates=gates,
        measurements=measurements,
    )
    return circuit


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_qubit(
    reg_name: str, index: int, qregs: dict[str, tuple[int, int]], line_num: int
) -> int:
    """Convert register_name[index] to absolute qubit index."""
    if reg_name not in qregs:
        raise QASMParseError(f"Undefined qreg: '{reg_name}'", line_num)
    start, size = qregs[reg_name]
    if index < 0 or index >= size:
        raise QASMParseError(
            f"Qubit index {index} out of range for qreg '{reg_name}' "
            f"(size {size})",
            line_num,
        )
    return start + index


def _resolve_clbit(
    reg_name: str, index: int, cregs: dict[str, tuple[int, int]], line_num: int
) -> int:
    """Convert register_name[index] to absolute classical bit index."""
    if reg_name not in cregs:
        raise QASMParseError(f"Undefined creg: '{reg_name}'", line_num)
    start, size = cregs[reg_name]
    if index < 0 or index >= size:
        raise QASMParseError(
            f"Classical bit index {index} out of range for creg '{reg_name}' "
            f"(size {size})",
            line_num,
        )
    return start + index


def _parse_qubit_args(
    qubit_str: str, qregs: dict[str, tuple[int, int]], line_num: int
) -> list[int]:
    """Parse comma-separated qubit arguments like 'q[0],q[1]'.

    Returns list of absolute qubit indices.
    """
    qubit_pattern = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")
    parts = [p.strip() for p in qubit_str.split(",")]
    qubits: list[int] = []

    for part in parts:
        m = qubit_pattern.match(part)
        if not m:
            raise QASMParseError(
                f"Invalid qubit argument: '{part}'", line_num
            )
        reg_name = m.group(1)
        index = int(m.group(2))
        qubits.append(_resolve_qubit(reg_name, index, qregs, line_num))

    return qubits


def _parse_params(param_str: str | None, line_num: int) -> list[float]:
    """Parse gate parameters like 'pi/2, -pi/4, 3.14'.

    Supports:
    - pi (and PI, Pi)
    - Arithmetic: +, -, *, /
    - Negative values
    - Integer and float literals
    - Parenthesized sub-expressions

    Uses Python's eval with a restricted namespace for safety.
    """
    if param_str is None or param_str.strip() == "":
        return []

    parts = _split_params(param_str)
    results: list[float] = []

    for part in parts:
        part = part.strip()
        if not part:
            raise QASMParseError(f"Empty parameter in expression", line_num)
        value = _eval_param_expr(part, line_num)
        results.append(value)

    return results


def _split_params(param_str: str) -> list[str]:
    """Split parameter string on commas, respecting parentheses.

    Example: 'pi/2, atan2(1,0)' -> ['pi/2', 'atan2(1,0)']
    """
    parts: list[str] = []
    depth = 0
    current = ""

    for ch in param_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch

    if current.strip():
        parts.append(current)

    return parts


def _eval_param_expr(expr: str, line_num: int) -> float:
    """Safely evaluate a parameter expression.

    Allowed: pi, integers, floats, +, -, *, /, parentheses.
    """
    # Normalize: replace 'pi' with math.pi value
    # We use a restricted eval with only math constants
    safe_namespace = {
        "pi": math.pi,
        "PI": math.pi,
        "Pi": math.pi,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "ln": math.log,
        "exp": math.exp,
    }

    # Validate: only allow safe characters
    allowed = set("0123456789.+-*/() piPIsncotaqrexpl_")
    sanitized = expr.replace(" ", "")
    for ch in sanitized:
        if ch not in allowed:
            raise QASMParseError(
                f"Invalid character '{ch}' in parameter expression: '{expr}'",
                line_num,
            )

    try:
        # Replace common patterns for eval
        eval_expr = expr.strip()
        result = eval(eval_expr, {"__builtins__": {}}, safe_namespace)  # noqa: S307
        return float(result)
    except (SyntaxError, TypeError, NameError, ZeroDivisionError) as e:
        raise QASMParseError(
            f"Cannot evaluate parameter expression '{expr}': {e}",
            line_num,
        ) from e
