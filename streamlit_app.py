"""Interactive Streamlit frontend for the quantum simulator."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from simulator.engine import DensityMatrixResult, SimulationResult
from simulator.parser import QASMParseError
from simulator.visualization import (
    plot_bloch_from_state,
    plot_counts,
    plot_probabilities,
)
from simulator.workbench import (
    APP_SUPPORTED_GATES,
    NOISE_OPTIONS,
    OperationRow,
    blank_operation,
    get_preset,
    import_qasm_program,
    noise_help_text,
    parameter_help_text,
    plot_circuit_diagram,
    preset_names,
    sanitize_operation_rows,
    simulate_operation_rows,
    state_table_rows,
)


st.set_page_config(
    page_title="Quantum Simulator",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply a quiet, readable dark theme."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b131a;
            color: #f3f7fb;
        }
        .block-container {
            max-width: 1460px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #101922;
            border-right: 1px solid rgba(148, 178, 196, 0.14);
        }
        html, body, [class*="css"], [data-testid="stSidebar"] * {
            font-family: Verdana, Geneva, sans-serif;
        }
        h1, h2, h3 {
            color: #f4f8fb;
            letter-spacing: 0.01em;
        }
        p, li, .stCaption, label {
            color: #c4d2dd;
        }
        div[data-testid="stMetric"] {
            background: rgba(14, 22, 30, 0.88);
            border: 1px solid rgba(148, 178, 196, 0.15);
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
        }
        div[data-testid="stCodeBlock"] pre {
            border-radius: 12px;
            border: 1px solid rgba(148, 178, 196, 0.10);
        }
        div[data-testid="stDataFrame"] div[role="table"] {
            border: 1px solid rgba(148, 178, 196, 0.10);
            border-radius: 12px;
        }
        button {
            border-radius: 10px !important;
        }
        .tool-summary {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: rgba(13, 20, 27, 0.82);
            border: 1px solid rgba(148, 178, 196, 0.12);
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Seed session state the first time the app loads."""
    if st.session_state.get("workbench_initialized"):
        return

    preset = get_preset("Bell State")
    st.session_state.workbench_initialized = True
    st.session_state.workbench_num_qubits = preset.num_qubits
    st.session_state.workbench_measured_qubits = list(preset.measured_qubits)
    st.session_state.workbench_operations = list(preset.operations)
    st.session_state.workbench_shots = 1024
    st.session_state.workbench_noise_mode = "none"
    st.session_state.workbench_noise_strength = 0.08
    st.session_state.workbench_bloch_qubit = 0
    st.session_state.workbench_source_name = "Bell State preset"
    st.session_state.workbench_import_source = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )


def rows_to_dataframe(rows: list[OperationRow]) -> pd.DataFrame:
    """Convert operation rows into a DataFrame for editing."""
    records = [
        {
            "enabled": row.enabled,
            "gate": row.gate,
            "target": row.target,
            "control": row.control,
            "param_1": row.param_1,
            "param_2": row.param_2,
            "param_3": row.param_3,
        }
        for row in rows
    ]
    return pd.DataFrame(
        records,
        columns=[
            "enabled",
            "gate",
            "target",
            "control",
            "param_1",
            "param_2",
            "param_3",
        ],
    )


def dataframe_to_rows(frame: pd.DataFrame) -> list[OperationRow]:
    """Convert edited DataFrame rows back into OperationRow objects."""
    rows: list[OperationRow] = []
    for record in frame.to_dict("records"):
        gate = str(record.get("gate", "h") or "h").strip().lower()
        target = _safe_int(record.get("target"), 0)
        control_value = record.get("control")
        control = (
            None
            if pd.isna(control_value) or control_value == ""
            else _safe_int(control_value, 0)
        )
        rows.append(
            OperationRow(
                gate=gate,
                target=target,
                control=control,
                param_1=_safe_float(record.get("param_1"), 0.0),
                param_2=_safe_float(record.get("param_2"), 0.0),
                param_3=_safe_float(record.get("param_3"), 0.0),
                enabled=bool(record.get("enabled", True)),
            )
        )
    return rows


def load_preset(name: str) -> None:
    """Replace the current workbench state with a preset."""
    preset = get_preset(name)
    st.session_state.workbench_num_qubits = preset.num_qubits
    st.session_state.workbench_measured_qubits = list(preset.measured_qubits)
    st.session_state.workbench_operations = list(preset.operations)
    st.session_state.workbench_bloch_qubit = 0
    st.session_state.workbench_source_name = f"{name} preset"


def load_qasm_source(source: str, source_name: str) -> None:
    """Replace the current workbench state from OpenQASM source."""
    imported_qubits, imported_measured, imported_rows = import_qasm_program(source)
    st.session_state.workbench_import_source = source
    st.session_state.workbench_num_qubits = min(imported_qubits, 6)
    st.session_state.workbench_measured_qubits = [
        qubit for qubit in imported_measured if qubit < 6
    ]
    st.session_state.workbench_operations = sanitize_operation_rows(
        imported_rows,
        st.session_state.workbench_num_qubits,
    )
    st.session_state.workbench_bloch_qubit = 0
    st.session_state.workbench_source_name = source_name


def format_full_state(result: SimulationResult | DensityMatrixResult) -> str:
    """Format the full pre-measurement state for display."""
    array = result.statevector if isinstance(result, SimulationResult) else result.density_matrix
    return np.array2string(
        array,
        precision=6,
        suppress_small=False,
        threshold=sys.maxsize,
    )


def run_app() -> None:
    """Render the interactive quantum workbench."""
    inject_styles()
    initialize_state()

    st.title("Quantum Simulator")
    st.caption(
        "Load a `.qasm` file or build a circuit manually, choose the number of shots, "
        "optionally apply noise, and inspect the pre-measurement state and counts."
    )
    st.caption(f"Current source: {st.session_state.workbench_source_name}")

    with st.sidebar:
        st.header("Input")

        preset_choice = st.selectbox(
            "Starter circuit",
            options=preset_names(),
            index=1,
            help="Quick way to load a known-good circuit into the editor.",
        )
        if st.button("Load preset", use_container_width=True):
            load_preset(preset_choice)
            st.rerun()
        st.caption(get_preset(preset_choice).description)

        uploaded_qasm = st.file_uploader(
            "Upload a QASM file",
            type=["qasm"],
            help="Use a local OpenQASM 2.0 file as the circuit source.",
        )
        if uploaded_qasm is not None and st.button("Load uploaded file", use_container_width=True):
            try:
                load_qasm_source(
                    uploaded_qasm.getvalue().decode("utf-8"),
                    uploaded_qasm.name,
                )
            except UnicodeDecodeError:
                st.error("The uploaded file must be UTF-8 text.")
            except (QASMParseError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.rerun()

        with st.expander("Paste QASM text", expanded=False):
            import_source = st.text_area(
                "OpenQASM 2.0 source",
                value=st.session_state.workbench_import_source,
                height=220,
            )
            if st.button("Load pasted QASM", use_container_width=True):
                try:
                    load_qasm_source(import_source, "Pasted QASM")
                except (QASMParseError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

        st.header("Run settings")
        num_qubits = st.slider(
            "Qubit count",
            min_value=1,
            max_value=6,
            value=int(st.session_state.workbench_num_qubits),
        )
        shots = st.slider(
            "Measurement shots",
            min_value=128,
            max_value=4096,
            step=128,
            value=int(st.session_state.workbench_shots),
        )

        measured_default = [
            qubit
            for qubit in st.session_state.workbench_measured_qubits
            if 0 <= qubit < num_qubits
        ]
        measured_qubits = st.multiselect(
            "Measured qubits",
            options=list(range(num_qubits)),
            default=measured_default or list(range(num_qubits)),
            help="Counts are sampled only from these measured qubits.",
        )

        noise_mode = st.selectbox(
            "Noise model",
            options=list(NOISE_OPTIONS),
            format_func=lambda option: option.replace("_", " ").title(),
            index=list(NOISE_OPTIONS).index(st.session_state.workbench_noise_mode),
        )
        if noise_mode == "depolarizing":
            noise_strength = st.slider(
                "Depolarizing p",
                min_value=0.0,
                max_value=0.75,
                value=float(st.session_state.workbench_noise_strength),
                step=0.01,
            )
        elif noise_mode == "amplitude_damping":
            noise_strength = st.slider(
                "Amplitude damping gamma",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.workbench_noise_strength),
                step=0.01,
            )
        else:
            noise_strength = 0.0

        st.caption(noise_help_text(noise_mode))

        bloch_qubit = st.selectbox(
            "Bloch-sphere qubit",
            options=list(range(num_qubits)),
            index=min(int(st.session_state.workbench_bloch_qubit), num_qubits - 1),
            help="For multi-qubit states this uses the reduced single-qubit state.",
        )

        with st.expander("Supported gates", expanded=False):
            st.markdown(
                """
                - Single-qubit: `id`, `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`
                - Parametric: `rx`, `ry`, `rz`, `u1`, `u2`, `u3`
                - Two-qubit: `cx`, `cz`
                """
            )

    st.session_state.workbench_num_qubits = num_qubits
    st.session_state.workbench_shots = shots
    st.session_state.workbench_measured_qubits = measured_qubits
    st.session_state.workbench_noise_mode = noise_mode
    st.session_state.workbench_noise_strength = noise_strength
    st.session_state.workbench_bloch_qubit = bloch_qubit
    st.session_state.workbench_operations = sanitize_operation_rows(
        st.session_state.workbench_operations,
        num_qubits,
    )

    if noise_mode != "none" and num_qubits > 4:
        st.warning(
            "Noisy simulation uses density matrices, so performance drops quickly as "
            "the qubit count grows. Four or fewer qubits keeps the app responsive."
        )

    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with left_col:
        st.subheader("Circuit editor")
        st.caption(
            "Edit the gate sequence below. The preview above it is regenerated from the current table."
        )
        fig, _ = plot_circuit_diagram(
            num_qubits=num_qubits,
            measured_qubits=measured_qubits,
            rows=st.session_state.workbench_operations,
        )
        st.pyplot(fig, use_container_width=True)
        fig.clear()

        toolbar_cols = st.columns(3)
        if toolbar_cols[0].button("Add step", use_container_width=True):
            st.session_state.workbench_operations.append(blank_operation(num_qubits))
            st.rerun()
        if toolbar_cols[1].button("Duplicate last", use_container_width=True):
            if st.session_state.workbench_operations:
                st.session_state.workbench_operations.append(
                    st.session_state.workbench_operations[-1]
                )
            else:
                st.session_state.workbench_operations.append(blank_operation(num_qubits))
            st.rerun()
        if toolbar_cols[2].button("Clear circuit", use_container_width=True):
            st.session_state.workbench_operations = []
            st.rerun()

        operations_df = rows_to_dataframe(st.session_state.workbench_operations)
        edited_df = st.data_editor(
            operations_df,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("On"),
                "gate": st.column_config.SelectboxColumn(
                    "Gate",
                    options=list(APP_SUPPORTED_GATES),
                    required=True,
                ),
                "target": st.column_config.NumberColumn(
                    "Target",
                    min_value=0,
                    max_value=max(num_qubits - 1, 0),
                    step=1,
                    format="%d",
                ),
                "control": st.column_config.NumberColumn(
                    "Control",
                    min_value=0,
                    max_value=max(num_qubits - 1, 0),
                    step=1,
                    format="%d",
                    help="Only used for CX and CZ.",
                ),
                "param_1": st.column_config.NumberColumn("Param 1", format="%.6f"),
                "param_2": st.column_config.NumberColumn("Param 2", format="%.6f"),
                "param_3": st.column_config.NumberColumn("Param 3", format="%.6f"),
            },
        )
        st.session_state.workbench_operations = sanitize_operation_rows(
            dataframe_to_rows(edited_df),
            num_qubits,
        )

        st.caption(
            "Gate parameters: "
            + ", ".join(
                f"`{gate}` uses {parameter_help_text(gate)}"
                for gate in ["rx", "ry", "rz", "u1", "u2", "u3"]
            )
        )
        st.info(
            "Measurements happen at the end of the circuit. Counts are sampled from the final state "
            "for the measured qubits you selected."
        )

    with right_col:
        st.subheader("Simulation output")
        try:
            snapshot = simulate_operation_rows(
                num_qubits=num_qubits,
                measured_qubits=measured_qubits,
                rows=st.session_state.workbench_operations,
                shots=shots,
                noise_mode=noise_mode,
                noise_strength=noise_strength,
                seed=7,
            )
        except Exception as exc:
            st.error(str(exc))
            return

        metric_cols = st.columns(4)
        metric_cols[0].metric("Qubits", snapshot.circuit.num_qubits)
        metric_cols[1].metric("Gates", len(snapshot.circuit.gates))
        metric_cols[2].metric("Mode", snapshot.mode_label)
        metric_cols[3].metric("Purity", f"{snapshot.purity:.3f}")

        plots_tab, state_tab, qasm_tab = st.tabs(["Plots", "State & counts", "QASM"])

        with plots_tab:
            chart_cols = st.columns(2, gap="large")
            with chart_cols[0]:
                if snapshot.counts:
                    fig, _ = plot_counts(snapshot.counts, title="Measurement Counts")
                    st.pyplot(fig, use_container_width=True)
                    fig.clear()
                else:
                    st.info("Counts appear after you select one or more measured qubits.")

            with chart_cols[1]:
                min_probability = 0.0 if num_qubits <= 4 else 0.01
                fig, _ = plot_probabilities(
                    snapshot.result,
                    title="Basis-State Probabilities",
                    min_probability=min_probability,
                )
                st.pyplot(fig, use_container_width=True)
                fig.clear()

            fig, _ = plot_bloch_from_state(
                snapshot.result,
                qubit=bloch_qubit,
                title=f"Bloch Sphere for q{bloch_qubit}",
            )
            st.pyplot(fig, use_container_width=True)
            fig.clear()

        with state_tab:
            if isinstance(snapshot.result, SimulationResult):
                st.caption(
                    "The exact pre-measurement statevector is shown below. The table lists the most likely basis states first."
                )
            else:
                st.caption(
                    "Noise switches the simulator to a density matrix. The table shows the most likely diagonal outcomes first."
                )

            st.dataframe(
                pd.DataFrame(state_table_rows(snapshot.result, max_rows=16)),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Full pre-measurement state", expanded=False):
                st.code(format_full_state(snapshot.result), language="python")

            if snapshot.counts:
                st.caption("Counts for the requested number of shots")
                st.code(str(snapshot.counts), language="python")
            else:
                st.info("No counts are available until at least one qubit is measured.")

        with qasm_tab:
            st.code(snapshot.qasm, language="qasm")
            st.download_button(
                "Download QASM",
                data=snapshot.qasm,
                file_name="quantum_workbench.qasm",
                mime="text/plain",
                use_container_width=True,
            )


def _safe_float(value, default: float) -> float:
    """Parse a float from editor data, falling back when blank."""
    if pd.isna(value) or value == "":
        return default
    return float(value)


def _safe_int(value, default: int) -> int:
    """Parse an int from editor data, falling back when blank."""
    if pd.isna(value) or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    run_app()
