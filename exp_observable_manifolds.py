"""
Experiment for the paper "Observable Geometry for Effective Quantum Circuits".

The script produces:
1. ``table_i_circuit_pool_statistics.csv``
2. ``gate_pool_histogram_and_final_energy.png``

The numerical configuration is fixed to the paper: eight qubits, all four
candidate families, r = 24 selected generators, and six VQE trials.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from vqe_helpers import (P_label, bind, ev, min_cd, sparse_P, 
H_8qubit_su2_non_simple, dense_matrix_from_sparse_pauli, spectral_decomposition_from_H)

NUM_QUBITS = 8
NUM_SELECTED = 24
NUM_TRIALS = 6
MAXITER = 4
SPECTRAL_TOLERANCE = 1e-8
RANDOM_SEED = 2033
PUBLISHED_TRIAL_RNG_OFFSET = 384

OUTPUT_DIRECTORY = Path("results")
FIGURE_2_FILENAME = "gate_pool_histogram_and_final_energy.png"
TABLE_I_FILENAME = "table_i_circuit_pool_statistics.csv"

G1_SINGLE_PAULI = "G1_single_pauli"
G2_TWO_PAULI = "G2_two_pauli"
G3_COLLECTIVE_PAULI = "G3_collective_pauli"
G4_SPIN_SPIN = "G4_spin_spin"
SCORE_KEY = "stabilizer_overlap_score_s_A"

FAMILY_ORDER = [
    (G1_SINGLE_PAULI, r"$\mathcal{G}_1$: single Pauli"),
    (G2_TWO_PAULI, r"$\mathcal{G}_2$: two Pauli"),
    (G3_COLLECTIVE_PAULI, r"$\mathcal{G}_3$: collective Pauli"),
    (G4_SPIN_SPIN, r"$\mathcal{G}_4$: spin-spin"),
]


@dataclass(frozen=True)
class CircuitCandidate:
    """
    A Hermitian generator A and a native implementation of R_A(theta).
    """

    name: str
    family: str
    generator_terms: tuple[tuple[str, float], ...]
    rotation_ops: tuple[tuple[str, tuple[int, ...], float], ...]


def apply_rotation_R_A(
    circuit: QuantumCircuit,
    theta: Parameter | float,
    candidate: CircuitCandidate,
) -> None:
    """
    Append R_A(theta) = exp(-i theta A / 2) to circuit.
    """
    for operation_name, qubits, coefficient in candidate.rotation_ops:
        angle = coefficient * theta
        if operation_name == "rx":
            circuit.rx(angle, qubits[0])
        elif operation_name == "ry":
            circuit.ry(angle, qubits[0])
        elif operation_name == "rz":
            circuit.rz(angle, qubits[0])
        elif operation_name == "rxx":
            circuit.rxx(angle, qubits[0], qubits[1])
        elif operation_name == "ryy":
            circuit.ryy(angle, qubits[0], qubits[1])
        elif operation_name == "rzz":
            circuit.rzz(angle, qubits[0], qubits[1])
        else:
            raise ValueError(f"unknown native operation {operation_name!r}")


def all_qubit_pairs(num_qubits: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
    ]


def circuit_candidate_pool_G(num_qubits: int) -> list[CircuitCandidate]:
    """
    Construct G = G_1 union G_2 union G_3 union G_4 from Sec. V-B.
    """
    pairs = all_qubit_pairs(num_qubits)
    candidates: list[CircuitCandidate] = []

    # G_1 = {alpha_i}: one-qubit Pauli generators.
    one_qubit_operations = {"X": "rx", "Y": "ry", "Z": "rz"}
    for qubit in range(num_qubits):
        for axis, operation_name in one_qubit_operations.items():
            candidates.append(
                CircuitCandidate(
                    name=f"{axis}_{qubit}",
                    family=G1_SINGLE_PAULI,
                    generator_terms=(
                        (P_label(num_qubits, {qubit: axis}), 1.0),
                    ),
                    rotation_ops=((operation_name, (qubit,), 1.0),),
                )
            )

    # G_2 = {alpha_i alpha_j}: same-axis two-qubit Pauli generators.
    two_qubit_operations = {"X": "rxx", "Y": "ryy", "Z": "rzz"}
    for i, j in pairs:
        for axis, operation_name in two_qubit_operations.items():
            candidates.append(
                CircuitCandidate(
                    name=f"{axis}_{i}_{axis}_{j}",
                    family=G2_TWO_PAULI,
                    generator_terms=(
                        (P_label(num_qubits, {i: axis, j: axis}), 1.0),
                    ),
                    rotation_ops=((operation_name, (i, j), 1.0),),
                )
            )

    # G_3 = {sum_i alpha_i}: collective Pauli generators.
    for axis, operation_name in one_qubit_operations.items():
        candidates.append(
            CircuitCandidate(
                name=f"sum_{axis}",
                family=G3_COLLECTIVE_PAULI,
                generator_terms=tuple(
                    (P_label(num_qubits, {qubit: axis}), 1.0)
                    for qubit in range(num_qubits)
                ),
                rotation_ops=tuple(
                    (operation_name, (qubit,), 1.0)
                    for qubit in range(num_qubits)
                ),
            )
        )

    # G_4 = {D_ij}, where D_ij = X_iX_j + Y_iY_j + Z_iZ_j.
    for i, j in pairs:
        candidates.append(
            CircuitCandidate(
                name=f"D_{i}_{j}",
                family=G4_SPIN_SPIN,
                generator_terms=tuple(
                    (P_label(num_qubits, {i: axis, j: axis}), 1.0)
                    for axis in ("X", "Y", "Z")
                ),
                rotation_ops=(
                    ("rxx", (i, j), 1.0),
                    ("ryy", (i, j), 1.0),
                    ("rzz", (i, j), 1.0),
                ),
            )
        )

    return candidates


def dense_generator_A(
    num_qubits: int,
    candidate: CircuitCandidate,
) -> np.ndarray:
    """
    Return the dense Hermitian matrix A for one candidate.
    """
    sparse_generator_A = sparse_P(num_qubits, candidate.generator_terms)
    return dense_matrix_from_sparse_pauli(sparse_generator_A)


def same_eigenspace_mask(spectral_data: Any) -> np.ndarray:
    """
    Return the entries retained by Pi_is(A) = sum_j T_j A T_j.
    """
    mask = np.zeros((spectral_data.N, spectral_data.N), dtype=bool)
    for eigenspace in spectral_data.blocks:
        indices = np.asarray(eigenspace.indices, dtype=int)
        mask[np.ix_(indices, indices)] = True
    return mask


def score_circuit_candidates(
    num_qubits: int,
    spectral_data: Any,
    candidates: Sequence[CircuitCandidate],
) -> list[dict[str, Any]]:
    """
    Compute s(A) = ||Pi_is(A)||_F^2 / ||A||_F^2 for every A in G.
    """
    eigenvector_matrix = spectral_data.evecs
    projector_mask = same_eigenspace_mask(spectral_data)
    scored_candidates: list[dict[str, Any]] = []

    for pool_index, candidate in enumerate(candidates):
        A = dense_generator_A(num_qubits, candidate)
        A_in_eigenbasis = eigenvector_matrix.conj().T @ A @ eigenvector_matrix
        norm_sq = float(np.sum(np.abs(A_in_eigenbasis) ** 2))
        projected_norm_sq = float(
            np.sum(np.abs(A_in_eigenbasis[projector_mask]) ** 2)
        )
        score_s_A = projected_norm_sq / norm_sq

        scored_candidates.append(
            {
                "pool_index": pool_index,
                "name": candidate.name,
                "family": candidate.family,
                SCORE_KEY: float(np.clip(score_s_A, 0.0, 1.0)),
            }
        )

    return scored_candidates


def select_ansatz_candidates(
    candidates: Sequence[CircuitCandidate],
    scored_candidates: Sequence[dict[str, Any]],
    eta: str,
) -> tuple[list[CircuitCandidate], list[dict[str, Any]]]:
    """
    Select the r = 24 candidates defining U_low or U_high.
    """
    if eta == "low":
        ordered = sorted(
            scored_candidates,
            key=lambda row: (row[SCORE_KEY], row["family"], row["name"]),
        )
    elif eta == "high":
        ordered = sorted(
            scored_candidates,
            key=lambda row: (-row[SCORE_KEY], row["family"], row["name"]),
        )
    else:
        raise ValueError("eta must be 'low' or 'high'")

    selected_rows = ordered[:NUM_SELECTED]
    selected_candidates = [
        candidates[row["pool_index"]]
        for row in selected_rows
    ]
    return selected_candidates, selected_rows


def build_ansatz_U_eta(
    num_qubits: int,
    selected_candidates: Sequence[CircuitCandidate],
    eta: str,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """
    Build the ansatz U_eta(theta), eta in {low, high}.
    """
    theta = ParameterVector(f"theta_{eta}", len(selected_candidates))
    U_eta = QuantumCircuit(num_qubits, name=f"U_{eta}")
    for parameter, candidate in zip(theta, selected_candidates):
        apply_rotation_R_A(U_eta, parameter, candidate)
    return U_eta, list(theta)


def family_statistics(
    scored_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    """
    Return the four rows reported in Table I.
    """
    table_labels = {
        G1_SINGLE_PAULI: r"G1: single Pauli",
        G2_TWO_PAULI: r"G2: 2-Pauli",
        G3_COLLECTIVE_PAULI: r"G3: collective Pauli",
        G4_SPIN_SPIN: r"G4: spin-spin",
    }
    rows: list[dict[str, float | int | str]] = []

    for family, _ in FAMILY_ORDER:
        scores = np.asarray(
            [
                row[SCORE_KEY]
                for row in scored_candidates
                if row["family"] == family
            ],
            dtype=float,
        )
        rows.append(
            {
                "Circuit family": table_labels[family],
                "|G_i|": int(len(scores)),
                "Avg s(A)": float(np.mean(scores)),
                "min s(A)": float(np.min(scores)),
                "Max s(A)": float(np.max(scores)),
            }
        )

    return rows


def run_optimization_trials(
    E_low: Callable[[Sequence[float]], float],
    E_high: Callable[[Sequence[float]], float],
) -> list[dict[str, float | int]]:
    """
    Run the six independent trials shown in Fig. 2(b).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rng.random(PUBLISHED_TRIAL_RNG_OFFSET)
    trials: list[dict[str, float | int]] = []

    for trial_index in range(NUM_TRIALS):
        theta0_low = rng.uniform(-1.0, 1.0, NUM_SELECTED)
        theta0_high = rng.uniform(-1.0, 1.0, NUM_SELECTED)

        result_low = min_cd(E_low, theta0_low, maxiter=MAXITER)
        result_high = min_cd(E_high, theta0_high, maxiter=MAXITER)
        trials.append(
            {
                "trial": trial_index,
                "E_low": float(result_low.fun),
                "E_high": float(result_high.fun),
            }
        )

    return trials


def write_table_i(
    path: Path,
    rows: Sequence[dict[str, float | int | str]],
) -> None:
    """
    Write the values displayed in Table I.
    """
    fieldnames = ["Circuit family", "|G_i|", "Avg s(A)", "min s(A)", "Max s(A)"]
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Circuit family": row["Circuit family"],
                    "|G_i|": row["|G_i|"],
                    "Avg s(A)": f"{float(row['Avg s(A)']):.4f}",
                    "min s(A)": f"{float(row['min s(A)']):.4f}",
                    "Max s(A)": f"{float(row['Max s(A)']):.4f}",
                }
            )


def make_figure_2(
    path: Path,
    scored_candidates: Sequence[dict[str, Any]],
    trials: Sequence[dict[str, float | int]],
    E0: float,
) -> None:
    """
    Create the two-panel publication figure used as Fig. 2.
    """
    score_values = [float(row[SCORE_KEY]) for row in scored_candidates]
    bins = np.linspace(0.0, max(1.0, max(score_values)), 11)

    palette = sns.color_palette("Spectral", 10)
    family_colors = [palette[0], palette[3], palette[8], palette[9]]
    line_colors = [palette[0], palette[-1]]

    family_values = [
        [
            float(row[SCORE_KEY])
            for row in scored_candidates
            if row["family"] == family
        ]
        for family, _ in FAMILY_ORDER
    ]
    family_labels = [label for _, label in FAMILY_ORDER]

    trial_indices = [int(row["trial"]) for row in trials]
    low_energies = [float(row["E_low"]) for row in trials]
    high_energies = [float(row["E_high"]) for row in trials]

    with plt.style.context("seaborn-v0_8"):
        figure, (histogram_axis, energy_axis) = plt.subplots(
            1,
            2,
            figsize=(14.4, 5.2),
        )

        histogram_axis.hist(
            family_values,
            bins=bins,
            label=family_labels,
            stacked=True,
            color=family_colors,
            rwidth=0.92,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.8,
        )
        histogram_axis.set_xlim(0.0, max(1.0, max(score_values)))
        histogram_axis.set_xlabel(r"$s(A)$", fontsize=20)
        histogram_axis.set_ylabel("count", fontsize=20)
        histogram_axis.tick_params(axis="both", which="major", labelsize=20)
        histogram_axis.legend(loc="upper right", fontsize=20, frameon=True)
        histogram_axis.text(
            0.02,
            0.95,
            "(a)",
            transform=histogram_axis.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )

        energy_axis.plot(
            trial_indices,
            low_energies,
            marker="o",
            linewidth=2.0,
            color=line_colors[0],
            label=r"$U_{\mathrm{low}}$",
        )
        energy_axis.plot(
            trial_indices,
            high_energies,
            marker="o",
            linewidth=2.0,
            color=line_colors[1],
            label=r"$U_{\mathrm{high}}$",
        )
        energy_axis.axhline(
            E0,
            linestyle="--",
            linewidth=1.8,
            color="black",
            label="exact ground truth",
        )
        energy_axis.set_xlabel("trial", fontsize=20)
        energy_axis.set_ylabel("energy", fontsize=20)
        energy_axis.tick_params(axis="both", which="major", labelsize=18)
        energy_axis.legend(
            fontsize=18,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.04),
            frameon=True,
        )
        energy_axis.text(
            0.02,
            0.95,
            "(b)",
            transform=energy_axis.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )

        figure.tight_layout(w_pad=2.2)
        figure.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(figure)


def sample_standard_deviation(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def print_paper_values(
    table_rows: Sequence[dict[str, float | int | str]],
    low_rows: Sequence[dict[str, Any]],
    high_rows: Sequence[dict[str, Any]],
    trials: Sequence[dict[str, float | int]],
    E0: float,
) -> None:
    """Print the numerical values quoted in Sec. V-B and Sec. V-C."""
    print("\nTable I")
    for row in table_rows:
        print(
            f"  {row['Circuit family']}: |G_i|={row['|G_i|']}, "
            f"Avg={float(row['Avg s(A)']):.4f}, "
            f"min={float(row['min s(A)']):.4f}, "
            f"Max={float(row['Max s(A)']):.4f}"
        )

    low_scores = np.asarray([row[SCORE_KEY] for row in low_rows], dtype=float)
    high_scores = np.asarray([row[SCORE_KEY] for row in high_rows], dtype=float)
    low_energies = [float(row["E_low"]) for row in trials]
    high_energies = [float(row["E_high"]) for row in trials]

    print("\nSection V-C")
    print(f"  E0 = {E0:.10f}")
    print(
        f"  U_low score: mean={np.mean(low_scores):.4f}, "
        f"range=[{np.min(low_scores):.4f}, {np.max(low_scores):.4f}]"
    )
    print(
        f"  U_high score: mean={np.mean(high_scores):.4f}, "
        f"range=[{np.min(high_scores):.4f}, {np.max(high_scores):.4f}]"
    )
    print(
        f"  U_low final energy: {np.mean(low_energies):.4f} "
        f"+/- {sample_standard_deviation(low_energies):.4f}"
    )
    print(
        f"  U_high final energy: {np.mean(high_energies):.4f} "
        f"+/- {sample_standard_deviation(high_energies):.4f}"
    )


def main() -> None:
    H, _, _ = H_8qubit_su2_non_simple()
    H_matrix = dense_matrix_from_sparse_pauli(H)
    spectral_data = spectral_decomposition_from_H(
        H_matrix,
        n=NUM_QUBITS,  # Legacy parameter name in the imported Hamiltonian helper.
        tol=SPECTRAL_TOLERANCE,
    )

    candidate_pool_G = circuit_candidate_pool_G(NUM_QUBITS)
    scored_candidates = score_circuit_candidates(
        NUM_QUBITS,
        spectral_data,
        candidate_pool_G,
    )

    low_candidates, low_rows = select_ansatz_candidates(
        candidate_pool_G,
        scored_candidates,
        eta="low",
    )
    high_candidates, high_rows = select_ansatz_candidates(
        candidate_pool_G,
        scored_candidates,
        eta="high",
    )

    U_low, theta_low = build_ansatz_U_eta(
        NUM_QUBITS,
        low_candidates,
        eta="low",
    )
    U_high, theta_high = build_ansatz_U_eta(
        NUM_QUBITS,
        high_candidates,
        eta="high",
    )

    def E_low(theta_values: Sequence[float]) -> float:
        return ev(bind(U_low, theta_low, theta_values), H)

    def E_high(theta_values: Sequence[float]) -> float:
        return ev(bind(U_high, theta_high, theta_values), H)

    trials = run_optimization_trials(E_low, E_high)
    E0 = float(np.min(spectral_data.evals))
    table_rows = family_statistics(scored_candidates)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIRECTORY / FIGURE_2_FILENAME
    table_path = OUTPUT_DIRECTORY / TABLE_I_FILENAME

    make_figure_2(figure_path, scored_candidates, trials, E0)
    write_table_i(table_path, table_rows)
    print_paper_values(table_rows, low_rows, high_rows, trials, E0)

    print(f"\nWrote {figure_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
