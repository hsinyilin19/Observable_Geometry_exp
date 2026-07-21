"""
Helper functions for the observable-manifolds VQE experiment.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import OptimizeResult
from dataclasses import dataclass


def P_label(num_qubits: int, support: Mapping[int, str]) -> str:
    """
    Return a big-endian Qiskit Pauli label.
    """
    label = ["I"] * num_qubits
    for qubit, pauli in support.items():
        if not 0 <= qubit < num_qubits:
            raise ValueError(
                f"qubit index {qubit} is outside 0..{num_qubits - 1}"
            )
        axis = pauli.upper()
        if axis not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"unknown Pauli operator {pauli!r}")
        label[num_qubits - 1 - qubit] = axis
    return "".join(label)


def sparse_P(
    num_qubits: int,
    terms: Sequence[tuple[str, float]],
) -> SparsePauliOp:
    """
    Build a simplified sparse Pauli operator from ``(label, coefficient)`` terms.
    """
    if not terms:
        return SparsePauliOp.from_list([("I" * num_qubits, 0.0)])
    return SparsePauliOp.from_list(
        [(label, complex(coefficient)) for label, coefficient in terms]
    ).simplify(atol=1e-14)


def bind(
    circuit: QuantumCircuit,
    parameters: Sequence[Any],
    values: Sequence[float],
) -> QuantumCircuit:
    """
    Return a copy of ``circuit`` with all variational parameters assigned.
    """
    if len(parameters) != len(values):
        raise ValueError(
            f"parameter/value mismatch: {len(parameters)} vs {len(values)}"
        )
    assignments = {
        parameter: float(value)
        for parameter, value in zip(parameters, values)
    }
    return circuit.assign_parameters(assignments, inplace=False)


def ev(circuit: QuantumCircuit, H: SparsePauliOp) -> float:
    """
    Evaluate ``<0|U^dagger H U|0>`` by exact statevector simulation.
    """
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(H)
    return float(np.real_if_close(value).real)


def min_cd(
    objective: Callable[[np.ndarray], float],
    initial_point: Sequence[float],
    *,
    maxiter: int = 4,
    eps: float = 1e-5,
    initial_step: float = 0.25,
    shrink: float = 0.5,
) -> OptimizeResult:
    """
    Run the coordinate-descent optimizer used for the paper experiment.
    """
    x = np.asarray(initial_point, dtype=float).copy()
    value = float(objective(x))
    evaluations = 1
    completed_iterations = 0
    step = float(initial_step)

    for iteration in range(maxiter):
        completed_iterations = iteration + 1
        improved = False

        for coordinate in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[coordinate] += step
            x_minus[coordinate] -= step

            value_plus = float(objective(x_plus))
            value_minus = float(objective(x_minus))
            evaluations += 2

            if value_plus < value and value_plus <= value_minus:
                x = x_plus
                value = value_plus
                improved = True
            elif value_minus < value:
                x = x_minus
                value = value_minus
                improved = True

        if not improved:
            step *= shrink
        if step < eps:
            break

    return OptimizeResult(
        x=x,
        fun=value,
        nfev=evaluations,
        nit=completed_iterations,
        success=True,
        message="coordinate descent complete",
    )


@dataclass(frozen=True)
class DenseBlock:
    """
    A spectral block W_lambda = ker(H - lambda I).
    """

    lam: float
    indices: tuple[int, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.indices)

@dataclass(frozen=True)
class DenseSpectralDecomposition:
    """Numerical version of G_H ~= product_lambda U(W_lambda)."""

    n: int
    evals: np.ndarray
    evecs: np.ndarray
    blocks: tuple[DenseBlock, ...]

    @property
    def N(self) -> int:
        return 2**self.n

    @property
    def block_sizes(self) -> list[int]:
        return sorted((B.multiplicity for B in self.blocks), reverse=True)

    @property
    def dim_G_H(self) -> int:
        return int(sum(m * m for m in self.block_sizes))

    @property
    def dim_M_H(self) -> int:
        return int(self.N * self.N - self.dim_G_H)

    def to_dict(self, max_blocks: int = 16) -> dict:
        items = sorted(self.blocks, key=lambda B: (B.multiplicity, B.lam), reverse=True)
        return {
            "num_qubits": self.n,
            "hilbert_dim": self.N,
            "num_distinct_eigenvalues": len(self.blocks),
            "block_sizes_desc": self.block_sizes,
            "stabilizer_lie_dimension": self.dim_G_H,
            "observable_orbit_dimension": self.dim_M_H,
            "largest_blocks": [
                {"eigenvalue": float(B.lam), "multiplicity": int(B.multiplicity)}
                for B in items[:max_blocks]
            ],
        }



def H_8qubit_su2_non_simple() -> tuple[object, list[tuple[str, float]], dict]:
    """Build the fixed native non-diagonal 8-qubit SU(2)-invariant Hamiltonian."""
    n = 8
    two_body_edges = [
        (0, 1, 1.00),
        (1, 2, 0.73),
        (2, 3, 1.16),
        (3, 4, 0.91),
        (4, 5, 1.08),
        (5, 6, 0.67),
        (6, 7, 1.21),
        (0, 4, 0.31),
        (1, 5, -0.27),
        (2, 6, 0.24),
        (3, 7, -0.19),
    ]
    four_body_pair_products = [
        ((0, 1), (2, 3), 0.13),
        ((4, 5), (6, 7), -0.11),
        ((0, 4), (3, 7), 0.07),
        ((1, 5), (2, 6), 0.05),
    ]

    terms: list[tuple[str, float]] = []
    for i, j, J_ij in two_body_edges:
        for P in ["X", "Y", "Z"]:
            terms.append((P_label(n, {i: P, j: P}), float(J_ij)))

    for (i, j), (k, l), K_ij_kl in four_body_pair_products:
        if len({i, j, k, l}) != 4:
            raise ValueError("four-body pair products must use disjoint pairs")
        for P in ["X", "Y", "Z"]:
            for Q in ["X", "Y", "Z"]:
                terms.append((P_label(n, {i: P, j: P, k: Q, l: Q}), float(K_ij_kl)))

    H = sparse_P(n, terms)
    metadata = {
        "two_body_edges": two_body_edges,
        "four_body_pair_products": [
            {"pair_a": [i, j], "pair_b": [k, l], "coefficient": float(K)}
            for (i, j), (k, l), K in four_body_pair_products
        ],
        "definition": (
            "H = sum_edges J_ij (XX+YY+ZZ)_ij "
            "+ sum_disjoint_pair_products K_ij_kl (XX+YY+ZZ)_ij (XX+YY+ZZ)_kl"
        ),
        "symmetry_protecting_degeneracy": "global SU(2): [H,S_x]=[H,S_y]=[H,S_z]=0",
    }
    return H, terms, metadata


def dense_matrix_from_sparse_pauli(H: object) -> np.ndarray:
    M = H.to_matrix()
    if hasattr(M, "toarray"):
        M = M.toarray()
    return np.asarray(M, dtype=np.complex128)


def spectral_decomposition_from_H(H_mat: np.ndarray, n: int, tol: float = 1e-8) -> DenseSpectralDecomposition:
    evals, evecs = np.linalg.eigh(H_mat)
    order = np.argsort(evals)
    evals = np.real_if_close(evals[order]).astype(float)
    evecs = evecs[:, order]

    blocks: list[DenseBlock] = []
    if len(evals) == 0:
        raise ValueError("empty spectrum")
    start = 0
    for k in range(1, len(evals) + 1):
        if k == len(evals) or abs(evals[k] - evals[start]) > tol:
            idx = tuple(range(start, k))
            lam = float(np.mean(evals[start:k]))
            blocks.append(DenseBlock(lam=lam, indices=idx))
            start = k
    return DenseSpectralDecomposition(n=n, evals=evals, evecs=evecs, blocks=tuple(blocks))
