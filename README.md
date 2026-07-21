# Observable Geometry for Effective Quantum Circuits

This repository contains the experiment code for the paper **“Observable Geometry for Effective Quantum Circuits.”**

## Files

- `exp_observable_manifolds.py` — runs the observable-manifold experiment and generates the paper outputs.
- `vqe_helpers.py` — helper functions for the Hamiltonian, spectral decomposition, statevector evaluation, and optimization.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Qiskit

Install the required packages with:

```bash
python -m pip install numpy scipy matplotlib seaborn qiskit
```

## Run

Place both Python files in the same directory, then run:

```bash
python Exp_observable_manifolds.py
```

The script uses the numerical configuration reported in the paper: 8 qubits, four circuit-candidate families, 24 selected generators, and 6 VQE trials.

## Outputs

Results are written to the `results/` directory:

- `gate_pool_histogram_and_final_energy.png` — the two-panel experiment figure.
- `table_i_circuit_pool_statistics.csv` — the circuit-family statistics reported in Table I.

The main numerical values are also printed in the terminal.


