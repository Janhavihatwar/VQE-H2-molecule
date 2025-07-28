# VQE for H2 Molecule Ground State Energy

This project implements the Variational Quantum Eigensolver (VQE) algorithm to calculate the ground state energy of a Hydrogen (H2) molecule. The simulation is performed using a 4-qubit Hamiltonian representation and is built with Python and Qiskit.

## Description

The script defines a 4-qubit Hamiltonian for the H2 molecule and uses a custom ansatz with two entangling layers. It then iteratively runs the VQE algorithm to find the optimal parameters for the ansatz that minimize the molecule's energy.

The key features are:
- **Term-by-Term Expectation:** The energy is calculated by measuring the expectation value of each Pauli term in the Hamiltonian individually.
- **Qiskit Aer Simulator:** The quantum circuits are executed on Qiskit's high-performance `qasm_simulator`.
- **Classical Optimization:** The `Powell` and `Nelder-Mead` classical optimizers from SciPy are used to find the minimum energy.
- **Multi-Start Approach:** The optimization is run from multiple random starting points to increase the probability of finding the global minimum.
- **Visualization:** The script generates plots for the convergence of the best VQE run and a histogram of the final energies from all runs.

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the VQE simulation, simply execute the Python script:

```bash
python VQE_H2.py
```

## Output

Upon completion, the script will:
1.  Print the best (lowest) ground state energy found and the corresponding optimal parameters to the console.
2.  Save two plot images in the project directory:
    * `vqe_4qubit_h2_convergence.png`: Shows the energy convergence over iterations for the best run.
    * `vqe_4qubit_h2_histogram.png`: A histogram showing the distribution of final energies found from the different starting points.
