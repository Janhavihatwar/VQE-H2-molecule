# VQE Implementation for a 4-Qubit H2 Molecule Hamiltonian
#
# This script finds the ground state energy of the H2 molecule.
# It uses a term-by-term measurement approach and a multi-start classical optimizer.
#

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# ==== Backend ====
# Using the qasm_simulator from Qiskit Aer for simulation.
backend = Aer.get_backend("qasm_simulator")
# Number of measurement shots for each expectation value calculation.
shots = 8192

# ==== H2 Hamiltonian (4-qubit) ====
# The Hamiltonian is defined as a dictionary of Pauli strings and their coefficients.
# The Pauli strings are ordered q3, q2, q1, q0.
coeffs = {
    'IIII': -0.0970663,
    'YYXX': -0.0453026, # From X0 X1 Y2 Y3
    'XYYX':  0.0453026, # From X0 Y1 Y2 X3
    'YXXY':  0.0453026, # From Y0 X1 X2 Y3
    'XXYY': -0.0453026, # From Y0 Y1 X2 X3
    'IIIZ':  0.171413,  # From Z0
    'IIZZ':  0.168689,  # From Z0 Z1
    'IZIZ':  0.120625,  # From Z0 Z2
    'ZIIZ':  0.165928,  # From Z0 Z3
    'IIZI':  0.171413,  # From Z1
    'IZZI':  0.165928,  # From Z1 Z2
    'ZIZI':  0.120625,  # From Z1 Z3
    'IZII': -0.223432,  # From Z2
    'ZZII':  0.174413,  # From Z2 Z3
    'ZIII': -0.223432,  # From Z3
}
num_qubits = 4

# ==== Ansatz (8 parameters, 2 entangling layers) ====
def ansatz(theta):
    """
    Constructs the trial wavefunction (ansatz) circuit.
    This ansatz uses two layers of Ry rotations and CNOT gates.
    """
    qc = QuantumCircuit(num_qubits)
    # Layer 1
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.ry(theta[2], 2)
    qc.ry(theta[3], 3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    # Layer 2
    qc.ry(theta[4], 0)
    qc.ry(theta[5], 1)
    qc.ry(theta[6], 2)
    qc.ry(theta[7], 3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    return qc

# ==== Measurement expectation for one Pauli term ====
def measure_expectation(circuit, pauli_label, shots=8192):
    """
    Measures the expectation value of a single Pauli term in the Hamiltonian.
    """
    qc = circuit.copy()
    # Apply basis transformation gates for X and Y measurements
    for idx, pauli_char in enumerate(pauli_label):
        if pauli_char == 'X':
            qc.h(idx)
        elif pauli_char == 'Y':
            qc.sdg(idx)
            qc.h(idx)

    qc.measure_all()
    qc_t = transpile(qc, backend)
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Calculate expectation value from counts
    exp_val = 0
    for bitstring, count in counts.items():
        # Determine the sign based on the parity of measured qubits for non-Identity Paulis
        sign = 1
        for idx, pauli_char in enumerate(pauli_label):
            if pauli_char != 'I':
                # Qiskit's bitstring is reversed, so we access it as bitstring[::-1]
                if bitstring[::-1][idx] == '1':
                    sign *= -1
        exp_val += sign * count
    return exp_val / shots

# ==== Compute Hamiltonian energy from counts ====
def compute_energy(theta):
    """
    Calculates the total energy of the Hamiltonian for a given set of parameters.
    """
    circuit = ansatz(theta)
    energy_val = 0
    # Sum the expectation values of all Pauli terms
    for pauli_string, coeff in coeffs.items():
        energy_val += coeff * measure_expectation(circuit, pauli_string, shots)
    return energy_val

# ==== Main execution block ====
if __name__ == "__main__":
    # ==== Multi-start optimization ====
    num_starts = 10  # Can be increased for better results, but takes longer
    all_final_energies = []
    best_energy = float('inf')
    best_params = None
    best_convergence = []

    print(f"Starting VQE with {num_starts} random initial points...")

    for run in range(num_starts):
        print(f"--- Run {run + 1}/{num_starts} ---")
        energies = []

        # Wrapper to store energy at each iteration for plotting
        def energy_wrapper(params):
            e = compute_energy(params)
            print(f"  Current energy: {e:.6f}")
            energies.append(e)
            return e

        # Start with random parameters
        init_params = np.random.uniform(0, 2 * np.pi, 8)

        # Use Powell optimizer first
        res = minimize(energy_wrapper, x0=init_params, method="Powell", options={'maxiter': 150})

        # If the result isn't great, try another optimizer from the found point
        if res.fun > -1.1:
            print("Powell result not optimal, trying Nelder-Mead to refine...")
            res = minimize(energy_wrapper, x0=res.x, method="Nelder-Mead", options={'maxiter': 100})

        all_final_energies.append(res.fun)
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
            best_convergence = energies

    print("\n==== VQE Complete ====")
    print(f"Best energy: {best_energy:.6f} Hartree")
    print(f"Best parameters: {best_params}")

    # ==== Plot best convergence ====
    plt.figure(figsize=(10, 6))
    plt.plot(best_convergence, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.title("VQE Convergence (Best Run)")
    plt.grid(True)
    plt.savefig("vqe_4qubit_h2_convergence.png")
    print("Convergence plot saved to 'vqe_4qubit_h2_convergence.png'")

    # ==== Plot histogram of final energies from all runs ====
    plt.figure(figsize=(10, 6))
    plt.hist(all_final_energies, bins=10, edgecolor='black')
    plt.xlabel("Final Energy (Hartree)")
    plt.ylabel("Count")
    plt.axvline(best_energy, color='r', linestyle='--', label=f'Best Energy: {best_energy:.4f}')
    plt.title(f"Distribution of Final Energies from {num_starts} Multi-Start Runs")
    plt.legend()
    plt.grid(True)
    plt.savefig("vqe_4qubit_h2_histogram.png")
    print("Histogram plot saved to 'vqe_4qubit_h2_histogram.png'")
