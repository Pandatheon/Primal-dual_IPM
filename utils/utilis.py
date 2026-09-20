import json
import copy
import numpy as np
import pandas as pd
import os
from scipy.linalg import block_diag

def write_csv(name, solver, recorder, reg, filename='qp_results.csv'):
    """
    Writes QP problem and solver data to a CSV file using Pandas.

    Parameters:
    QP_name (str): The name of the QP problem.
    QP_solver (dict): A dictionary containing solver details.
                      Must include keys: 'regularization', 'iteration',
                      'cost', 'condition number'.
    filename (str): The output filename, defaults to 'qp_results.csv'.
    """

    # Construct a dictionary for the new row
    # Note: Values are wrapped in lists [] to create a DataFrame
    data = {
        'QP_name': [name],
        'Regularization': [reg],
        'Iteration': [solver.stat_dict['iter_count']],
        'Regularization Count': [int(recorder.reg_count/2)],
        'Cost': [solver.get_cost()],
        'Status': [solver.status],
    }

    # Create a temporary DataFrame for the single row
    new_df = pd.DataFrame(data)

    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Write to CSV
    # mode='a': Append mode
    # header=not file_exists: Write header only if file is new
    # index=False: Do not write row numbers (0, 1, 2...)
    new_df.to_csv(filename, mode='a', header=not file_exists, index=False)

    print(f"Data appended for: {name}")

def test_condensing(QP, M):

    def riccati_recursion(QP):
            P = QP.Q_e
            riccati_H_list = []
            riccati_K_list = []
            L_riccati = np.eye(QP.N * QP.nu)

            for step in range(QP.N):
                k = QP.N - 1 - step
                A_k = QP.A[k]
                B_k = QP.B[k]
                Q_k = QP.Q[k]
                R_k = QP.R[k]
                S_k = QP.S[k]
                reduced_hessian = R_k + B_k.T @ P @ B_k
                riccati_H_list.append(reduced_hessian)

                K = -np.linalg.solve(reduced_hessian, (S_k + B_k.T @ P @ A_k))
                riccati_K_list.append(K)

                QP._record_L_block(QP, L_riccati, step, K)

                P = Q_k + A_k.T @ P @ A_k - (S_k.T + A_k.T @ P @ B_k) @ (-K)

            return riccati_H_list, riccati_K_list, L_riccati

    N = QP.N
    nu = QP.nu
    H_riccati, K_riccati, L_riccati = riccati_recursion(QP)
    M_reconstructed = L_riccati @ block_diag(*H_riccati) @ L_riccati.T
    idx_reversed = np.arange(N * nu).reshape(N, nu)[::-1].ravel()
    M_reversed = M[np.ix_(idx_reversed, idx_reversed)]
    error = np.linalg.norm(M_reconstructed - M_reversed)
    print(f"Riccati reconstruction error: {error:.2e}")
    return error