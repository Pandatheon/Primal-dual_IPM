import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script's location

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
import json

class BaseQP():
    """Base class providing generic QP algorithms and variable structures."""

    def __init__(self):
        # Initialize basic variables
        self.name = ""

        # Core QP data structures
        self.N = 0
        self.A = None
        self.B = None
        self.b = None
        self.C = None
        self.D = None
        self.d = None
        self.nx = 0
        self.nu = 0
        self.Q = None
        self.R = None
        self.S = None
        self.q = None
        self.r = None
        self.Q_e = None
        self.q_e = None

        # Algorithmic variables
        self.feedback = None

    def populate_from_json(self, json_filepath):
        """Populate the QP instance using data from a JSON file."""
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        # Use filename as the QP name
        self.name = os.path.splitext(os.path.basename(json_filepath))[0]

        # Helper to dynamically extract sequences like A_00, A_01...
        def get_seq(prefix):
            seq = []
            i = 0
            while f"{prefix}_{i:02d}" in data:
                seq.append(np.array(data[f"{prefix}_{i:02d}"]))
                i += 1
            return seq or None

        # Populate LTV sequences[cite: 5]
        for key in ["A", "B", "b", "C", "D", "d", "Q", "R", "S", "q", "r"]:
            setattr(self, key, get_seq(key))

        # Determine horizon and dimensions
        self.N = len(self.A) if self.A else 0
        self.nx = self.A[0].shape[0] if self.A else 0
        self.nu = self.B[0].shape[1] if self.B else 0
        self.nh = self.C[0].shape[0] if self.C else 0

        # Terminal costs are typically stored at index N (e.g., Q_20 when N=20)
        terminal_Q_key = f"Q_{self.N:02d}"
        terminal_q_key = f"q_{self.N:02d}"

        if terminal_Q_key in data:
            self.Q_e = np.array(data[terminal_Q_key])
        if terminal_q_key in data:
            self.q_e = np.array(data[terminal_q_key])

        return self

    def _A_power_transpose(self, start_stage, end_stage):
        if end_stage <= start_stage:
            return np.eye(self.nx)
        A_prod = np.eye(self.nx)
        for k in range(start_stage, end_stage):
            A_prod = self.A[k].T @ A_prod
        return A_prod

    def Hessian_block(self, k):
        H_block = np.block([
            [self.Q[k], self.S[k].T],
            [self.S[k], self.R[k]],
        ])
        return H_block

    def condense_Hessian(self):
        Gamma = np.zeros((self.N * self.nx, self.N * self.nu))

        for i in range(self.N):
            idx_x = slice(i * self.nx, (i + 1) * self.nx)

            for j in range(i + 1):
                idx_u = slice(j * self.nu, (j + 1) * self.nu)
                if i == j:
                    Gamma[idx_x, idx_u] = self.B[j]
                else:
                    prev_x = slice((i - 1) * self.nx, i * self.nx)
                    Gamma[idx_x, idx_u] = self.A[i] @ Gamma[prev_x, idx_u]

        Q_list = [self.Q[k] for k in range(1, self.N)]
        Qe_fallback = self.Q[self.N - 1] if self.N > 0 else None
        Q_list.append(self.Q_e if self.Q_e is not None else Qe_fallback)
        Q_bar = block_diag(*Q_list)
        R_list = [self.R[k] for k in range(self.N)]
        R_bar = block_diag(*R_list)
        # Build S_bar matrix (N*nu x N*nx) for u_k^T * S_k * x_k
        S_bar = np.zeros((self.N * self.nu, self.N * self.nx))
        for k in range(1, self.N):
            row_idx_u = slice(k * self.nu, (k + 1) * self.nu)  # Maps to u_k
            col_idx_x = slice((k - 1) * self.nx, k * self.nx)  # Maps to x_k
            S_bar[row_idx_u, col_idx_x] = self.S[k]

        # Pure quadratic term computation including S
        Hc = Gamma.T @ Q_bar @ Gamma + R_bar + S_bar @ Gamma + Gamma.T @ S_bar.T
        return Hc

    def partial_condense_range(self, start_idx, end_idx):
        """
        Build condensed matrices from start_idx to end_idx.
        start_idx: starting stage index.
        end_idx: ending stage index.
        """
        QP_condensed = deepcopy(self)

        # Calculate block size based on the specified range
        Nc = end_idx - start_idx + 1

        Gamma_x = np.zeros((Nc * self.nx, self.nx))
        Gamma_u = np.zeros((Nc * self.nx, Nc * self.nu))

        # 1. Iteratively build Gamma_x and Gamma_u
        Gamma_x[0:self.nx, :] = np.eye(self.nx)

        for i in range(1, Nc):
            idx_x = slice(i * self.nx, (i + 1) * self.nx)
            prev_x = slice((i - 1) * self.nx, i * self.nx)
            stage = start_idx + i - 1
            A_stage = self.A[stage]

            # Propagate state
            Gamma_x[idx_x, :] = A_stage @ Gamma_x[prev_x, :]

            # Propagate inputs
            for j in range(i):
                idx_u = slice(j * self.nu, (j + 1) * self.nu)
                B_stage = self.B[start_idx + j]
                if i - 1 == j:
                    Gamma_u[idx_x, idx_u] = B_stage
                else:
                    Gamma_u[idx_x, idx_u] = A_stage @ Gamma_u[prev_x, idx_u]

        # 2. Build block diagonal cost matrices
        Q_list = [self.Q[k] for k in range(start_idx, end_idx + 1)]
        R_list = [self.R[k] for k in range(start_idx, end_idx + 1)]
        Q_bar = block_diag(*Q_list)
        R_bar = block_diag(*R_list)
        S_bar = np.zeros((Nc * self.nu, Nc * self.nx))
        for k in range(start_idx, end_idx + 1):
            row_idx_u = slice((k - start_idx) * self.nu, (k - start_idx + 1) * self.nu)
            col_idx_x = slice((k - start_idx) * self.nx, (k - start_idx + 1) * self.nx)
            S_bar[row_idx_u, col_idx_x] = self.S[k]

        # 3. Compute condensed cost matrices
        Q_cond = Gamma_x.T @ Q_bar @ Gamma_x
        R_cond = Gamma_u.T @ Q_bar @ Gamma_u + R_bar + S_bar @ Gamma_u + Gamma_u.T @ S_bar.T
        S_cond = Gamma_u.T @ Q_bar @ Gamma_x + S_bar @ Gamma_x

        # 4. Compute condensed dynamics matrices for the transition
        A_cond = self.A[end_idx] @ Gamma_x[-self.nx:, :]

        B_cond = np.zeros((self.nx, Nc * self.nu))
        B_cond[:, -self.nu:] = self.B[end_idx]
        for j in range(Nc - 1):
            idx_u = slice(j * self.nu, (j + 1) * self.nu)
            B_cond[:, idx_u] = self.A[end_idx] @ Gamma_u[-self.nx:, idx_u]

        QP_condensed.A = A_cond
        QP_condensed.B = B_cond
        QP_condensed.Q = Q_cond
        QP_condensed.R = R_cond
        QP_condensed.S = S_cond
        QP_condensed.nx = self.nx
        QP_condensed.nu = Nc * self.nu

        return QP_condensed

    @staticmethod
    def _record_L_block(QP, L_riccati, step, K, actual_range=1):
        k_end = QP.N - 1 - step
        k_start = k_end - actual_range + 1
        idx_col = slice(step * QP.nu, (step + actual_range) * QP.nu)
        for next_step in range(step + actual_range, QP.N):
            j = QP.N - 1 - next_step
            idx_row = slice(next_step * QP.nu, (next_step + 1) * QP.nu)
            Phi_T = np.eye(QP.nx)
            for i in range(j + 1, k_start):
                Phi_T = Phi_T @ QP.A[i].T
            L_jk_ric = -QP.B[j].T @ Phi_T @ K.T
            L_riccati[idx_row, idx_col] = L_jk_ric