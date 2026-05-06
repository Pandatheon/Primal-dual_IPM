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
            return seq if seq else None

        # Populate LTV sequences[cite: 5]
        self.A = get_seq("A")
        self.B = get_seq("B")
        self.b = get_seq("b")
        self.C = get_seq("C")
        self.D = get_seq("D")
        self.d = get_seq("d")
        self.Q = get_seq("Q")
        self.R = get_seq("R")
        self.S = get_seq("S")
        self.q = get_seq("q")
        self.r = get_seq("r")

        # Determine N from the length of A sequence[cite: 5]
        self.N = len(self.A) if self.A else 0
        self.nx = self.A[0].shape[0] if self.A else 0
        self.nu = self.B[0].shape[1] if self.B else 0
        self.nh = self.C[0].shape[0] if self.C else 0

        # Determine state and control dimensions[cite: 5]
        if self.A and len(self.A) > 0:
            self.nx = self.A[0].shape[0]
        if self.B and len(self.B) > 0:
            self.nu = self.B[0].shape[1]

        # Terminal costs are typically stored at index N (e.g., Q_20 when N=20)
        terminal_Q_key = f"Q_{self.N:02d}"
        terminal_q_key = f"q_{self.N:02d}"

        if terminal_Q_key in data:
            self.Q_e = np.array(data[terminal_Q_key])
        if terminal_q_key in data:
            self.q_e = np.array(data[terminal_q_key])

        return self

    def _get_stage_matrix(self, mat, k):
        if mat is None:
            return None
        if isinstance(mat, (list, tuple)):
            return mat[k]
        if isinstance(mat, np.ndarray) and mat.ndim == 3:
            return mat[k]
        return mat

    def _get_terminal_matrix(self, mat, fallback=None):
        if mat is None:
            return fallback
        if isinstance(mat, (list, tuple)):
            return mat[-1]
        if isinstance(mat, np.ndarray) and mat.ndim == 3:
            return mat[-1]
        return mat

    def _get_A(self, k):
        return self._get_stage_matrix(self.A, k)

    def _get_B(self, k):
        return self._get_stage_matrix(self.B, k)

    def _get_b(self, k):
        return self._get_stage_matrix(self.b, k)

    def _get_C(self, k):
        return self._get_stage_matrix(self.C, k)

    def _get_D(self, k):
        return self._get_stage_matrix(self.D, k)

    def _get_d(self, k):
        return self._get_stage_matrix(self.d, k)

    def _get_Q(self, k):
        return self._get_stage_matrix(self.Q, k)

    def _get_R(self, k):
        return self._get_stage_matrix(self.R, k)

    def _get_S(self, k):
        return self._get_stage_matrix(self.S, k)

    def _get_q(self, k):
        return self._get_stage_matrix(self.q, k)

    def _get_r(self, k):
        return self._get_stage_matrix(self.r, k)

    def _get_Qe(self):
        fallback = self._get_Q(self.N - 1) if self.N > 0 else None
        return self._get_terminal_matrix(self.Q_e, fallback=fallback)

    def _get_qe(self):
        fallback = self._get_q(self.N - 1) if self.N > 0 else None
        return self._get_terminal_matrix(self.q_e, fallback=fallback)

    def _A_power_transpose(self, start_stage, end_stage):
        if end_stage <= start_stage:
            return np.eye(self.nx)
        A_prod = np.eye(self.nx)
        for k in range(start_stage, end_stage):
            A_prod = self._get_A(k).T @ A_prod
        return A_prod

    def Hessian_block(self, k):
        H_block = np.block([[self._get_Q(k), self._get_S(k).T],
                            [self._get_S(k), self._get_R(k)]])
        return H_block

    def condense_Hessian(self):
        Gamma = np.zeros((self.N * self.nx, self.N * self.nu))

        for i in range(self.N):
            idx_x = slice(i * self.nx, (i + 1) * self.nx)

            for j in range(i + 1):
                idx_u = slice(j * self.nu, (j + 1) * self.nu)
                if i == j:
                    Gamma[idx_x, idx_u] = self._get_B(j)
                else:
                    prev_x = slice((i - 1) * self.nx, i * self.nx)
                    Gamma[idx_x, idx_u] = self._get_A(i) @ Gamma[prev_x, idx_u]

        Q_list = [self._get_Q(k) for k in range(1, self.N)] + [self._get_Qe()]
        Q_bar = block_diag(*Q_list)
        R_list = [self._get_R(k) for k in range(self.N)]
        R_bar = block_diag(*R_list)
        # Build S_bar matrix (N*nu x N*nx) for u_k^T * S_k * x_k
        S_bar = np.zeros((self.N * self.nu, self.N * self.nx))
        for k in range(1, self.N):
            row_idx_u = slice(k * self.nu, (k + 1) * self.nu)  # Maps to u_k
            col_idx_x = slice((k - 1) * self.nx, k * self.nx)  # Maps to x_k
            S_bar[row_idx_u, col_idx_x] = self._get_S(k)

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
            A_stage = self._get_A(stage)

            # Propagate state
            Gamma_x[idx_x, :] = A_stage @ Gamma_x[prev_x, :]

            # Propagate inputs
            for j in range(i):
                idx_u = slice(j * self.nu, (j + 1) * self.nu)
                B_stage = self._get_B(start_idx + j)
                if i - 1 == j:
                    Gamma_u[idx_x, idx_u] = B_stage
                else:
                    Gamma_u[idx_x, idx_u] = A_stage @ Gamma_u[prev_x, idx_u]

        # 2. Build block diagonal cost matrices
        Q_list = [self._get_Q(k) for k in range(start_idx, end_idx + 1)]
        R_list = [self._get_R(k) for k in range(start_idx, end_idx + 1)]
        Q_bar = block_diag(*Q_list)
        R_bar = block_diag(*R_list)
        S_bar = np.zeros((Nc * self.nu, Nc * self.nx))
        for k in range(start_idx, end_idx + 1):
            row_idx_u = slice((k - start_idx) * self.nu, (k - start_idx + 1) * self.nu)
            col_idx_x = slice((k - start_idx) * self.nx, (k - start_idx + 1) * self.nx)
            S_bar[row_idx_u, col_idx_x] = self._get_S(k)

        # 3. Compute condensed cost matrices
        Q_cond = Gamma_x.T @ Q_bar @ Gamma_x
        R_cond = Gamma_u.T @ Q_bar @ Gamma_u + R_bar + S_bar @ Gamma_u + Gamma_u.T @ S_bar.T
        S_cond = Gamma_u.T @ Q_bar @ Gamma_x + S_bar @ Gamma_x

        # 4. Compute condensed dynamics matrices for the transition
        A_cond = self._get_A(end_idx) @ Gamma_x[-self.nx:, :]

        B_cond = np.zeros((self.nx, Nc * self.nu))
        B_cond[:, -self.nu:] = self._get_B(end_idx)
        for j in range(Nc - 1):
            idx_u = slice(j * self.nu, (j + 1) * self.nu)
            B_cond[:, idx_u] = self._get_A(end_idx) @ Gamma_u[-self.nx:, idx_u]

        QP_condensed.A = A_cond
        QP_condensed.B = B_cond
        QP_condensed.Q = Q_cond
        QP_condensed.R = R_cond
        QP_condensed.S = S_cond
        QP_condensed.nx = self.nx
        QP_condensed.nu = Nc * self.nu

        return QP_condensed