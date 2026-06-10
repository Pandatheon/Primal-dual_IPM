import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script's location

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
import json

class BaseQP():
    """Base class providing generic QP algorithms and variable structures."""

    def __init__(self, x0_elimination=True):
        # Initialize basic variables
        self.name = ""
        self.x0_elimination = x0_elimination

        # Core QP data structures
        self.N = 0
        self.nx = 0
        self.nu = 0
        self.nb_0 = 0
        self.nbx_0 = 0
        self.nbu_0 = 0
        self.nb = 0
        self.nbx = 0
        self.nbu = 0
        self.nb_e = 0
        self.nbx_e = 0
        self.nbu_e = 0
        self.nh_0 = 0
        self.nh = 0
        self.nh_e = 0
        # cost
        self.Q = None
        self.R = None
        self.S = None
        self.q = None
        self.r = None
        self.Q_e = None
        self.q_e = None
        # constraints
        self.A = None
        self.B = None
        self.b = None
        self.Jbx = None
        self.Jbu = None
        self.C = None
        self.D = None
        self.d = None

        # Algorithmic variables
        self.feedback = None
        self.x0 = None

    def populate_from_json(self, json_filepath):
        """Populate the QP instance using data from a JSON file."""
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        # Use filename as the QP name
        self.name = os.path.splitext(os.path.basename(json_filepath))[0]
        matrix_fields = ['Q', 'R', 'S', 'A', 'B', 'C', 'D']
        vector_fields = ['q', 'r', 'b', 'd', 'idxb', 'lbu', 'ubu', 'lbx', 'ubx', 'lg', 'ug']

        dynamics_fields = ['A', 'B', 'b']
        cost_fields = ['Q', 'R', 'S', 'q', 'r']
        constraints_fields = ['C', 'D', 'idxb', 'lbu', 'ubu', 'lbx', 'ubx', 'lg', 'ug']

        def extract_sequence(prefix):
            seq = []

            # Get the actual padding length
            pad_len = 2
            for key in data:
                if key.startswith(f"{prefix}_"):
                    pad_len = len(key.split('_')[-1])
                    break

            if prefix in cost_fields + dynamics_fields:
                i = 0
                while True:
                    key = f"{prefix}_{i:0{pad_len}d}"
                    if key not in data:
                        break

                    if prefix in matrix_fields:
                        seq.append(np.array(data[key]))
                    elif prefix in vector_fields:
                        seq.append(np.array(data[key]).flatten())
                    i += 1
            else:
                N = len(self.A)
                for k in range(N+1):
                    key = f"{prefix}_{k:0{pad_len}d}"

                    if key in data:
                        if prefix in matrix_fields:
                            seq.append(np.array(data[key]))
                        elif prefix in vector_fields:
                            seq.append(np.array(data[key]).flatten())
                    else:
                        seq.append(np.zeros((0, self.nx if prefix in ['C'] else self.nu)))

            return seq if seq else None

        # Populate dynamics matrices
        self.A = extract_sequence("A")
        self.B = extract_sequence("B")
        self.b = extract_sequence("b")

        # Determine horizon and dimensions
        self.N = len(self.A) if self.A else 0
        self.nx = self.A[0].shape[0] if self.A else 0
        self.nu = self.B[0].shape[1] if self.B else 0

        # Populate cost matrices and vectors
        self.Q = extract_sequence("Q")[:-1]
        self.R = extract_sequence("R")
        self.S = extract_sequence("S")
        self.q = extract_sequence("q")
        self.r = extract_sequence("r")
        self.Q_e = np.array(data[ f"Q_{self.N:02d}"])
        self.q_e = np.array(data[f"q_{self.N:02d}"]).flatten()

        # Populate constraints
        # boundary for x and u, represented as Jbx*x and Jbu*u
        idxb = extract_sequence("idxb")
        self.Jbx = [np.zeros((0, self.nx))] * (self.N + 1)
        self.Jbu = [np.zeros((0, self.nu))] * (self.N + 1)
        if idxb is not None:
            for k in range(len(idxb)):
                idxb_k = idxb[k]
                if idxb_k.size == 0:
                    continue
                idxb_u = idxb_k[idxb_k < self.nu]
                idxb_x = idxb_k[idxb_k >= self.nu] - self.nu
                Jbu_k = np.zeros((len(idxb_u), self.nu))
                Jbu_k[np.arange(len(idxb_u)), idxb_u] = 1.0
                Jbx_k = np.zeros((len(idxb_x), self.nx))
                Jbx_k[np.arange(len(idxb_x)), idxb_x] = 1.0
                self.Jbu[k] = Jbu_k
                self.Jbx[k] = Jbx_k

        if self.x0_elimination:
            # ad-hoc fix for x0 elimination
            self.Jbx[0] = np.zeros((0, self.nx))

        self.nbx_0 = self.Jbx[0].shape[0]
        self.nbu_0 = self.Jbu[0].shape[0]
        self.nb_0 = self.nbx_0 + self.nbu_0

        self.nbx = self.Jbx[1].shape[0]
        self.nbu = self.Jbu[1].shape[0]
        self.nb = self.nbx + self.nbu

        self.nbx_e = self.Jbx[-1].shape[0]
        self.nbu_e = self.Jbu[-1].shape[0]
        self.nb_e = self.nbx_e + self.nbu_e

        # general linear constraints C*x + D*u <= d
        self.C = []
        self.D = []
        C_list = extract_sequence("C")
        D_list = extract_sequence("D")

        for k in range(self.N+1):
            C_k = C_list[k] if (C_list and k < len(C_list)) else np.zeros((0, self.nx))
            D_k = D_list[k] if (D_list and k < len(D_list)) else np.zeros((0, self.nu))
            self.C.append(C_k)
            self.D.append(D_k)

        self.nh_0 = self.C[0].shape[0]
        self.nh = self.C[1].shape[0]
        self.nh_e = self.C[-1].shape[0]

        # concatanate Jbx, Jbu with C, D to form the full constraint matrices for each stage
        for k in range(self.N+1):
            C_lower = np.vstack((np.zeros((self.Jbu[k].shape[0], self.nx)), -self.Jbx[k], -self.C[k]))
            C_upper = np.vstack((np.zeros((self.Jbu[k].shape[0], self.nx)), self.Jbx[k], self.C[k]))
            self.C[k] = np.vstack((C_lower, C_upper))
            D_lower = np.vstack((-self.Jbu[k], np.zeros((self.Jbx[k].shape[0], self.nu)), -self.D[k]))
            D_upper = np.vstack((self.Jbu[k], np.zeros((self.Jbx[k].shape[0], self.nu)), self.D[k]))
            self.D[k] = np.vstack((D_lower, D_upper))

        lbu, ubu = extract_sequence("lbu"), extract_sequence("ubu")
        lbx, ubx= extract_sequence("lbx"), extract_sequence("ubx")
        lg, ug = extract_sequence("lg"), extract_sequence("ug")
        self.d = []
        for k in range(self.N+1):
            lb_k = []
            ub_k = []
            for i, (lb, ub) in enumerate([(lbu, ubu), (lbx, ubx), (lg, ug)]):
                if k == 0 and i ==1 and self.x0_elimination:
                    self.x0 = np.array(lb[0])
                    # ad-hoc fix for x0 elimination
                    b_0_remove = self.b[0] + self.A[0] @ np.array(lb[0])
                    self.b[0] = b_0_remove
                    r_0_remove = self.r[0] + self.S[0] @ np.array(lb[0])
                    self.r[0] = r_0_remove
                    continue

                if (lb is not None and k < len(lb)) and (ub is not None and k < len(ub)):
                    lb_k.append(-lb[k].flatten())
                    ub_k.append(ub[k].flatten())
            if lb_k and ub_k:
                d_k = lb_k + ub_k
                self.d.append(np.concatenate(d_k))
            else:
                self.d.append(np.array([]))

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
        """
        Compute the condensed Hessian matrix using Riccati-based exact condensing.
        This avoids the numerically unstable Gamma.T @ Q_bar @ Gamma operation,
        preventing condition number explosion and reducing complexity.
        """
        # 1. Compute state-input mapping matrix Gamma
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

        # 2. Initialize dense Hessian and terminal cost matrix
        Hc = np.zeros((self.N * self.nu, self.N * self.nu))
        P_next = self.Q_e if self.Q_e is not None else self.Q[-1]

        # Enforce symmetry to prevent numerical drift
        P_next = (P_next + P_next.T) / 2.0

        # 3. Backward Riccati sweep for exact condensing
        for i in range(self.N - 1, -1, -1):
            A_i = self.A[i]
            B_i = self.B[i]

            # Compute cross-coupling matrix M_i
            M_i = self.S[i] + B_i.T @ P_next @ A_i

            # Compute diagonal block Hc[i, i]
            idx_u_i = slice(i * self.nu, (i + 1) * self.nu)
            Hc[idx_u_i, idx_u_i] = self.R[i] + B_i.T @ P_next @ B_i

            # Update Riccati cost matrix P_i for the next iteration (i-1)
            P_next = self.Q[i] + A_i.T @ P_next @ A_i
            P_next = (P_next + P_next.T) / 2.0  # Crucial for numerical stability

            # Compute off-diagonal blocks using the mapping Gamma
            if i > 0:
                idx_x_prev = slice((i - 1) * self.nx, i * self.nx)

                for j in range(i):
                    idx_u_j = slice(j * self.nu, (j + 1) * self.nu)
                    Gamma_ij = Gamma[idx_x_prev, idx_u_j]

                    # Compute interaction between future state cost and past inputs
                    H_ij = M_i @ Gamma_ij

                    # Fill symmetric blocks
                    Hc[idx_u_i, idx_u_j] = H_ij
                    Hc[idx_u_j, idx_u_i] = H_ij.T

        # Ensure the final condensed Hessian is perfectly symmetric
        Hc = (Hc + Hc.T) / 2.0

        return Hc


    def condense_gradient(self, x0, rhs_kkt):
        """
        Compute the condensed gradient vector using Riccati-based exact condensing.
        This avoids the numerically unstable Gamma.T @ g_bar operation,
        preventing condition number explosion and reducing complexity.
        """
        # 1. Forward Pass: Simulate state trajectory with zero control input
        # Complexity: O(N)
        X_sim = np.zeros((self.N + 1, self.nx))
        X_sim[0] = x0
        for k in range(self.N):
            X_sim[k+1] = self.A[k] @ X_sim[k] + rhs_kkt.rb[k]

        # 2. Backward Pass: Adjoint variables and condensed gradient computation
        # Complexity: O(N)
        gc = np.zeros(self.N * self.nu)

        # Terminal condition for the adjoint variable (costate p_N)
        q_N = rhs_kkt.rg[-1][:self.nx]
        p_next = q_N + self.Q_e @ X_sim[self.N]

        for k in range(self.N - 1, -1, -1):
            idx_u = slice(k * self.nu, (k + 1) * self.nu)

            # Extract stage-wise linear penalties
            q_k = rhs_kkt.rg[k][:self.nx]
            r_k = rhs_kkt.rg[k][self.nx:]

            # Compute condensed gradient block for step k
            gc[idx_u] = r_k + self.S[k] @ X_sim[k] + self.B[k].T @ p_next

            # Propagate adjoint variable backwards (skip for k=0)
            if k > 0:
                p_next = q_k + self.Q[k] @ X_sim[k] + self.A[k].T @ p_next
        return gc


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