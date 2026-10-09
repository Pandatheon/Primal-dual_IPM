import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script's location

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
from acados_template.acados_ocp_qp import AcadosOcpQp

class BaseQP():
    """Base class providing generic QP algorithms and variable structures."""

    def __init__(self, x0_elimination=True):
        # Initialize basic variables
        self.name = ""
        self.x0_elimination = x0_elimination

        # Core QP data structures
        self.N = 0
        self.nx = [None] * (self.N + 1)
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
        self.row_mask = None

        # Algorithmic variables
        self.feedback = None
        self.x0 = None
        self.ocp_qp = None

    def populate_from_json(self, json_filepath):
        """Populate the QP instance using data from an acados QP JSON file."""
        self.populate_from_ocp_qp(AcadosOcpQp.from_json(json_filepath))
        self.name = os.path.splitext(os.path.basename(json_filepath))[0]
        return self

    def populate_from_ocp_qp(self, ocp_qp: AcadosOcpQp):
        """Populate the QP instance from an acados AcadosOcpQp. The arrays are copied, ocp_qp is left untouched."""
        self.ocp_qp = ocp_qp
        dims = ocp_qp.dims

        self.N = ocp_qp.N
        self.nx = [int(nx_k) for nx_k in dims.nx]
        self.nu = int(dims.nu[0])

        # Populate dynamics matrices
        self.A = [A_k.copy() for A_k in ocp_qp.A]
        self.B = [B_k.copy() for B_k in ocp_qp.B]
        self.b = [b_k.copy() for b_k in ocp_qp.b]

        self.Q = [Q_k.copy() for Q_k in ocp_qp.Q[:-1]]
        self.R = [R_k.copy() for R_k in ocp_qp.R[:-1]]
        self.S = [S_k.copy() for S_k in ocp_qp.S[:-1]]
        self.q = [q_k.copy() for q_k in ocp_qp.q[:-1]]
        self.r = [r_k.copy() for r_k in ocp_qp.r[:-1]]
        self.Q_e = ocp_qp.Q[-1].copy()
        self.q_e = ocp_qp.q[-1].copy()

        self.Jbx = [np.zeros((0, self.nx[k])) for k in range(self.N + 1)]
        self.Jbu = [np.zeros((0, self.nu))] * (self.N + 1)
        for k, idxb_k in enumerate(ocp_qp.idxb):
            if idxb_k.size == 0:
                continue
            nu_k = dims.nu[k]
            idxb_u = idxb_k[idxb_k < nu_k]
            idxb_x = idxb_k[idxb_k >= nu_k] - nu_k
            Jbu_k = np.zeros((len(idxb_u), self.nu))
            Jbu_k[np.arange(len(idxb_u)), idxb_u] = 1.0
            Jbx_k = np.zeros((len(idxb_x), self.nx[k]))
            Jbx_k[np.arange(len(idxb_x)), idxb_x] = 1.0
            self.Jbu[k] = Jbu_k
            self.Jbx[k] = Jbx_k

        if self.x0_elimination:
            # ad-hoc fix for x0 elimination
            self.idxe_0 = ocp_qp.idxe[0] - self.Jbu[0].shape[0]
            self.free_idx_0 = np.delete(np.arange(self.nx[0]), self.idxe_0)
            self.A0e = self.A[0][:, self.idxe_0].copy()
            self.S0e = self.S[0][:, self.idxe_0].copy()
            self.Q0e = self.Q[0][self.idxe_0, :].copy()
            self.q0e = self.q[0][self.idxe_0].copy()
            # eliminate dimension corresponding to idexe_0 in Jbx[0]
            if self.idxe_0.size > 0:
                Jbx_0_new = np.delete(self.Jbx[0], self.idxe_0, axis=0)
                self.Jbx[0] = Jbx_0_new

        self.nbx_0 = self.Jbx[0].shape[0]
        self.nbu_0 = self.Jbu[0].shape[0]
        self.nb_0 = self.nbx_0 + self.nbu_0

        self.nbx = self.Jbx[1].shape[0]
        self.nbu = self.Jbu[1].shape[0]
        self.nb = self.nbx + self.nbu

        self.nbx_e = self.Jbx[-1].shape[0]
        self.nbu_e = self.Jbu[-1].shape[0]
        self.nb_e = self.nbx_e + self.nbu_e

        self.C = []
        self.D = []
        for k in range(self.N+1):
            has_g = dims.ng[k] > 0
            C_k = ocp_qp.C[k].copy() if has_g else np.zeros((0, self.nx[k]))
            D_k = ocp_qp.D[k].copy() if has_g and k < self.N else np.zeros((0, self.nu))
            self.C.append(C_k)
            self.D.append(D_k)

        self.nh_0 = self.C[0].shape[0]
        self.nh = self.C[1].shape[0]
        self.nh_e = self.C[-1].shape[0]

        for k in range(self.N+1):
            C_lower = np.vstack((np.zeros((self.Jbu[k].shape[0], self.nx[k])), -self.Jbx[k], -self.C[k]))
            C_upper = np.vstack((np.zeros((self.Jbu[k].shape[0], self.nx[k])), self.Jbx[k], self.C[k]))
            self.C[k] = np.vstack((C_lower, C_upper))
            D_lower = np.vstack((-self.Jbu[k], np.zeros((self.Jbx[k].shape[0], self.nu)), -self.D[k]))
            D_upper = np.vstack((self.Jbu[k], np.zeros((self.Jbx[k].shape[0], self.nu)), self.D[k]))
            self.D[k] = np.vstack((D_lower, D_upper))

        lbu, ubu = ocp_qp.lbu, ocp_qp.ubu
        lbx, ubx = ocp_qp.lbx, ocp_qp.ubx
        lg, ug = ocp_qp.lg, ocp_qp.ug
        masks = [(ocp_qp.lbu_mask, ocp_qp.ubu_mask), (ocp_qp.lbx_mask, ocp_qp.ubx_mask), (ocp_qp.lg_mask, ocp_qp.ug_mask)]
        self.d = []

        self.row_mask = []
        for k in range(self.N+1):
            lb_k = []
            ub_k = []
            lb_mask_k = []
            ub_mask_k = []
            for i, (lb, ub) in enumerate([(lbu, ubu), (lbx, ubx), (lg, ug)]):
                lb_mask, ub_mask = masks[i]
                if k == 0 and i == 1 and self.x0_elimination:
                    self.x0 = lb[0][self.idxe_0]
                    self.b[0] = self.b[0] + self.A[0][:, self.idxe_0] @ self.x0
                    self.r[0] = self.r[0] + self.S[0][:, self.idxe_0] @ self.x0
                    self.q[0] = self.q[0][self.free_idx_0] + self.Q[0][np.ix_(self.free_idx_0, self.idxe_0)] @ self.x0
                    self.A[0] = self.A[0][:, self.free_idx_0]
                    self.S[0] = self.S[0][:, self.free_idx_0]
                    self.Q[0] = self.Q[0][np.ix_(self.free_idx_0, self.free_idx_0)]
                    lb_k.append(-lb[0][self.free_idx_0])
                    ub_k.append( ub[0][self.free_idx_0])
                    lb_mask_k.append(lb_mask[0][self.free_idx_0])
                    ub_mask_k.append(ub_mask[0][self.free_idx_0])
                    continue

                lb_k.append(-lb[k])
                ub_k.append(ub[k])
                lb_mask_k.append(lb_mask[k])
                ub_mask_k.append(ub_mask[k])

            self.d.append(np.concatenate(lb_k + ub_k))
            self.row_mask.append(np.concatenate(lb_mask_k + ub_mask_k) != 0.0)

        if self.x0_elimination:
            self.d[0] = self.d[0] - self.C[0][:, self.idxe_0] @ self.x0
            self.C[0] = self.C[0][:, self.free_idx_0]
            self.nx[0] = len(self.free_idx_0)

        self.C_all, self.D_all, self.d_all = self.C, self.D, self.d
        self.D_all[self.N] = np.zeros((self.d_all[self.N].size, self.nu))
        self.C = [C_k[m] for C_k, m in zip(self.C_all, self.row_mask)]
        self.D = [D_k[m] for D_k, m in zip(self.D_all, self.row_mask)]
        self.d = [d_k[m] for d_k, m in zip(self.d_all, self.row_mask)]

        return self

    def __deepcopy__(self, memo):
        memo[id(self.ocp_qp)] = self.ocp_qp
        QP_copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = QP_copy
        for key, value in self.__dict__.items():
            setattr(QP_copy, key, deepcopy(value, memo))
        return QP_copy

    def _A_power_transpose(self, start_stage, end_stage):
        if end_stage <= start_stage:
            return np.eye(self.nx[k])
        A_prod = np.eye(self.nx[k])
        for k in range(start_stage, end_stage):
            A_prod = self.A[k].T @ A_prod
        return A_prod

    def Hessian_block(self, k):
        H_block = np.block([
            [self.Q[k], self.S[k].T],
            [self.S[k], self.R[k]],
        ])
        return H_block

    def Riccati_recursion(self):
        """
        Perform Riccati recursion to compute the feedback gains and cost-to-go matrices.
        This method computes the optimal control law for the QP using dynamic programming.
        """
        # Initialize storage for feedback gains and cost-to-go matrices
        M_list = [None] * self.N
        K_list = [None] * self.N
        P_list = [None] * (self.N + 1)

        # Terminal cost
        P_list[self.N] = self.Q_e

        for k in range(self.N - 1, -1, -1):
            # Compute the Riccati recursion matrices
            A_k = self.A[k]
            B_k = self.B[k]
            Q_k = self.Q[k]
            R_k = self.R[k]
            S_k = self.S[k]

            # Compute the cost-to-go matrix P_k
            P_next = P_list[k + 1]
            M_k = R_k + B_k.T @ P_next @ B_k
            M_k = 0.5*(M_k + M_k.T)
            G_k = S_k + B_k.T @ P_next @ A_k
            K_k = np.linalg.solve(M_k, G_k)  # Feedback gain
            P_curr = Q_k + A_k.T @ P_next @ A_k - K_k.T @ (R_k + B_k.T @ P_next @ B_k) @ K_k
            P_curr = 0.5*(P_curr + P_curr.T)  # Ensure symmetry
            P_list[k] = P_curr

            # Store feedback gain and cross-coupling matrix
            M_list[k] = M_k
            K_list[k] = K_k

        return P_list, K_list, M_list

    def condense_Hessian(self):
        """
        Compute the condensed Hessian matrix using Riccati-based exact condensing.
        This avoids the numerically unstable Gamma.T @ Q_bar @ Gamma operation,
        preventing condition number explosion and reducing complexity.
        """
        # 1. Compute state-input mapping matrix Gamma
        Gamma = np.zeros((self.N * self.nx[k], self.N * self.nu))
        for i in range(self.N):
            idx_x = slice(i * self.nx[k], (i + 1) * self.nx[k])
            for j in range(i + 1):
                idx_u = slice(j * self.nu, (j + 1) * self.nu)
                if i == j:
                    Gamma[idx_x, idx_u] = self.B[j]
                else:
                    prev_x = slice((i - 1) * self.nx[k], i * self.nx[k])
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
                idx_x_prev = slice((i - 1) * self.nx[k], i * self.nx[k])

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
        X_sim = np.zeros((self.N + 1, self.nx[k]))
        X_sim[0] = x0
        for k in range(self.N):
            X_sim[k+1] = self.A[k] @ X_sim[k] + rhs_kkt.rb[k]

        # 2. Backward Pass: Adjoint variables and condensed gradient computation
        # Complexity: O(N)
        gc = np.zeros(self.N * self.nu)

        # Terminal condition for the adjoint variable (costate p_N)
        q_N = rhs_kkt.rg[-1][:self.nx[k]]
        p_next = q_N + self.Q_e @ X_sim[self.N]

        for k in range(self.N - 1, -1, -1):
            idx_u = slice(k * self.nu, (k + 1) * self.nu)

            # Extract stage-wise linear penalties
            q_k = rhs_kkt.rg[k][:self.nx[k]]
            r_k = rhs_kkt.rg[k][self.nx[k]:]

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

        Gamma_x = np.zeros((Nc * self.nx[k], self.nx[k]))
        Gamma_u = np.zeros((Nc * self.nx[k], Nc * self.nu))

        # 1. Iteratively build Gamma_x and Gamma_u
        Gamma_x[0:self.nx[k], :] = np.eye(self.nx[k])

        for i in range(1, Nc):
            idx_x = slice(i * self.nx[k], (i + 1) * self.nx[k])
            prev_x = slice((i - 1) * self.nx[k], i * self.nx[k])
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
        S_bar = np.zeros((Nc * self.nu, Nc * self.nx[k]))
        for k in range(start_idx, end_idx + 1):
            row_idx_u = slice((k - start_idx) * self.nu, (k - start_idx + 1) * self.nu)
            col_idx_x = slice((k - start_idx) * self.nx[k], (k - start_idx + 1) * self.nx[k])
            S_bar[row_idx_u, col_idx_x] = self.S[k]

        # 3. Compute condensed cost matrices
        Q_cond = Gamma_x.T @ Q_bar @ Gamma_x
        R_cond = Gamma_u.T @ Q_bar @ Gamma_u + R_bar + S_bar @ Gamma_u + Gamma_u.T @ S_bar.T
        S_cond = Gamma_u.T @ Q_bar @ Gamma_x + S_bar @ Gamma_x

        # 4. Compute condensed dynamics matrices for the transition
        A_cond = self.A[end_idx] @ Gamma_x[-self.nx[k]:, :]

        B_cond = np.zeros((self.nx[k], Nc * self.nu))
        B_cond[:, -self.nu:] = self.B[end_idx]
        for j in range(Nc - 1):
            idx_u = slice(j * self.nu, (j + 1) * self.nu)
            B_cond[:, idx_u] = self.A[end_idx] @ Gamma_u[-self.nx[k]:, idx_u]

        QP_condensed.A = A_cond
        QP_condensed.B = B_cond
        QP_condensed.Q = Q_cond
        QP_condensed.R = R_cond
        QP_condensed.S = S_cond
        QP_condensed.nx = self.nx[k]
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