from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import do_regularization
from solvers.Regularization_config import Regularizations_config
from utils.modified_Cholesky import modified_cholesky_SE
import numpy as np
import copy

from scipy.linalg import block_diag


class Condensed_IPM():
    def __init__(self, qp: BaseQP, reg_config: Regularizations_config, ineq_num:int):
        self.qp = qp
        self.reg_config = reg_config
        self.ineq_num = ineq_num
        self.rhs_kkt = None

    def forward_sweep(self, prev_sol:iterate_sol, feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*self.qp.N, [None]*self.qp.N
        lam_i = prev_sol.lam
        s_i = prev_sol.s

        condensed_Hessian, gc = feedback
        try:
            Hc_chol = np.linalg.cholesky(condensed_Hessian)
        except np.linalg.LinAlgError:
            if self.reg_config.method == 'PROJECT_M':
                condensed_Hessian_reg = do_regularization(condensed_Hessian, self.reg_config.delta, method='PROJECT')
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            elif self.reg_config.method == 'MIRROR_M':
                condensed_Hessian_reg = do_regularization(condensed_Hessian, self.reg_config.delta, method='MIRROR')
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            elif self.reg_config.method == 'CHOLESKY_M':
                L, D, E, p = modified_cholesky_SE(condensed_Hessian)
                p_inv = np.argsort(p)
                condensed_Hessian_reg = (L @ np.diag(D) @ L.T)[np.ix_(p_inv, p_inv)]
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            else:
                raise ValueError(f"Unknown regularization method: {self.reg_config.method}, set self.reg_config.method to 'PROJECT_H_M', 'MIRROR_H_M' or 'CHOLESKY_M'")

        y1 = np.linalg.solve(Hc_chol, -gc)
        delta_u_bar = np.linalg.solve(Hc_chol.T, y1)
        delta_u = delta_u_bar.reshape((self.qp.N, self.qp.nu))

        delta_x[0] = np.zeros(self.qp.nx)
        for k in range(self.qp.N):
            C_k = self.qp.C[k]
            D_k = self.qp.D[k]
            rhs_b = self.rhs_kkt.rb[k]
            rhs_d= self.rhs_kkt.rd[k]
            rhs_m = self.rhs_kkt.rm[k]

            # recover x and u
            delta_x[k+1] = self.qp.A[k]@delta_x[k] + self.qp.B[k]@delta_u[k] + rhs_b

            # recover s
            delta_s[k] = -(C_k @delta_x[k] + D_k @delta_u[k]) - rhs_d

            # recover lam
            rhs_m_modified = -rhs_m - (lam_i[k] * delta_s[k])
            delta_lam[k] = (rhs_m_modified/s_i[k]).flatten()

        delta_pi[self.qp.N] = self.rhs_kkt.rg[-1][:self.qp.nx] + self.qp.Q_e @ delta_x[self.qp.N]
        for k in reversed(range(self.qp.N)):
            A_k = self.qp.A[k]
            C_k = self.qp.C[k]
            Q_k = self.qp.Q[k]
            S_k = self.qp.S[k]
            delta_pi[k] = Q_k @ delta_x[k] + S_k.T @ delta_u[k] + A_k.T @ delta_pi[k+1] + C_k.T @ delta_lam[k] + self.rhs_kkt.rg[k][:self.qp.nx]

        delta_dict = {
            'delta_x': delta_x,
            'delta_u': delta_u,
            'delta_pi': delta_pi,
            'delta_lam': delta_lam,
            'delta_s': delta_s,
        }
        return delta_dict

    def backward_sweep(self, rhs, prev_sol:iterate_sol)-> Feedback:
        x0 = np.zeros(self.qp.nx)
        nx = self.qp.nx
        self.rhs_kkt = copy.deepcopy(rhs)

        modified_qp = self.modify_QP(self.qp, prev_sol)

        condensed_Hessian = modified_qp.condense_Hessian()

        Gamma = np.zeros((modified_qp.N * modified_qp.nx, modified_qp.N * modified_qp.nu))
        Phi = np.zeros((modified_qp.N * modified_qp.nx, modified_qp.nx))
        upsilon_b = np.zeros(modified_qp.N * modified_qp.nx)

        for k in range(modified_qp.N):
            A_k = modified_qp.A[k]
            b_k = self.rhs_kkt.rb[k]
            idx_x = slice(k * modified_qp.nx, (k + 1) * modified_qp.nx)

            if k == 0:
                Phi[idx_x, :] = A_k
                upsilon_b[idx_x] = b_k
            else:
                prev_x = slice((k - 1) * modified_qp.nx, k * modified_qp.nx)
                Phi[idx_x, :] = modified_qp.A[k] @ Phi[prev_x, :]
                upsilon_b[idx_x] = A_k @ upsilon_b[prev_x] + b_k

            # Gamma matrix (same as in condense_Hessian)
            for j in range(k + 1):
                idx_u = slice(j * modified_qp.nu, (j + 1) * modified_qp.nu)
                if k == j:
                    Gamma[idx_x, idx_u] = modified_qp.B[j]
                else:
                    prev_x = slice((k - 1) * modified_qp.nx, k * modified_qp.nx)
                    Gamma[idx_x, idx_u] = modified_qp.A[k] @ Gamma[prev_x, idx_u]

        # Stack Q and q
        Q_list = [modified_qp.Q[k] for k in range(1, modified_qp.N)]
        Q_list.append(modified_qp.Q_e)
        Q_bar = block_diag(*Q_list)

        q_list = [self.rhs_kkt.rg[k][:nx] for k in range(1, modified_qp.N)]
        q_list.append(self.rhs_kkt.rg[-1][:nx])
        q_bar = np.concatenate(q_list)

        r_list = [self.rhs_kkt.rg[k][nx:] for k in range(modified_qp.N)]
        r_bar = np.concatenate(r_list)

        X_zero_input = Phi @ x0 + upsilon_b
        
        # Build S_bar for k = 1 to N-1
        S_bar = np.zeros((modified_qp.N * modified_qp.nu, modified_qp.N * modified_qp.nx))
        for k in range(1, modified_qp.N):
            row_idx_u = slice(k * modified_qp.nu, (k + 1) * modified_qp.nu)
            col_idx_x = slice((k - 1) * modified_qp.nx, k * modified_qp.nx)
            S_bar[row_idx_u, col_idx_x] = modified_qp.S[k]

        # Calculate condensed gradient including S contributions
        gc = r_bar + Gamma.T @ (q_bar + Q_bar @ X_zero_input) + S_bar @ X_zero_input

        # Add contribution of S_0 @ x_0 to the first control step u_0
        gc[:modified_qp.nu] += modified_qp.S[0] @ x0

        return (condensed_Hessian, gc)

    def modify_QP(self, qp, prev_sol:iterate_sol):
        modified_QP = copy.deepcopy(qp)

        for k in range(qp.N):
            s_k = prev_sol.s[k]
            lam_k = prev_sol.lam[k]
            C_k = qp.C[k]
            D_k = qp.D[k]

            C_full = np.hstack((C_k, D_k))
            T_mat = np.diag(s_k)
            Lam_mat = np.diag(lam_k)
            rd = self.rhs_kkt.rd[k]
            rm = self.rhs_kkt.rm[k]

            T_mat_inv = np.linalg.inv(T_mat)
            lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
            rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

            Q_tilt = lhs_modification[:qp.nx, :qp.nx]
            R_tilt = lhs_modification[qp.nx:, qp.nx:]
            S_tilt = lhs_modification[qp.nx:, :qp.nx]
            modified_QP.Q[k] = qp.Q[k] + Q_tilt
            modified_QP.R[k] = qp.R[k] + R_tilt
            modified_QP.S[k] = qp.S[k] + S_tilt

            self.rhs_kkt.rg[k][:qp.nx] = self.rhs_kkt.rg[k][:qp.nx] + rhs_modification[:qp.nx]
            self.rhs_kkt.rg[k][qp.nx:] = self.rhs_kkt.rg[k][qp.nx:] + rhs_modification[qp.nx:]
        self.rhs_kkt.rg[-1] = self.rhs_kkt.rg[-1]
        return modified_QP