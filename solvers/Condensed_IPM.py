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

    def update_rhs(self, centering_param:float, rhs:dict, statistics_dict:dict,
                   prev_sol:iterate_sol, delta_aff_dict:dict = None, step_type:str='predictor')-> np.ndarray:
        nx = self.qp.nx
        nu = self.qp.nu

        res_stat = []
        res_eq = []
        res_ineq = []
        res_comp = []
        duality_measure = 0.0

        for k in range(self.qp.N):
            Q_k = self.qp.Q[k]
            R_k = self.qp.R[k]
            S_k = self.qp.S[k]
            A_k = self.qp.A[k]
            B_k = self.qp.B[k]
            b_k = self.qp.b[k]
            C_k = self.qp.C[k]
            D_k = self.qp.D[k]
            d_k = self.qp.d[k]
            q_k = self.qp.q[k]
            r_k = self.qp.r[k]
            x, u, pi, lam, s = prev_sol

            # stationarity residual
            temp_res_stat = np.zeros(nx + nu)
            temp_res_stat[:nx] = Q_k @ x[k] + S_k.T @ u[k] + q_k + A_k.T @ pi[k+1] + C_k.T @ lam[k] - pi[k]
            temp_res_stat[nx:] = R_k @ u[k] + S_k @ x[k] + r_k + B_k.T @ pi[k+1] + D_k.T @ lam[k]
            res_stat.append(temp_res_stat)

            #equality residual
            res_eq.append(A_k @ x[k] + B_k @ u[k] - x[k+1] + b_k)

            #inequality residual
            res_ineq.append((C_k @ x[k] + D_k @ u[k] + s[k]) - d_k)

            #complementarity residual
            if step_type == 'predictor':
                temp_res_comp = (np.diag(lam[k]) @ s[k]).flatten()
                duality_measure += (lam[k] @ s[k])
            elif step_type == 'centering':
                temp_res_comp = (lam[k] * s[k]).flatten() - centering_param * rhs.duality_measure * np.ones(len(lam[k]))
            elif step_type == 'predictor-corrector':
                temp_res_comp = (lam[k] * s[k]).flatten() + (delta_aff_dict['delta_lam'][k] * delta_aff_dict['delta_s'][k]) - centering_param * rhs.duality_measure * np.ones(len(lam[k]))
            res_comp.append(temp_res_comp)

        # terminal residual
        temp_res_stat = np.zeros(nx+nu)
        Q_e = self.qp.Q_e
        q_e = self.qp.q_e
        temp_res_stat[:nx] = Q_e @ x[self.qp.N] + q_e - pi[self.qp.N]
        res_stat.append(temp_res_stat)

        rhs.rg = res_stat
        rhs.rb = res_eq
        rhs.rd = res_ineq
        rhs.rm = res_comp
        if step_type == 'predictor':
            rhs.duality_measure = max(duality_measure / self.ineq_num, 1e-16)
            statistics_dict['res_stat_list'].append(np.linalg.norm(np.concatenate(rhs.rg), ord=np.inf))
            statistics_dict['res_eq_list'].append(np.linalg.norm(np.concatenate(rhs.rb), ord=np.inf))
            statistics_dict['res_ineq_list'].append(np.linalg.norm(np.concatenate(rhs.rd), ord=np.inf))
            statistics_dict['res_comp_list'].append(np.linalg.norm(np.concatenate(rhs.rm), ord=np.inf))
            statistics_dict['duality_measure_list'].append(rhs.duality_measure)
        return rhs

    def foward_sweep(self, rhs, prev_sol:iterate_sol, feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*self.qp.N, [None]*self.qp.N
        lam_i = prev_sol.lam
        s_i = prev_sol.s

        condensed_Hessian, gc = feedback
        try:
            Hc_chol = np.linalg.cholesky(condensed_Hessian)
        except np.linalg.LinAlgError:
            if self.reg_config.method == 'M_PROJECT':
                eigvals, eigvecs = np.linalg.eig(condensed_Hessian)
                eigvals_reg = do_regularization(eigvals, self.reg_config.delta, 'PROJECT')
                condensed_Hessian_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            elif self.reg_config.method == 'M_MIRROR':
                eigvals, eigvecs = np.linalg.eig(condensed_Hessian)
                eigvals_reg = do_regularization(eigvals, self.reg_config.delta, 'MIRROR')
                condensed_Hessian_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            elif self.reg_config.method == 'M_CM':
                L, D, E, p = modified_cholesky_SE(condensed_Hessian)
                p_inv = np.argsort(p)
                condensed_Hessian_reg = (L @ np.diag(D) @ L.T)[np.ix_(p_inv, p_inv)]
                Hc_chol = np.linalg.cholesky(condensed_Hessian_reg)
            else:
                raise ValueError(f"Unknown regularization method: {self.reg_config.method}, set self.reg_config.method to 'M_PROJECT', 'M_MIRROR' or 'M_CM'")

        y1 = np.linalg.solve(Hc_chol, -gc)
        delta_u_bar = np.linalg.solve(Hc_chol.T, y1)
        delta_u = delta_u_bar.reshape((self.qp.N, self.qp.nu))

        delta_x[0] = np.zeros(self.qp.nx)
        for k in range(self.qp.N):
            C_k = self.qp.C[k]
            D_k = self.qp.D[k]
            rhs_b = rhs.rb[k]
            rhs_d= rhs.rd[k]
            rhs_m = rhs.rm[k]

            # recover x and u
            delta_x[k+1] = self.qp.A[k]@delta_x[k] + self.qp.B[k]@delta_u[k] + rhs_b

            # recover s
            delta_s[k] = -(C_k @delta_x[k] + D_k @delta_u[k]) - rhs_d

            # recover lam
            rhs_m_modified = -rhs_m - (lam_i[k] * delta_s[k])
            delta_lam[k] = (rhs_m_modified/s_i[k]).flatten()

        delta_pi[self.qp.N] = rhs.rg[-1][:self.qp.nx] + self.qp.Q_e @ delta_x[self.qp.N]
        for k in reversed(range(self.qp.N)):
            A_k = self.qp.A[k]
            C_k = self.qp.C[k]
            Q_k = self.qp.Q[k]
            S_k = self.qp.S[k]
            delta_pi[k] = Q_k @ delta_x[k] + S_k.T @ delta_u[k] + A_k.T @ delta_pi[k+1] + C_k.T @ delta_lam[k] + rhs.rg[k][:self.qp.nx]

        delta_dict = {
            'delta_x': delta_x,
            'delta_u': delta_u,
            'delta_pi': delta_pi,
            'delta_lam': delta_lam,
            'delta_s': delta_s,
        }
        return delta_dict

    def backward_sweep(self, rhs, prev_sol:iterate_sol)-> Feedback:
        x0 = prev_sol.x[0]
        nx = self.qp.nx

        modified_qp = self.modify_QP(self.qp, rhs, prev_sol)

        condensed_Hessian = modified_qp.condense_Hessian()

        Gamma = np.zeros((modified_qp.N * modified_qp.nx, modified_qp.N * modified_qp.nu))
        Phi = np.zeros((modified_qp.N * modified_qp.nx, modified_qp.nx))
        upsilon_b = np.zeros(modified_qp.N * modified_qp.nx)

        for k in range(modified_qp.N):
            A_k = modified_qp.A[k]
            b_k = rhs.rb[k]
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
        Qe_fallback = modified_qp.Q[modified_qp.N - 1] if modified_qp.N > 0 else None
        Q_list.append(modified_qp.Q_e if modified_qp.Q_e is not None else Qe_fallback)
        Q_bar = block_diag(*Q_list)

        q_list = [rhs.rg[k][:nx] for k in range(1, modified_qp.N)]
        q_list.append(rhs.rg[-1][:nx])
        q_bar = np.concatenate(q_list)

        r_list = [rhs.rg[k][nx:] for k in range(modified_qp.N)]
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
        if modified_qp.S is not None and len(modified_qp.S) > 0:
            gc[:modified_qp.nu] += modified_qp.S[0] @ x0

        return (condensed_Hessian, gc)

    @staticmethod
    def modify_QP(qp, rhs, prev_sol:iterate_sol):
        modified_QP = copy.deepcopy(qp)

        for k in range(qp.N):
            s_k = prev_sol.s[k]
            lam_k = prev_sol.lam[k]
            C_k = qp.C[k]
            D_k = qp.D[k]

            C_full = np.hstack((C_k, D_k))
            T_mat = np.diag(s_k)
            Lam_mat = np.diag(lam_k)
            rd = rhs.rd[k]
            rm = rhs.rm[k]

            T_mat_inv = np.linalg.inv(T_mat)
            lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
            rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

            Q_tilt = lhs_modification[:qp.nx, :qp.nx]
            R_tilt = lhs_modification[qp.nx:, qp.nx:]
            S_tilt = lhs_modification[qp.nx:, :qp.nx]
            modified_QP.Q[k] = qp.Q[k] + Q_tilt
            modified_QP.R[k] = qp.R[k] + R_tilt
            modified_QP.S[k] = qp.S[k] + S_tilt

            rhs.rg[k][:qp.nx] = rhs.rg[k][:qp.nx] + rhs_modification[:qp.nx]
            rhs.rg[k][qp.nx:] = rhs.rg[k][qp.nx:] + rhs_modification[qp.nx:]
        rhs.rg[-1] = rhs.rg[-1]
        return modified_QP