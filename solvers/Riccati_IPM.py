from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import stage_regularization
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Block_Hessian_reg
import numpy as np
import copy


def potrf_blasfeo(M: np.ndarray):
    '''
    Cholesky factorization as in BLASFEO (the one used by HPIPM): a non-positive pivot does not abort the
    factorization, its column of L is set to zero instead. Solving with such a factor leaves the corresponding
    variable at zero, i.e. the system is solved on the rows and columns of the accepted pivots only.
    Returns L and the mask of the accepted pivots.
    '''
    n = M.shape[0]
    L = np.zeros((n, n))
    keep = np.zeros(n, dtype=bool)
    for j in range(n):
        pivot = M[j, j] - L[j, :j] @ L[j, :j]
        if pivot > 0:
            keep[j] = True
            L[j:, j] = (M[j:, j] - L[j:, :j] @ L[j, :j]) / np.sqrt(pivot)
    return L, keep

def mirror_lower(M: np.ndarray) -> np.ndarray:
    '''Symmetric matrix from the lower triangle of M, 
    as HPIPM does for the Riccati matrix P (BLASFEO dtrtr_l).'''
    return np.tril(M) + np.tril(M, -1).T

def solve_blasfeo(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    '''Solve M sol = rhs with the factorization of potrf_blasfeo, the variables of the rejected pivots stay at zero.'''
    L, keep = potrf_blasfeo(M)
    L_keep = L[np.ix_(keep, keep)]
    sol = np.zeros_like(rhs, dtype=float)
    sol[keep] = np.linalg.solve(L_keep.T, np.linalg.solve(L_keep, rhs[keep]))
    return sol

class Riccati_IPM():
    def __init__(self, qp: BaseQP, reg_config: Regularizations_config, ineq_num:int, stat_dict: dict):
        self.qp = qp
        self.reg_config = reg_config
        self.ineq_num = ineq_num
        self.rhs_kkt = None
        self.stat_dict = stat_dict

        self.reg = False
        self.rho = (self.qp.nx[1]/self.qp.nu)**3

    def forward_sweep(self, prev_sol:iterate_sol, feedback:Feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*(self.qp.N+1), [None]*(self.qp.N+1)
        lam_i = prev_sol.lam
        s_i = prev_sol.s
        #     delta_x[0] = -solve_blasfeo(feedback.P[0], feedback.p[0])
        #     delta_pi[0] = np.zeros(self.qp.nx[0])

        # recover delta_x and delta_u, delta_pi first with modified KKT system
        # then recover delta_s and delta_lam with original KKT system
        delta_x[0] = -np.linalg.solve(feedback.P[0], feedback.p[0])
        delta_pi[0] = feedback.P[0] @ delta_x[0] + feedback.p[0]

        for k in range(self.qp.N+1):
            if k < self.qp.N:
                C_k = self.qp.C[k]
                D_k = self.qp.D[k]
                rhs_b = self.rhs_kkt.rb[k]
                rhs_d= self.rhs_kkt.rd[k]
                rhs_m = self.rhs_kkt.rm[k]

                # recover x and u
                delta_u[k] = feedback.compute_control(k, delta_x[k])
                delta_x[k+1] = self.qp.A[k]@delta_x[k] + self.qp.B[k]@delta_u[k] + rhs_b

                # recover s
                delta_s[k] = -(C_k @delta_x[k] + D_k @delta_u[k]) - rhs_d

                # recover lam
                rhs_m_modified = -rhs_m - (lam_i[k] * delta_s[k])
                delta_lam[k] = (rhs_m_modified/s_i[k]).flatten()

                # recover pi
                delta_pi[k+1] =  feedback.P[k+1] @ delta_x[k+1] + feedback.p[k+1]
            else:
                C_k = self.qp.C[k]
                D_k = self.qp.D[k]
                rhs_d= self.rhs_kkt.rd[k]
                rhs_m = self.rhs_kkt.rm[k]

                # recover s
                delta_s[k] = -(C_k @delta_x[k]) - rhs_d

                # recover lam
                rhs_m_modified = -rhs_m - (lam_i[k] * delta_s[k])
                delta_lam[k] = (rhs_m_modified/s_i[k]).flatten()

        delta_dict = {
            'delta_x': delta_x,
            'delta_u': delta_u,
            'delta_pi': delta_pi,
            'delta_lam': delta_lam,
            'delta_s': delta_s,
        }
        return delta_dict

    def backward_sweep(self, rhs, prev_sol:iterate_sol)-> Feedback:
        horizon = self.qp.N

        self.rhs_kkt = copy.deepcopy(rhs)
        self.reg = False

        modified_qp = self.modify_QP(self.qp, prev_sol)

        feedback = Feedback(
            P = [None]*(horizon+1),
            p = [None]*(horizon+1),
            K = [None]*(horizon),
            k = [None]*(horizon),
        )
        feedback.P[horizon] = modified_qp.Q_e
        feedback.p[horizon] = self.rhs_kkt.rg[-1][:self.qp.nx[horizon]]

        for step in range(self.qp.N):
            k = modified_qp.N - 1 - step
            nx = self.qp.nx[k]
            A_k = modified_qp.A[k]
            B_k = modified_qp.B[k]
            Q_k = modified_qp.Q[k]
            R_k = modified_qp.R[k]
            S_k = modified_qp.S[k]
            q_k = self.rhs_kkt.rg[k][:nx]
            r_k = self.rhs_kkt.rg[k][nx:]
            b_k = self.rhs_kkt.rb[k]
            P_prev = feedback.P[k+1]
            p_prev = feedback.p[k+1]

            reduced_hessian = R_k + B_k.T @ P_prev @ B_k
            try:
                self.stat_dict['decomp_flop'] += 1
                R_chol = np.linalg.cholesky(reduced_hessian)
                g1 = S_k + B_k.T@P_prev@A_k
                g2 = r_k + B_k.T@(P_prev@b_k + p_prev)
                y1 = np.linalg.solve(R_chol, g1)
                y2 = np.linalg.solve(R_chol, g2)
                K_curr = -np.linalg.solve(R_chol.T, y1)
                k_curr = -np.linalg.solve(R_chol.T, y2)
                P_curr = mirror_lower(Q_k + A_k.T@P_prev@A_k - y1.T@y1)
                p_curr = q_k + A_k.T@(P_prev@b_k + p_prev) - y1.T@y2

                feedback.P[k] = P_curr
                feedback.p[k] = p_curr
                feedback.K[k] = K_curr
                feedback.k[k] = k_curr

                # self.stat_dict['decomp_flop'] += 1
                # L, keep = potrf_blasfeo(reduced_hessian)
                # R_chol = L[np.ix_(keep, keep)]
                # g1 = S_k + B_k.T@P_prev@A_k
                # g2 = r_k + B_k.T@(P_prev@b_k + p_prev)
                # y1 = np.linalg.solve(R_chol, g1[keep])
                # y2 = np.linalg.solve(R_chol, g2[keep])
                # K_curr = np.zeros_like(g1)
                # k_curr = np.zeros_like(g2)
                # K_curr[keep] = -np.linalg.solve(R_chol.T, y1)
                # k_curr[keep] = -np.linalg.solve(R_chol.T, y2)
                # P_curr = mirror_lower(Q_k + A_k.T@P_prev@A_k - y1.T@y1)
                # p_curr = q_k + A_k.T@(P_prev@b_k + p_prev) - y1.T@y2

                # feedback.P[k] = P_curr
                # feedback.p[k] = p_curr
                # feedback.K[k] = K_curr
                # feedback.k[k] = k_curr

            except np.linalg.LinAlgError:
                if self.reg_config.method == 'NO_REGULARIZE':
                    eigs = np.linalg.eigvals(reduced_hessian)
                    min_eig = np.min(eigs)
                    print(f"R + B'PB is not positive definite with {min_eig} at stage {k} without regularization.")
                    return None

                elif self.reg_config.method not in Block_Hessian_reg:

                    try:
                        reduced_hessian_reg, P_reg = stage_regularization(modified_qp, feedback, reduced_hessian, self.reg_config, k)
                        coeff = 27 if 'GMW' not in self.reg_config.method else 1
                        self.rho = (self.qp.nx[1]/self.qp.nu)**3 if "PM" in self.reg_config.method else 0
                        self.stat_dict['decomp_flop'] += coeff*(1+self.rho) if "GMW" in self.reg_config.method else coeff*(1+self.rho) + 1

                        R_chol = np.linalg.cholesky(reduced_hessian_reg)
                        g1 = S_k + B_k.T@P_reg@A_k
                        g2 = r_k + B_k.T@(P_reg@b_k + p_prev)
                        y1 = np.linalg.solve(R_chol, g1)
                        y2 = np.linalg.solve(R_chol, g2)
                        K_curr = -np.linalg.solve(R_chol.T, y1)
                        k_curr = -np.linalg.solve(R_chol.T, y2)
                        P_curr = mirror_lower(Q_k + A_k.T@P_reg@A_k - y1.T@y1)
                        p_curr = q_k + A_k.T@(P_reg@b_k + p_prev) - y1.T@y2

                        feedback.P[k] = P_curr
                        feedback.p[k] = p_curr
                        feedback.K[k] = K_curr
                        feedback.k[k] = k_curr

                    except np.linalg.LinAlgError:
                        # ad-hoc handling
                        eigs = np.linalg.eigvals(reduced_hessian_reg)
                        min_eig = np.min(eigs)
                        print(f"Regularization failed with {min_eig} during method " + self.reg_config.method)
                        return None
                else:
                    # ad-hoc handling
                    eigs = np.linalg.eigvals(reduced_hessian)
                    min_eig = np.min(eigs)
                    print(f"Regularization method " + self.reg_config.method + f" failed with {min_eig} at stage {k}.")
                    return None

        # # print P for debug
        # for k in reversed(range(self.qp.N + 1)):
        #     print(f"Stage {k}:")
        #     print(f"P[{k}] = \n{feedback.P[k]}")

        return feedback

    def modify_QP(self, qp, prev_sol:iterate_sol):
        modified_QP = copy.deepcopy(qp)

        for k in range(qp.N+1):
            if k < qp.N:
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

                Q_tilt = lhs_modification[:qp.nx[k], :qp.nx[k]]
                R_tilt = lhs_modification[qp.nx[k]:, qp.nx[k]:]
                S_tilt = lhs_modification[qp.nx[k]:, :qp.nx[k]]
                modified_QP.Q[k] = qp.Q[k] + Q_tilt
                modified_QP.R[k] = qp.R[k] + R_tilt
                modified_QP.S[k] = qp.S[k] + S_tilt

                self.rhs_kkt.rg[k][:qp.nx[k]] = self.rhs_kkt.rg[k][:qp.nx[k]] + rhs_modification[:qp.nx[k]]
                self.rhs_kkt.rg[k][qp.nx[k]:] = self.rhs_kkt.rg[k][qp.nx[k]:] + rhs_modification[qp.nx[k]:]
            else:
                s_k = prev_sol.s[k]
                lam_k = prev_sol.lam[k]
                C_k = qp.C[k]
                D_k = qp.D[k]

                C_full = C_k
                T_mat = np.diag(s_k)
                Lam_mat = np.diag(lam_k)
                rd = self.rhs_kkt.rd[k]
                rm = self.rhs_kkt.rm[k]

                T_mat_inv = np.linalg.inv(T_mat)
                lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
                rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

                Q_tilt = lhs_modification[:qp.nx[k], :qp.nx[k]]
                modified_QP.Q_e = qp.Q_e + Q_tilt

                self.rhs_kkt.rg[k][:qp.nx[k]] = self.rhs_kkt.rg[k][:qp.nx[k]] + rhs_modification[:qp.nx[k]]

        return modified_QP