from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import stage_regularization
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Block_Hessian_reg
import numpy as np
import copy
from utils.modified_Cholesky import modified_cholesky_SE, modified_cholesky_GMW, modified_cholesky_SE99


class Riccati_IPM():
    def __init__(self, qp: BaseQP, reg_config: Regularizations_config, ineq_num:int, stat_dict: dict):
        self.qp = qp
        self.reg_config = reg_config

        self.ineq_num = ineq_num
        self.rhs_kkt = None
        self.stat_dict = stat_dict

        self.reg = False
        self.reg_config.delta = 0.0 if self.reg_config.method in ['INERTIA_LM', 'INERTIA_LM_RE'] else self.reg_config.delta
        self.rho = ((self.qp.nx[1] + self.qp.nu)/self.qp.nu)**3
        self.coeff = 27

        if self.reg_config.method == 'PRECOMPUTE':
            self.pre_regularized_QP = regularize_QP(self.qp, self.reg_config)
            self.stat_dict['decomp_flop'] += self.qp.N*self.coeff*self.rho

    def forward_sweep(self, prev_sol:iterate_sol, feedback:Feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*(self.qp.N+1), [None]*(self.qp.N+1)
        lam_i = prev_sol.lam
        s_i = prev_sol.s

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

        coeff = self.coeff
        self.reg = False
        positive_flag = True   # only used by the 'PRECOMPUTE' branch

        if self.reg_config.method in ['INERTIA_LM', 'INERTIA_LM_RE']:
            # --- Inertia Correction (IC) constants ---
            delta_w_0      = 1e-4    # delta_w^0
            delta_w_min    = 1e-16   # delta_w^min
            delta_w_max    = 1e16    # delta_w^max
            kappa_minus    = 1/3     # kappa_w^-
            kappa_plus     = 8       # kappa_w^+
            kappa_plus_bar = 1e2     # kappa_w^+_bar.
            LM_delta = 0.0           # IC-1: start by attempting the unmodified matrix

        if self.reg_config.method == 'iterative_increasing':
            self.reg_config.delta = 1e-4
            inreasing_factor = 1e2

        while True:
            if self.reg_config.method == 'PRECOMPUTE':
                self.rhs_kkt = copy.deepcopy(rhs)
                modified_qp = self.modify_QP(self.qp, prev_sol) if positive_flag else self.modify_QP(self.pre_regularized_QP, prev_sol)

            elif self.reg_config.method in Block_Hessian_reg:
                self.rhs_kkt = copy.deepcopy(rhs)
                modified_qp = self.modify_QP(self.qp, prev_sol)
                if not positive_flag:
                    modified_qp = regularize_QP(modified_qp, self.reg_config)
                    self.stat_dict['decomp_flop'] += horizon*coeff*self.rho/2

            elif self.reg_config.method == 'INERTIA_LM':
                self.rhs_kkt = copy.deepcopy(rhs)
                modified_qp = self.modify_QP(self.qp, prev_sol, LM_delta)

            elif self.reg_config.method == 'INERTIA_LM_RE':
                self.rhs_kkt = copy.deepcopy(rhs)
                modified_qp = self.modify_QP(self.qp, prev_sol)

            else:
                self.rhs_kkt = copy.deepcopy(rhs)
                modified_qp = self.modify_QP(self.qp, prev_sol)
            reduced_delta = LM_delta if self.reg_config.method == 'INERTIA_LM_RE' or self.reg_config.method == 'INERTIA_SE' else 0.0

            feedback = Feedback(
                P = [None]*(horizon+1),
                p = [None]*(horizon+1),
                K = [None]*(horizon),
                k = [None]*(horizon),
            )
            feedback.P[horizon] = modified_qp.Q_e + reduced_delta * np.eye(self.qp.nx[horizon])
            feedback.p[horizon] = self.rhs_kkt.rg[-1][:self.qp.nx[horizon]]

            factorization_ok = True
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

                reduced_hessian = R_k + B_k.T @ P_prev @ B_k + reduced_delta * np.eye(B_k.shape[1])
                try:
                    self.stat_dict['decomp_flop'] += 1

                    R_chol = np.linalg.cholesky(reduced_hessian)
                    g1 = S_k + B_k.T@P_prev@A_k
                    g2 = r_k + B_k.T@(P_prev@b_k + p_prev)
                    y1 = np.linalg.solve(R_chol, g1)
                    y2 = np.linalg.solve(R_chol, g2)
                    K_curr = -np.linalg.solve(R_chol.T, y1)
                    k_curr = -np.linalg.solve(R_chol.T, y2)
                    P_curr = Q_k + A_k.T@P_prev@A_k - (S_k.T + A_k.T@P_prev@B_k)@(-K_curr) + reduced_delta * np.eye(nx)
                    p_curr = q_k + A_k.T@(P_prev@b_k + p_prev) - (S_k.T + A_k.T@P_prev@B_k)@(-k_curr)

                    feedback.P[k] = P_curr
                    feedback.p[k] = p_curr
                    feedback.K[k] = K_curr
                    feedback.k[k] = k_curr

                except np.linalg.LinAlgError:
                    factorization_ok = False
                    break

            if factorization_ok:
                if self.reg_config.method == 'INERTIA_LM' or self.reg_config.method == 'INERTIA_LM_RE':
                    # IC-4: set delta_w^last <- delta_w used for this successful factorization
                    self.reg_config.delta = LM_delta
                    self.reg = (LM_delta > 0.0)
                break

            if self.reg_config.method == 'PRECOMPUTE':
                if positive_flag:
                    positive_flag = False
                    continue
                else:
                    print("Hessian projection (PRECOMPUTE) failed to give a positive definite reduced Hessian.")
                    return None

            elif self.reg_config.method in Block_Hessian_reg:
                if positive_flag:
                    positive_flag = False
                    continue
                else:
                    print(f"Regularization method " + self.reg_config.method + " failed to yield a positive definite reduced Hessian.")
                    return None

            elif self.reg_config.method in ['INERTIA_LM', 'INERTIA_LM_RE']:
                if LM_delta == 0.0:
                    # IC-3: first regularization of this iteration
                    if self.reg_config.delta == 0.0:
                        LM_delta = delta_w_0
                    else:
                        LM_delta = max(delta_w_min, kappa_minus * self.reg_config.delta)
                else:
                    # IC-5: the current delta_w already failed -> increase it
                    if self.reg_config.delta == 0.0:
                        LM_delta = kappa_plus_bar * LM_delta
                    else:
                        LM_delta = kappa_plus * LM_delta

                # IC-6: give up if delta_w exceeds the cap (would switch to restoration phase)
                if LM_delta > delta_w_max:
                    print(
                        "Reduced Hessian is not positive definite even after maximal "
                        "regularization (delta_w > delta_w_max); restoration phase required.")
                    return None
                continue

            else:
                print(f"R + B'PB is not positive definite with {np.linalg.eigvals(reduced_hessian)} "
                      f"at stage {k} with regularization method {self.reg_config.method}.")
                return None

        return feedback

    def modify_QP(self, qp, prev_sol:iterate_sol, LM_delta=0.0):
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
                modified_QP.Q[k] = qp.Q[k] + Q_tilt + LM_delta * np.eye(qp.nx[k])
                modified_QP.R[k] = qp.R[k] + R_tilt + LM_delta * np.eye(qp.nu)
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
                # primal regularization delta_w*I must also be added to the terminal Hessian
                modified_QP.Q_e = qp.Q_e + Q_tilt + LM_delta * np.eye(qp.nx[k])

                self.rhs_kkt.rg[k][:qp.nx[k]] = self.rhs_kkt.rg[k][:qp.nx[k]] + rhs_modification[:qp.nx[k]]

        return modified_QP