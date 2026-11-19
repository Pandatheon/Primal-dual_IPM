from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Block_Hessian_reg
from solvers.Riccati_IPM import mirror_lower
from solvers.Riccati_factor import RiccatiFactor, modify_QP, modify_rhs, backward_pass
import numpy as np

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
        self.coeff = 27 if self.reg_config.method != "H_GMW" else 1

        if self.reg_config.method == 'PRECOMPUTE':
            self.pre_regularized_QP = regularize_QP(self.qp, self.reg_config)
            self.stat_dict['decomp_flop'] += self.coeff*(self.qp.N*self.rho + (self.qp.nx[self.qp.N]/self.qp.nu)**3)

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

    def factorize(self, prev_sol:iterate_sol)-> bool:
        horizon = self.qp.N

        coeff = self.coeff
        self.reg = False
        positive_flag = True

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
                modified_qp, CtTinv, Lam = modify_QP(self.qp if positive_flag else self.pre_regularized_QP, prev_sol)

            elif self.reg_config.method in Block_Hessian_reg:
                modified_qp, CtTinv, Lam = modify_QP(self.qp, prev_sol)
                if not positive_flag:
                    modified_qp = regularize_QP(modified_qp, self.reg_config)
                    self.stat_dict['decomp_flop'] += coeff*(horizon*self.rho + (self.qp.nx[horizon]/self.qp.nu)**3)

            elif self.reg_config.method == 'INERTIA_LM':
                modified_qp, CtTinv, Lam = modify_QP(self.qp, prev_sol, LM_delta)

            else:
                modified_qp, CtTinv, Lam = modify_QP(self.qp, prev_sol)
            reduced_delta = LM_delta if self.reg_config.method == 'INERTIA_LM_RE' or self.reg_config.method == 'INERTIA_SE' else 0.0

            fact = RiccatiFactor(
                P = [None]*(horizon+1),
                P_used = [None]*horizon,
                L = [None]*horizon,
                y1 = [None]*horizon,
                K = [None]*horizon,
                CtTinv = CtTinv,
                Lam = Lam,
            )
            fact.P[horizon] = mirror_lower(modified_qp.Q_e + reduced_delta * np.eye(self.qp.nx[horizon]))

            factorization_ok = True
            for k in reversed(range(horizon)):
                nx = self.qp.nx[k]
                A_k = modified_qp.A[k]
                B_k = modified_qp.B[k]
                Q_k = modified_qp.Q[k]
                R_k = modified_qp.R[k]
                S_k = modified_qp.S[k]
                P_prev = fact.P[k+1]

                reduced_hessian = R_k + B_k.T @ P_prev @ B_k + reduced_delta * np.eye(B_k.shape[1])
                try:
                    self.stat_dict['decomp_flop'] += 1
                    R_chol = np.linalg.cholesky(reduced_hessian)
                except np.linalg.LinAlgError:
                    factorization_ok = False
                    break

                g1 = S_k + B_k.T@P_prev@A_k
                y1 = np.linalg.solve(R_chol, g1)
                fact.P_used[k] = P_prev
                fact.L[k] = R_chol
                fact.y1[k] = y1
                fact.K[k] = -np.linalg.solve(R_chol.T, y1)
                fact.P[k] = mirror_lower(Q_k + A_k.T@P_prev@A_k - y1.T@y1 + reduced_delta * np.eye(nx))

            if factorization_ok:
                if self.reg_config.method == 'INERTIA_LM' or self.reg_config.method == 'INERTIA_LM_RE':
                    # IC-4: set delta_w^last <- delta_w used for this successful factorization
                    self.reg_config.delta = LM_delta
                    self.reg = (LM_delta > 0.0)
                self.fact = fact
                return True

            if self.reg_config.method == 'PRECOMPUTE':
                if positive_flag:
                    positive_flag = False
                    continue
                else:
                    print("Hessian projection (PRECOMPUTE) failed to give a positive definite reduced Hessian.")
                    return False

            elif self.reg_config.method in Block_Hessian_reg:
                if positive_flag:
                    positive_flag = False
                    continue
                else:
                    print(f"Regularization method " + self.reg_config.method + " failed to yield a positive definite reduced Hessian.")
                    return False

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
                    return False
                continue

            else:
                print(f"R + B'PB is not positive definite with {np.linalg.eigvals(reduced_hessian)} "
                      f"at stage {k} with regularization method {self.reg_config.method}.")
                return False

    def backward_sweep(self, rhs)-> Feedback:
        self.rhs_kkt = modify_rhs(self.qp, rhs, self.fact)
        return backward_pass(self.qp, self.fact, self.rhs_kkt)
