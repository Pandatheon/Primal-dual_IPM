from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import stage_regularization
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Block_Hessian_reg
import numpy as np
import copy


def chol_upper(M, jitter=0.0, max_tries=6):
    M = 0.5 * (M + M.T)
    n = M.shape[0]
    j = jitter
    for _ in range(max_tries):
        try:
            Mj = M if j == 0.0 else M + j * np.eye(n)
            L = np.linalg.cholesky(Mj)  # lower, Mj = L L^T
            return L.T, j               # upper, U = L^T, U^T U = Mj
        except np.linalg.LinAlgError:
            scale = np.trace(M) / max(n, 1)
            j = 1e-12 * max(scale, 1.0) if j == 0.0 else j * 10.0
    raise np.linalg.LinAlgError("chol_upper failed even after jitter")

def sqrt_riccati_stage(A, B, Q, R, S, SP_next, b, q, r, p_next, base_jitter=0.0):
    nu = R.shape[0]
    nx_k = Q.shape[0]

    # stage-cost Hessian, columns ordered [u, x].
    # S is nu x nx (matching g1 = S + B^T P A), so top-right = S, bottom-left = S^T
    base = np.block([[R, S], [S.T, Q]])
    Cb, _ = chol_upper(base, base_jitter)  # base = Cb^T Cb

    # propagate the factor, never forming P = SP_next^T SP_next
    Wb = SP_next @ B          # nx_next x nu
    Wa = SP_next @ A          # nx_next x nx_k
    W = np.hstack((Wb, Wa))   # nx_next x (nu + nx_k)

    pre = np.vstack((Cb, W))  # (nu + nx_k + nx_next) x (nu + nx_k)

    # QR:  pre = Qo @ T,  T upper,  T^T T = base + W^T W = G
    _, T = np.linalg.qr(pre)

    T11 = T[:nu, :nu]
    T12 = T[:nu, nu:]
    T22 = T[nu:, nu:]         # = SP_k, since P_k = T22^T T22

    SP_k = T22

    # gain  K = -H^{-1} g1 = -T11^{-1} T12
    K = -np.linalg.solve(T11, T12)

    # affine part
    w = SP_next.T @ (SP_next @ b) + p_next   # = P_next b + p_next
    g2 = r + B.T @ w
    zeta = np.linalg.solve(T11.T, g2)        # T11^T zeta = g2
    k_ff = -np.linalg.solve(T11, zeta)       # = -H^{-1} g2
    p_k = q + A.T @ w - T12.T @ zeta

    return SP_k, K, k_ff, p_k

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

    def backward_sweep(self, rhs, prev_sol):
        horizon = self.qp.N

        self.rhs_kkt = copy.deepcopy(rhs)
        self.reg = False

        modified_qp = self.modify_QP(self.qp, prev_sol)

        feedback = Feedback(
            P=[None] * (horizon + 1),
            p=[None] * (horizon + 1),
            K=[None] * horizon,
            k=[None] * horizon,
        )

        nx_N = self.qp.nx[horizon]
        SP = [None] * (horizon + 1)
        p_list = [None] * (horizon + 1)

        SP[horizon], jit = chol_upper(modified_qp.Q_e)
        p_list[horizon] = self.rhs_kkt.rg[-1][:nx_N]

        feedback.P[horizon] = SP[horizon].T @ SP[horizon]
        feedback.p[horizon] = p_list[horizon]

        for step in range(self.qp.N):
            k = self.qp.N - 1 - step
            nx_k = self.qp.nx[k]

            A = modified_qp.A[k]
            B = modified_qp.B[k]
            Q = modified_qp.Q[k]
            R = modified_qp.R[k]
            S = modified_qp.S[k]
            q = self.rhs_kkt.rg[k][:nx_k]
            r = self.rhs_kkt.rg[k][nx_k:]
            b = self.rhs_kkt.rb[k]

            try:
                self.stat_dict['decomp_flop'] += 1
                SP_k, K, k_ff, p_k = sqrt_riccati_stage(
                    A, B, Q, R, S, SP[k + 1], b, q, r, p_list[k + 1],
                    base_jitter=0.0,
                )
            except np.linalg.LinAlgError:
                if self.reg_config.method == 'NO_REGULARIZE':
                    print(f"base cost not PD at stage {k}, no regularization allowed.")
                    return None
                self.reg = True
                SP_k, K, k_ff, p_k = sqrt_riccati_stage(
                    A, B, Q, R, S, SP[k + 1], b, q, r, p_list[k + 1],
                    base_jitter=1e-8,
                )
                self.stat_dict['decomp_flop'] += 1

            SP[k] = SP_k
            p_list[k] = p_k

            feedback.P[k] = SP_k.T @ SP_k
            feedback.p[k] = p_k
            feedback.K[k] = K
            feedback.k[k] = k_ff

        self._SP = SP
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
                rd = self.rhs_kkt.rd[k]
                rm = self.rhs_kkt.rm[k]

                sigma = lam_k / s_k
                lhs_modification = C_full.T @ (sigma[:, None] * C_full)
                rhs_modification = C_full.T @ (sigma * rd - rm / s_k)

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
                rd = self.rhs_kkt.rd[k]
                rm = self.rhs_kkt.rm[k]

                sigma = lam_k / s_k
                lhs_modification = C_full.T @ (sigma[:, None] * C_full)
                rhs_modification = C_full.T @ (sigma * rd - rm / s_k)

                Q_tilt = lhs_modification[:qp.nx[k], :qp.nx[k]]
                modified_QP.Q_e = qp.Q_e + Q_tilt

                self.rhs_kkt.rg[k][:qp.nx[k]] = self.rhs_kkt.rg[k][:qp.nx[k]] + rhs_modification[:qp.nx[k]]

        return modified_QP