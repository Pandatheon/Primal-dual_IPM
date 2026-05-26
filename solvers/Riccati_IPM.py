from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import stage_regularization
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Hessian_reg, stage_reg
import numpy as np
import copy



class Riccati_IPM():
    def __init__(self, qp: BaseQP, reg_config: Regularizations_config, ineq_num:int):
        self.qp = qp
        self.reg_config = reg_config
        self.ineq_num = ineq_num
        self.rhs_kkt = None

        self.reg = False

    def forward_sweep(self, prev_sol:iterate_sol, feedback:Feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*self.qp.N, [None]*self.qp.N
        lam_i = prev_sol.lam
        s_i = prev_sol.s

        # recover delta_x and delta_u, delta_pi first with modified KKT system
        # then recover delta_s and delta_lam with original KKT system
        delta_x[0] = np.zeros((self.qp.nx,))
        delta_pi[0] = feedback.P[0] @ delta_x[0] + feedback.p[0]

        for k in range(self.qp.N):
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
        nx = self.qp.nx
        self.rhs_kkt = copy.deepcopy(rhs)

        modified_qp = self.modify_QP(self.qp, prev_sol)

        if self.reg_config.method in Hessian_reg:
            modified_qp = regularize_QP(modified_qp, self.reg_config)

        feedback = Feedback(
            P = [None]*(horizon+1),
            p = [None]*(horizon+1),
            K = [None]*(horizon),
            k = [None]*(horizon),
        )
        feedback.P[horizon] = modified_qp.Q_e
        feedback.p[horizon] = self.rhs_kkt.rg[-1][:nx]

        for step in range(self.qp.N):
            k = modified_qp.N - 1 - step
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

                R_chol = np.linalg.cholesky(reduced_hessian)
                g1 = S_k + B_k.T@P_prev@A_k
                g2 = r_k + B_k.T@(P_prev@b_k + p_prev)
                y1 = np.linalg.solve(R_chol, g1)
                y2 = np.linalg.solve(R_chol, g2)
                K_curr = -np.linalg.solve(R_chol.T, y1)
                k_curr = -np.linalg.solve(R_chol.T, y2)
                P_curr = Q_k + A_k.T@P_prev@A_k - (S_k.T + A_k.T@P_prev@B_k)@(-K_curr)
                p_curr = q_k + A_k.T@(P_prev@b_k + p_prev) - (S_k.T + A_k.T@P_prev@B_k)@(-k_curr)

                feedback.P[k] = P_curr
                feedback.p[k] = p_curr
                feedback.K[k] = K_curr
                feedback.k[k] = k_curr

            except np.linalg.LinAlgError:
                if self.reg_config.method == 'NO_REGULARIZE':
                    raise np.linalg.LinAlgError("R + B'PB is not positive definite. Consider using regularization.")

                elif self.reg_config.method in stage_reg:

                    try:
                        reduced_hessian_reg, P_reg = stage_regularization(modified_qp, feedback, reduced_hessian, self.reg_config, k)
                        R_chol = np.linalg.cholesky(reduced_hessian_reg)
                        g1 = S_k + B_k.T@P_reg@A_k
                        g2 = r_k + B_k.T@(P_reg@b_k + p_prev)
                        y1 = np.linalg.solve(R_chol, g1)
                        y2 = np.linalg.solve(R_chol, g2)
                        K_curr = -np.linalg.solve(R_chol.T, y1)
                        k_curr = -np.linalg.solve(R_chol.T, y2)
                        P_curr = Q_k + A_k.T@P_reg@A_k - (S_k.T + A_k.T@P_reg@B_k)@(-K_curr)
                        p_curr = q_k + A_k.T@(P_reg@b_k + p_prev) - (S_k.T + A_k.T@P_reg@B_k)@(-k_curr)

                        feedback.P[k] = P_curr
                        feedback.p[k] = p_curr
                        feedback.K[k] = K_curr
                        feedback.k[k] = k_curr
                    except np.linalg.LinAlgError:
                        raise np.linalg.LinAlgError("Regularization failed to make the reduced Hessian positive definite.")

                else:
                    raise np.linalg.LinAlgError("Unkonwn error happened")

        # # print P for debug
        # for k in reversed(range(self.qp.N + 1)):
        #     print(f"Stage {k}:")
        #     print(f"P[{k}] = \n{feedback.P[k]}")

        return feedback

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