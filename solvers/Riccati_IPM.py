from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs, Feedback
from solvers.Regularization import stage_regularization
from solvers.Regularization import regularize_QP
from solvers.Regularization_config import Regularizations_config, Reduced_Hessian_based_regularization
import numpy as np
import copy



class Riccati_IPM():
    def __init__(self, qp: BaseQP, reg_config: Regularizations_config):
        self.qp = qp
        self.ineq_num = self.qp.nh * self.qp.N + 2*self.qp.nx
        self.reg_config = reg_config

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
            if k == 0:
                temp_res_stat[nx:] = R_k @ u[0] + r_k + B_k.T @ pi[0] + D_k.T @ lam[0]
            else:
                temp_res_stat[:nx] = Q_k @ x[k] + S_k.T @ u[k] + q_k + A_k.T @ pi[k] + C_k.T @ lam[k] - pi[k-1]
                temp_res_stat[nx:] = R_k @ u[k] + S_k @ x[k] + r_k + B_k.T @ pi[k] + D_k.T @ lam[k]
            res_stat.append(temp_res_stat)

            #equality residual
            res_eq.append(A_k @ x[k] + B_k @ u[k] - x[k+1] - b_k)

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
        temp_res_stat[:nx] = Q_e @ x[self.qp.N] + q_e - pi[self.qp.N-1]
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

    def foward_sweep(self, rhs, prev_sol:iterate_sol, feedback:Feedback)-> dict:
        delta_x, delta_u, delta_pi, delta_lam, delta_s = [None]*(self.qp.N+1), [None]*self.qp.N, [None]*(self.qp.N+1), [None]*self.qp.N, [None]*self.qp.N
        lam_i = prev_sol.lam
        s_i = prev_sol.s

        # recover delta_x and delta_u, delta_pi first with modified KKT system
        # then recover delta_s and delta_lam with original KKT system
        delta_x[0] = np.zeros((self.qp.nx,))
        for k in range(self.qp.N):
            C_k = self.qp.C[k]
            D_k = self.qp.D[k]
            rhs_b = rhs.rb[k]
            rhs_d= rhs.rd[k]
            rhs_m = rhs.rm[k]

            # recover x and u
            delta_u[k] = feedback.compute_control(k, delta_x[k])
            delta_x[k+1] = self.qp.A[k]@delta_x[k] + self.qp.B[k]@delta_u[k] + rhs_b

            # recover s
            delta_s[k] = -(C_k @delta_x[k] + D_k @delta_u[k]) - rhs_d

            # recover lam

            rhs_m_modified = -rhs_m - (lam_i[k] * delta_s[k])
            delta_lam[k] = (rhs_m_modified/s_i[k]).flatten()

            # recover pi
            delta_pi[k] =  feedback.P[k+1] @ delta_x[k+1] + feedback.p[k+1]

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

        if self.reg_config.method == 'PROJECT' or self.reg_config.method == 'MIRROR':
            regularized_QP = regularize_QP(self.qp, self.reg_config)
        else:
            regularized_QP = copy.deepcopy(self.qp)
        modified_qp = self.modify_QP(regularized_QP, rhs, prev_sol)

        feedback = Feedback(
            P = [None]*(horizon+1),
            p = [None]*(horizon+1),
            K = [None]*(horizon),
            k = [None]*(horizon),
        )
        feedback.P[horizon] = modified_qp.Q_e
        feedback.p[horizon] = rhs.rg[-1][:nx]

        for step in range(self.qp.N):
            k = modified_qp.N - 1 - step
            A_k = modified_qp.A[k]
            B_k = modified_qp.B[k]
            Q_k = modified_qp.Q[k]
            R_k = modified_qp.R[k]
            S_k = modified_qp.S[k]
            q_k = rhs.rg[k][:nx]
            r_k = rhs.rg[k][nx:]
            b_k = rhs.rb[k]
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

                elif self.reg_config.method in Reduced_Hessian_based_regularization:

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

        return feedback

    @staticmethod
    def modify_QP(qp, rhs, prev_sol:iterate_sol):
        modified_QP = copy.deepcopy(qp)

        for k in range(qp.N):
            s_k = prev_sol.s[k]
            lam_k = prev_sol.lam[k]
            C_k = qp.C[k]
            D_k = qp.D[k]

            C_full = np.concatenate((C_k, D_k), axis=1)
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