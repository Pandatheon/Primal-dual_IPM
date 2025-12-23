from typing import Tuple
from dataclasses import dataclass

from QP_formulation import OCP_QP, QP_stage, QP_Terminal, iterate
import numpy as np
import copy

@dataclass
class Feedback:
    """QP terminal cost parameters arranged as (P, p)"""
    P: np.ndarray
    p: np.ndarray
    K: np.ndarray
    k: np.ndarray

    def compute_control(self, index: int, state: np.ndarray) -> np.ndarray:
        '''compute the control according to the feedback law u = k + Kx'''
        return self.K[index] @ state + self.k[index]

class QP_solver:

    def __init__(self, qp:OCP_QP):
        self.qp = qp
        self.feedback = None
        self.tau = 2.0
        self.rhs_dict = {
            'rg' : np.zeros((qp.N+1, qp.nx+qp.nu)),
            'rb' : np.zeros((qp.N, qp.nx)),
            'rd0' : np.zeros((qp.nh0, )),
            'rd' : np.zeros((qp.N-1, qp.nh)),
            'rm0' : np.zeros((qp.nh0, )),
            'rm' : np.zeros((qp.N-1, qp.nh)),
        }
        self.statistics_dict = {
            'res_stat_list' : [],
            'res_eq_list' : [],
            'res_ineq_list' : [],
            'res_comp_list' : [],
            'alpha_pri_list' : [],
            'alpha_dual_list' : [],
            'alpha_list' : [],
            'barrier_list' : [],
            'iter_count' : 0,
        }
        self.solution = iterate(
            x = np.zeros((qp.N + 1, qp.nx)),
            u =  np.zeros((qp.N, qp.nu)),
            pi = np.zeros((qp.N, self.qp.nx)),
            lam0= np.zeros((qp.nh0, )),
            lam = np.zeros((qp.N-1, self.qp.nh)),
            s0 = np.zeros((self.qp.nh0, )),
            s = np.zeros((qp.N-1, self.qp.nh)),
        )

    def _initilize_solution(self, x0:np.ndarray)-> iterate:
        self.solution.x = np.vstack([x0 for i in range(self.qp.N+1)])
        self.solution.u = np.zeros((self.qp.N, self.qp.nu))
        init_s = self.qp.stages[1].d - self.qp.stages[1].C @ self.solution.x[1] - self.qp.stages[1].D @ self.solution.u[1]
        self.solution.s = np.vstack([init_s for _ in range(self.qp.N-1)])
        self.solution.lam = self.tau / self.solution.s
        self.solution.pi = np.zeros((self.qp.N, self.qp.nx))

        init_d0 =  np.hstack(( -(x0-1e-1)*np.ones((self.qp.nx,)), (x0+1e-1)*np.ones((self.qp.nx,)), self.qp.stages[0].d))
        init_s0 = init_d0 -  self.qp.C0 @ self.solution.x[0] - self.qp.D0 @ self.solution.u[0]
        d0 = np.hstack(( -(x0)*np.ones((self.qp.nx,)), (x0)*np.ones((self.qp.nx,)), self.qp.stages[0].d))
        self.qp.stages[0] = QP_stage(
            Q = self.qp.Q,
            R = self.qp.R,
            S = self.qp.S,
            q = self.qp.q,
            r = self.qp.r,
            A = self.qp.A,
            B = self.qp.B,
            b = self.qp.b,
            C = self.qp.C0,
            D = self.qp.D0,
            d = d0,
        )
        self.solution.s0 = init_s0
        self.solution.lam0 = self.tau / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=50, reg:str='NO_REGULARIZE', tol:float=1e-6)-> iterate:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        assert x0 is not None, "Please provide a non-zero initial state x0 to solve the QP."
        self._initilize_solution(x0)
        for i in range(max_iter):
            prev_sol = copy.deepcopy(self.solution)
            res = self._update_rhs(prev_sol)
            if res < tol:
                print(f"\033[32mConverged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['barrier_list'][-1]:.2e}")
                return self.solution
            self._backward_sweep(prev_sol, reg=reg)
            self._forward_sweep(prev_sol)
            self.tau = max(self.tau*0.1, 1e-8)
            self.statistics_dict['iter_count'] += 1
        print(f"\033[31mNot Converged!\033[0m after {max_iter} iterations with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['barrier_list'][-1]:.2e}")
        return self.solution

    def _update_rhs(self, prev_sol:iterate)-> np.ndarray:
        nx = self.qp.nx
        nu = self.qp.nu
        N = self.qp.N
        stage0 = self.qp.stages[0]

        res_stat = np.zeros((N+1, nx+nu))
        res_eq = np.zeros((N, nx))
        res_ineq0 = np.zeros((1, stage0.C.shape[0]))
        res_ineq = np.zeros((N-1, self.qp.stages[1].C.shape[0]))
        res_comp0 = np.zeros((1, stage0.C.shape[0]))
        res_comp = np.zeros((N-1, self.qp.stages[1].C.shape[0]))
        Q = self.qp.stages[0].Q
        R = self.qp.stages[0].R
        S = self.qp.stages[0].S
        q = self.qp.stages[0].q
        r = self.qp.stages[0].r
        A = self.qp.stages[0].A
        B = self.qp.stages[0].B
        b = self.qp.stages[0].b
        C = self.qp.stages[1].C
        D = self.qp.stages[1].D
        d = self.qp.stages[1].d
        Q_e = self.qp.terminal.Q_e
        q_e = self.qp.terminal.q_e
        C0 = self.qp.stages[0].C
        D0 = self.qp.stages[0].D

        for i in range(len(self.qp.stages)):
            # stationarity residual
            if i == 0:
                res_stat[i][nx:] = R @ prev_sol.u[0] + S @ prev_sol.x[0] + r + B.T @ prev_sol.pi[0] + D0.T @ prev_sol.lam0
            else:
                res_stat[i][:nx] = Q @ prev_sol.x[i] + S.T @ prev_sol.u[i] + q +  A.T @ prev_sol.pi[i] + C.T @ prev_sol.lam[i-1] - prev_sol.pi[i-1]
                res_stat[i][nx:] = R @ prev_sol.u[i] + S @ prev_sol.x[i] + r + B.T @ prev_sol.pi[i] + D.T @ prev_sol.lam[i-1]

            #equality residual
            res_eq[i] = A @ prev_sol.x[i] + B @ prev_sol.u[i] - prev_sol.x[i+1] - b

            #inequality residual
            if i == 0:
                res_ineq0 =  C0 @ prev_sol.x[0] + D0 @ prev_sol.u[0] + prev_sol.s0 - stage0.d
            else:
                res_ineq[i-1] = (C @ prev_sol.x[i] + D @ prev_sol.u[i] + prev_sol.s[i-1]) - d

            #complementarity residual
            if i == 0:
                res_comp0 = prev_sol.lam0 * prev_sol.s0 - self.tau * np.ones((len(prev_sol.lam0), ))
            else:
                res_comp[i-1] = (np.diag(prev_sol.lam[i-1]) @ prev_sol.s[i-1] - self.tau * np.ones((len(prev_sol.lam[i-1]), ))).flatten()

        # terminal residual
            res_stat[N][:nx] = Q_e @ prev_sol.x[N] + q_e - prev_sol.pi[N-1]

        self.rhs_dict['rg'] = res_stat
        self.rhs_dict['rb'] = res_eq
        self.rhs_dict['rd0'] = res_ineq0
        self.rhs_dict['rd'] = res_ineq
        self.rhs_dict['rm0'] = res_comp0
        self.rhs_dict['rm'] = res_comp

        self.statistics_dict['res_stat_list'].append(np.linalg.norm(res_stat.flatten()))
        self.statistics_dict['res_eq_list'].append(np.linalg.norm(res_eq.flatten()))
        self.statistics_dict['res_ineq_list'].append(np.linalg.norm(res_ineq.flatten()))
        self.statistics_dict['res_comp_list'].append(np.linalg.norm(res_comp.flatten()))
        self.statistics_dict['barrier_list'].append(self.tau)
        res = np.linalg.norm(np.hstack((res_stat.flatten(), res_eq.flatten(), res_ineq0.flatten(), res_ineq.flatten(), res_comp0.flatten(), res_comp.flatten())))
        return res

    def _forward_sweep(self, prev_sol:iterate)-> 'iterate':
        # Solve for the next iterate
        delta_dict = self._solve_KKT_system(prev_sol)
        # Compute the step size using fraction to the boundary rule
        alpha_pri, alpha_dual, alpha = self._fraction_to_boundary(prev_sol, delta_dict)
        # Update the solution with the step size alpha
        x_aff = prev_sol.x + alpha_pri * delta_dict['delta_x']
        u_aff = prev_sol.u + alpha_pri * delta_dict['delta_u']
        pi_aff = prev_sol.pi + alpha_dual * delta_dict['delta_pi']
        lam0_aff = prev_sol.lam0 + alpha_dual * delta_dict['delta_lam0']
        lam_aff = prev_sol.lam + alpha_dual * delta_dict['delta_lam']
        s0_aff = prev_sol.s0 + alpha_pri * delta_dict['delta_s0']
        s_aff = prev_sol.s + alpha_pri * delta_dict['delta_s']

        self.solution = iterate(x_aff, u_aff, pi_aff, lam0_aff, lam_aff, s0_aff, s_aff)

    def _solve_KKT_system(self, prev_sol:iterate)-> iterate:
        delta_x = copy.deepcopy(prev_sol.x)
        delta_u = copy.deepcopy(prev_sol.u)
        delta_pi = copy.deepcopy(prev_sol.pi)
        delta_lam0 = copy.deepcopy(prev_sol.lam0)
        delta_lam = copy.deepcopy(prev_sol.lam)
        delta_s0 = copy.deepcopy(prev_sol.s0)
        delta_s = copy.deepcopy(prev_sol.s)
        Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e = self.qp()
        C0 = self.qp.stages[0].C
        D0 = self.qp.stages[0].D
        d0 = self.qp.stages[0].d

        # recover delta_x and delta_u, delta_pi first with modified KKT system
        # then recover delta_s and delta_lam with original KKT system
        delta_x[0] = 0
        for i in range(len(self.qp.stages)):
            # recover x and u
            rhs_b = self.rhs_dict['rb'][i]
            delta_u[i] = self.feedback.compute_control(i, delta_x[i])
            delta_x[i+1] = A@delta_x[i] + B@delta_u[i] + rhs_b

            # recover s
            if i == 0:
                rhs_d =  C0 @ prev_sol.x[i] + D0 @ prev_sol.u[i] + prev_sol.s0 - d0
                delta_s0 = -(C0 @delta_x[0] + D0 @delta_u[0]) - rhs_d
            else:
                rhs_d = C @ prev_sol.x[i] + D @ prev_sol.u[i] + prev_sol.s[i-1] - d
                delta_s[i-1] = -(C @delta_x[i] + D @delta_u[i]) - rhs_d
            # recover lam
            if i == 0:
                rhs_m = (prev_sol.lam0 * prev_sol.s0 - self.tau * np.ones(len(prev_sol.lam0)))
                rhs_m_modified = -rhs_m - (prev_sol.lam0 * delta_s0) 
                delta_lam0 = (np.linalg.solve(np.diag(prev_sol.s0), rhs_m_modified)).flatten()
            else:
                rhs_m = (prev_sol.lam[i-1] * prev_sol.s[i-1] - self.tau * np.ones(len(prev_sol.lam[i-1])))
                rhs_lam = -rhs_m - (prev_sol.lam[i-1] * delta_s[i-1])
                delta_lam[i-1] = (np.linalg.solve(np.diag(prev_sol.s[i-1]), rhs_lam)).flatten()
            # recover pi
            delta_pi[i] =  self.feedback.P[i+1] @ delta_x[i+1] + self.feedback.p[i+1]

        delta_dict = {
            'delta_x': delta_x,
            'delta_u': delta_u,
            'delta_pi': delta_pi,
            'delta_lam0': delta_lam0,
            'delta_lam': delta_lam,
            'delta_s0': delta_s0,
            'delta_s': delta_s,
        }

        return delta_dict

    def _fraction_to_boundary(self, prev_sol: iterate, delta_dict) -> iterate:
        s0_old = prev_sol.s0
        s_old = prev_sol.s
        lam0_old = prev_sol.lam0
        lam_old = prev_sol.lam
        delta_s0 = delta_dict['delta_s0']
        delta_s = delta_dict['delta_s']
        delta_lam0 = delta_dict['delta_lam0']
        delta_lam = delta_dict['delta_lam']

        # do fraction to the boundary
        frac_pri0 = - s0_old/delta_s0
        frac_dual0 = - lam0_old/delta_lam0
        frac_pri = - s_old.flatten()/delta_s.flatten()
        frac_dual = - lam_old.flatten()/delta_lam.flatten()
        frac_pri = np.hstack((frac_pri0, frac_pri))
        frac_dual = np.hstack((frac_dual0, frac_dual))
        # take the minimum positive step size
        alpha_pri_max = np.min([1, np.min(frac_pri[frac_pri > 0])])
        alpha_dual_max = np.min([1, np.min(frac_dual[frac_dual > 0])])
        # take 0.99 of the maximum step size
        alpha_pri = min(0.99 * alpha_pri_max, 1.0)
        alpha_dual = min(0.99 * alpha_dual_max, 1.0)
        alpha = max(alpha_pri, alpha_dual)

        self.statistics_dict['alpha_pri_list'].append(alpha_pri)
        self.statistics_dict['alpha_dual_list'].append(alpha_dual)
        self.statistics_dict['alpha_list'].append(alpha)
        return alpha_pri, alpha_dual, alpha

    def _backward_sweep(self, prev_sol:iterate, reg:str ='NO_REGULARIZE'):
        horizon = len(self.qp.stages)
        nu, nx = self.qp.stages[0].S.shape

        regularized_QP = self._regularize_QP(self.qp, reg)
        modified_QP = self._modify_QP(regularized_QP ,prev_sol)

        self.feedback = Feedback(
            P = np.empty((horizon+1, nx, nx), dtype=float),
            p = np.empty((horizon+1, nx), dtype=float), 
            K = np.empty((horizon, nu, nx), dtype=float), 
            k = np.empty((horizon, nu), dtype=float)
        )

        self.feedback.P[-1] = modified_QP.terminal.Q_e
        self.feedback.p[-1] = self.rhs_dict['rg'][-1][:nx]
        for i in reversed(range(horizon)):
            self.feedback.P[i], self.feedback.p[i], self.feedback.K[i], self.feedback.k[i] = self._riccati_recursion(
               modified_QP.stages[i], (self.feedback.P[i+1], self.feedback.p[i+1]), self.rhs_dict['rg'][i], self.rhs_dict['rb'][i]
            )

    def _modify_QP(self, qp, prev_sol:iterate):
        modified_QP = copy.deepcopy(qp)
        Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e = qp()
        C0 = qp.stages[0].C
        D0 = qp.stages[0].D
        d0 = qp.stages[0].d

        for i in range(len(self.qp.stages)):
            if i == 0:
                C_full = np.concatenate((C0, D0), axis=1)
                T_mat = np.diag(prev_sol.s0)
                Lam_mat = np.diag(prev_sol.lam0)
                rd = self.rhs_dict['rd0']
                rm = self.rhs_dict['rm0']
            else:
                C_full = np.concatenate((C, D), axis=1)
                T_mat = np.diag(prev_sol.s[i-1])
                Lam_mat = np.diag(prev_sol.lam[i-1])
                rd = self.rhs_dict['rd'][i-1]
                rm = self.rhs_dict['rm'][i-1]

            T_mat_inv = np.linalg.inv(T_mat)
            lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
            rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

            Q_tilt = lhs_modification[:qp.nx, :qp.nx]
            R_tilt = lhs_modification[qp.nx:, qp.nx:]
            S_tilt = lhs_modification[qp.nx:, :qp.nx]
            modified_QP.stages[i] = QP_stage(
                Q = qp.stages[i].Q + Q_tilt,
                R = qp.stages[i].R + R_tilt,
                S = qp.stages[i].S + S_tilt,
                q = qp.stages[i].q, r = qp.stages[i].r,
                A = qp.stages[i].A, B = qp.stages[i].B, b = qp.stages[i].b,
                C = qp.stages[i].C, D = qp.stages[i].D, d = qp.stages[i].d,
            )

            self.rhs_dict['rg'][i][:self.qp.nx] += rhs_modification[:self.qp.nx]
            self.rhs_dict['rg'][i][self.qp.nx:] += rhs_modification[self.qp.nx:]
        return modified_QP

    def _regularize_QP(self, qp:OCP_QP, method: str = 'PROJECT')-> OCP_QP:
        '''Regularize the modified QP to ensure positive definiteness of the reduced Hessian.'''
        regularized_QP = copy.deepcopy(qp)
        for i in range(self.qp.N):
            stage = qp.stages[i]
            H = np.block([[stage.Q, stage.S.T],
                          [stage.S, stage.R]])

            eigvals, eigvecs = np.linalg.eig(H)
            eigvals_reg = self._do_regularization(eigvals, method)
            H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

            Q_reg = H_reg[:qp.nx, :qp.nx]
            R_reg = H_reg[qp.nx:, qp.nx:]
            S_reg = H_reg[qp.nx:, :qp.nx]
            regularized_QP.stages[i] = QP_stage(
                Q = Q_reg,
                R = R_reg,
                S = S_reg,
                q = stage.q,
                r = stage.r,
                A = stage.A,
                B = stage.B,
                b = stage.b,
                C = stage.C,
                D = stage.D,
                d = stage.d,
            )
        Q_e = qp.terminal.Q_e
        eigvals_e, eigvecs_e = np.linalg.eig(Q_e)
        eigvals_e_reg = self._do_regularization(eigvals_e, method)
        Q_e_reg = eigvecs_e @ np.diag(eigvals_e_reg) @ eigvecs_e.T
        regularized_QP.terminal = QP_Terminal(
            Q_e = Q_e_reg,
            q_e = qp.terminal.q_e,
        )
        return regularized_QP

    def _do_regularization(self, eigvals: np.ndarray, method: str)-> np.ndarray:
            if method == 'PROJECT':
                eigvals_reg = np.array([max(eigval, 1e-4) for eigval in eigvals])
            elif method == 'MIRROR':
                eigvals_reg = np.array([abs(eigval) for eigval in eigvals])
            elif method == 'stage_LM':
                term = -min(eigvals) if min(eigvals) < 0 else 0
                eigvals_reg = eigvals + term * np.ones_like(eigvals)
            elif method == 'R_block_LM':
                term = -min(eigvals) if min(eigvals) < 0 else 0
                eigvals_reg = np.array(eigvals)
                eigvals_reg[self.qp.nx:] = eigvals[self.qp.nx:] + term * np.ones_like(eigvals[self.qp.nx:])
            else:
                eigvals_reg = eigvals
            return eigvals_reg

    def _riccati_recursion(self,
        qp_stage:QP_stage, qp_terminal:Feedback, rg, rb
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''The core step of Riccati recursion.'''
        Q, R, S, q, r, A, B, b, C, D, d = qp_stage
        P, p = qp_terminal
        q = rg[:self.qp.nx]
        r = rg[self.qp.nx:]
        b = rb
        try:
            K = -np.linalg.solve((R + B.T@P@B), (S + B.T@P@A))
            k = -np.linalg.solve((R + B.T@P@B), (r + B.T@(P@b + p)))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("singular matrix encountered in Riccati recursion. Try regularizing the QP.")
        new_P = Q + A.T@P@A - (S.T + A.T@P@B)@(-K)
        new_p = q + A.T@(P@b + p) - (S.T + A.T@P@B)@(-k)

        return new_P, new_p, K, k

    def check_QP_definiteness(self, sol_x=None, hessian: str = 'full')-> np.ndarray:
        """check definiteness of full hessian or reduced hessian with modified QP in QP problem"""
        eigval_list = []
        if hessian == 'full':
            for i, stage in enumerate(self.qp.stages):
                H = np.block([[stage.Q, stage.S.T],
                            [stage.S, stage.R]])
                eigvals = np.linalg.eigvals(H)
                eigval_list.extend(eigvals.tolist())
            eigval_term = np.linalg.eigvals(self.qp.terminal.Q_e)
            eigval_list.extend(eigval_term.tolist())

        elif hessian == 'reduced':
            self._initilize_solution(sol_x)
            self._backward_sweep(self.solution, reg='NO_REGULARIZE')
            # check R + B'PB
            for i in range(len(self.qp.stages)):
                stage = self.qp.stages[i]
                P_next = self.feedback.P[i+1]
                modified_R = stage.R + stage.B.T @ P_next @ stage.B
                eigvals = np.linalg.eigvalsh(modified_R)
                eigval_list.extend(eigvals.tolist())

        if all(np.all(eigvals > 0) for eigvals in eigval_list):
            print(f"{hessian.capitalize()} Hessian in QP is positive definite.")
        elif any(np.any(eigvals == 0) for eigvals in eigval_list):
            print(f"{hessian.capitalize()} Hessian in QP is positive semi-definite.")
        elif any(np.any(eigvals < 0) for eigvals in eigval_list):
            print(f"{hessian.capitalize()} Hessian in QP is indefinite.")

        return eigval_list

    def get_cost(self):
        cost = 0.0
        Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e = self.qp()
        for i in range(len(self.qp.stages)):
            x_i = self.solution.x[i]
            u_i = self.solution.u[i]
            cost += 0.5 * x_i.T @ Q @ x_i + 0.5 * u_i.T @ R @ u_i + x_i.T @ S.T @ u_i + q.T @ x_i + r.T @ u_i
        # terminal cost
        x_N = self.solution.x[-1]
        cost += 0.5 * x_N.T @ Q_e @ x_N + q_e.T @ x_N
        return cost

    def print_statistics(self):
        print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tbarrier\t\talpha_pri\talpha_dual')
        for i in range(self.statistics_dict['iter_count']):
            res_stat_norm = self.statistics_dict['res_stat_list'][i]
            res_eq_norm = self.statistics_dict['res_eq_list'][i]
            res_ineq_norm = self.statistics_dict['res_ineq_list'][i]
            res_comp_norm = self.statistics_dict['res_comp_list'][i]
            barrier = self.statistics_dict['barrier_list'][i]
            alpha_pri = self.statistics_dict['alpha_pri_list'][i] if i < len(self.statistics_dict['alpha_pri_list']) else 0.0
            alpha_dual = self.statistics_dict['alpha_dual_list'][i] if i < len(self.statistics_dict['alpha_dual_list']) else 0.0
            print(f"{i}\t{res_stat_norm:.2e}\t{res_eq_norm:.2e}\t{res_ineq_norm:.2e}\t{res_comp_norm:.2e}\t{barrier:.2e}\t{alpha_pri:.2e}\t{alpha_dual:.2e}")