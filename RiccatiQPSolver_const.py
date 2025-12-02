from typing import Tuple, TypeVar
from dataclasses import dataclass

from RiccatiQP_const import QP, QP_stage, iterate

import casadi as ca
import numpy as np
import copy

T = TypeVar('T', ca.SX, np.ndarray)
"""it can be 1D or 2D, symbol (ca.SX) or value (np.ndarray)."""

@dataclass
class Feedback:
    """QP terminal cost parameters arranged as (P, p)"""
    P: T
    p: T
    K: T
    k: T

    def compute_control(self, index: int, state: np.ndarray) -> np.ndarray:
        '''compute the control according to the feedback law u = k + Kx'''
        return self.K[index] @ state + self.k[index]

class QP_solver_const:

    def __init__(self, qp:QP, Const_mat:np.ndarray = None, uh:np.ndarray = None):
        self.qp = qp
        self.Const_mat = Const_mat if Const_mat is not None else np.zeros((1, qp.nx + qp.nu))
        self.uh = uh if uh is not None else np.zeros((self.Const_mat.shape[0],))
        self.feedback = None
        self.solution = None
        self.tau = 2 if Const_mat is not None else 0.0
        self.alpha_pri_list = []
        self.alpha_dual_list = []
        self.sol_init = iterate(
            x = np.zeros((qp.N + 1, qp.nx)),
            u =  np.zeros((qp.N, qp.nu)),
            pi = np.zeros((qp.N+1, qp.nx)),
            lam = np.zeros((qp.N, self.Const_mat.shape[0])),
            s = np.zeros((qp.N, self.Const_mat.shape[0]))
        )
        self.res_dict = {
            'res_stat': [],
            'res_eq': [],
            'res_ineq': [],
            'res_comp': [],
            'barrier': [],
        }

    def solve(self, x0:np.ndarray=None, max_iter:int=100, tol:float=1e-5)-> iterate:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        if x0 is not None:
            x_simulation = np.zeros((self.qp.N + 1, self.qp.nx))
            x_simulation[0] = x0
            for i in range(self.qp.N):
                x_simulation[i+1] = self.qp.stages[i].A @ x_simulation[i] + self.qp.stages[i].B @ self.sol_init.u[i] + self.qp.stages[i].b
            self.sol_init.x = x_simulation
            init_s = self.uh - self.Const_mat @ np.concatenate((x0, np.zeros((self.qp.nu,))))
            self.sol_init.s = np.vstack([init_s for _ in range(self.qp.N)])
            self.sol_init.lam = self.tau / (self.sol_init.s + 1e-10)
            self._backward_sweep(self.sol_init)
            for i in range(self.qp.N):
                self.sol_init.pi[i] = self.feedback.P[i] @ x_simulation[i] + self.feedback.p[i]
            self.sol_init.pi[-1] = self.feedback.P[-1] @ x_simulation[-1] + self.feedback.p[-1]

        self.solution = copy.deepcopy(self.sol_init)
        prev_sol = copy.deepcopy(self.sol_init)
        for i in range(max_iter):
            res = self._calculate_KKT_res(prev_sol)
            self.tau = max(self.tau*0.1, 1e-8)
            if res < tol:
                print(f"\033[32mConverged!\033[0m after {i} with KKT residual {res:.2e} and barrier parameter {self.tau:.2e}")
                return self.solution

            self._backward_sweep(prev_sol)
            self._forward_sweep(prev_sol)
            prev_sol = copy.deepcopy(self.solution)

        print(f"\033[31mNot Converged!\033[0m after {i} iterations with KKT residual {res:.2e} and barrier parameter {self.tau:.2e}")
        return self.solution


    def _calculate_KKT_res(self, this_iter:iterate)-> np.ndarray:
        N_horizon = len(self.qp.stages)
        ny = self.qp.nx + self.qp.nu
        res_stat = np.zeros((N_horizon+1, ny))
        res_eq = np.zeros((N_horizon, self.qp.nx))
        res_ineq = np.zeros((N_horizon, self.Const_mat.shape[0]))
        res_comp = np.zeros((N_horizon, self.Const_mat.shape[0]))
        e = np.ones((self.Const_mat.shape[0],))

        nx = self.qp.nx
        C_x = self.Const_mat[:, :nx]
        C_u = self.Const_mat[:, nx:]

        for i in range(len(self.qp.stages)):
            Q, R, S, q, r, A, B, b = self.qp.stages[i]
            x_i = this_iter.x[i]
            u_i = this_iter.u[i]
            slack = this_iter.s[i]
            lam_i = this_iter.lam[i]

            # stationarity residual
            res_stat[i][:nx] = Q @ x_i + q + C_x.T @ lam_i + A.T @ this_iter.pi[i+1] - this_iter.pi[i]
            res_stat[i][nx:] = R @ u_i + r + C_u.T @ lam_i + B.T @ this_iter.pi[i+1]

            # # without inequality constraints
            # if i == self.qp.N - 1:
            #     print("IGNORING INEQUALITY CONSTRAINTS IN RESIDUAL CALCULATION")
            # res_stat[i][:nx] = Q @ x_i + q + A.T @ this_iter.pi[i+1] - this_iter.pi[i]
            # res_stat[i][nx:] = R @ u_i + r + B.T @ this_iter.pi[i+1]

            # # only inequality constraints
            # if i == self.qp.N - 1:
            #     print("ONLY INEQUALITY CONSTRAINTS IN RESIDUAL CALCULATION")
            # res_stat[i][:nx] = C_x.T @ lam_i
            # res_stat[i][nx:] = C_u.T @ lam_i

            #equality residual
            res_eq[i] = A @ x_i + B @ u_i + b - this_iter.x[i+1]
            #inequality residual
            res_ineq[i] = (C_x @ x_i + C_u @ u_i + slack) - self.uh
            #complementarity residual
            res_comp[i] = (np.diag(lam_i) @ slack - self.tau * e).flatten()
        res_stat[-1][:nx] = self.qp.terminal.Q_e @ this_iter.x[-1] + self.qp.terminal.q_e - this_iter.pi[-1]

        res_stat_norm = np.linalg.norm(res_stat, ord='fro')
        self.res_dict['res_stat'].append(res_stat_norm)
        res_eq_norm = np.linalg.norm(res_eq, ord='fro')
        self.res_dict['res_eq'].append(res_eq_norm)
        res_ineq_norm = np.linalg.norm(res_ineq, ord='fro')
        self.res_dict['res_ineq'].append(res_ineq_norm)
        res_comp_norm = np.linalg.norm(res_comp, ord='fro')
        self.res_dict['res_comp'].append(res_comp_norm)
        self.res_dict['barrier'].append(self.tau)
        total_res = res_stat_norm + res_eq_norm + res_ineq_norm + res_comp_norm
        return total_res

    def _forward_sweep(self, this_iter:iterate)-> 'iterate':
        x_old = copy.deepcopy(this_iter.x)
        u_old = copy.deepcopy(this_iter.u)
        pi_old = copy.deepcopy(this_iter.pi)
        lam_old = copy.deepcopy(this_iter.lam)
        slack_old = copy.deepcopy(this_iter.s)
        param = copy.deepcopy(this_iter.p)

        # Solve for the next iterate
        x_aff, u_aff, pi_aff, lam_aff, slack_aff = self._solve_KKT_system(x_old, u_old, pi_old, lam_old, slack_old)

        # Line search to ensure positivity of slack and inequality multipliers
        delta_x = x_aff - x_old
        delta_u = u_aff - u_old
        delta_pi = pi_aff - pi_old
        delta_lam = lam_aff - lam_old
        delta_s = slack_aff - slack_old

        # do fraction to the boundary
        frac_pri = - slack_old/delta_s
        frac_dual = - lam_old/delta_lam
        # take the minimum positive step size
        alpha_pri_max = np.min(frac_pri[frac_pri > 0])
        alpha_dual_max = np.min(frac_dual[frac_dual > 0])
        # take 0.99 of the maximum step size
        alpha_pri = min(0.99 * alpha_pri_max, 1.0)
        alpha_dual = min(0.99 * alpha_dual_max, 1.0)
        self.alpha_pri_list.append(alpha_pri)
        self.alpha_dual_list.append(alpha_dual)

        # Update the solution with the step size alpha
        x_aff = x_old + alpha_pri * delta_x
        u_aff = u_old + alpha_pri * delta_u
        pi_aff = pi_old + alpha_dual * delta_pi
        lam_aff = lam_old + alpha_dual * delta_lam
        slack_aff = slack_old + alpha_pri * delta_s

        self.solution = iterate(x_aff, u_aff, pi_aff, lam_aff, slack_aff, param)

    def _solve_KKT_system(self, x_old, u_old, pi_old, lam_old, slack_old)-> iterate:
        x_aff = copy.deepcopy(x_old)
        u_aff = copy.deepcopy(u_old)
        pi_aff = copy.deepcopy(pi_old)
        lam_aff = copy.deepcopy(lam_old)
        slack_aff = copy.deepcopy(slack_old)
        e = np.ones((self.Const_mat.shape[0],))
        for i in range(len(self.qp.stages)):
            u_aff[i] = self.feedback.compute_control(i, x_aff[i])
            Q, R, S, q, r, A, B, b = self.qp.stages[i]
            lam_mat = np.diag(lam_old[i])
            slack = slack_old[i] + 1e-10

            slack_aff[i] = self.uh - self.Const_mat @ np.concatenate((x_aff[i], u_aff[i]))
            lam_aff[i] = (np.linalg.solve(np.diag(slack), ((lam_mat@slack) + self.tau * e - (lam_mat @ slack_aff[i])))).flatten() #####
            x_aff[i+1] = b + A@x_aff[i] + B@u_aff[i]
            pi_aff[i] =  self.feedback.P[i] @ x_aff[i] + self.feedback.p[i]
        pi_aff[-1] = self.feedback.P[-1] @ x_aff[-1] + self.feedback.p[-1]

        return x_aff, u_aff, pi_aff, lam_aff, slack_aff

    def _backward_sweep(self, prev_sol:iterate):
        horizon = len(self.qp.stages)
        nu, nx = self.qp.stages[0].S.shape
        modified_QP = self._modify_qp_with_constraints(prev_sol)

        self.feedback = Feedback(
            P = np.empty((horizon+1, nx, nx), dtype=float),
            p = np.empty((horizon+1, nx), dtype=float), 
            K = np.empty((horizon, nu, nx), dtype=float), 
            k = np.empty((horizon, nu), dtype=float)
        )

        self.feedback.P[-1] = modified_QP.terminal.Q_e
        self.feedback.p[-1] = modified_QP.terminal.q_e
        for i in reversed(range(horizon)):
            self.feedback.P[i], self.feedback.p[i], self.feedback.K[i], self.feedback.k[i] = self.riccati_recursion(
               modified_QP.stages[i], (self.feedback.P[i+1], self.feedback.p[i+1])
            )

    def _modify_qp_with_constraints(self, this_iter:iterate):
        lam = this_iter.lam
        slack = this_iter.s
        modified_QP = copy.deepcopy(self.qp)

        for i in range(len(self.qp.stages)):
            T_mat = np.diag(slack[i]+1e-10)
            Lam_mat = np.diag(lam[i])

            Schur = self.Const_mat.T @ np.linalg.inv(T_mat) @ Lam_mat @ self.Const_mat #####

            Q_tilt = Schur[:self.qp.nx, :self.qp.nx]
            R_tilt = Schur[self.qp.nx:, self.qp.nx:]
            S_tilt = Schur[:self.qp.nu, self.qp.nx:]
            modified_QP.stages[i] = QP_stage(
                Q = self.qp.stages[i].Q + Q_tilt,
                R = self.qp.stages[i].R + R_tilt,
                S = self.qp.stages[i].S + S_tilt,
                q = self.qp.stages[i].q, r = self.qp.stages[i].r,
                A = self.qp.stages[i].A, B = self.qp.stages[i].B,
                b = self.qp.stages[i].b
            )
        return modified_QP

    def riccati_recursion(self,
        qp_stage:QP_stage, qp_terminal:Feedback
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''The core step of Riccati recursion.'''
        Q, R, S, q, r, A, B, b = qp_stage
        P, p = qp_terminal

        K = -np.linalg.solve((R + B.T@P@B), (S + B.T@P@A))
        k = -np.linalg.solve((R + B.T@P@B), (r + B.T@(P@b + p)))
        new_P = Q + A.T@P@A - (S.T + A.T@P@B)@(-K)
        new_p = q + A.T@(P@b + p) - (S.T + A.T@P@B)@(-k)
        return new_P, new_p, K, k

    def get_cost(self):
        cost = 0.0
        for i in range(len(self.qp.stages)):
            Q, R, S, q, r, A, B, b = self.qp.stages[i]
            x_i = self.solution.x[i]
            u_i = self.solution.u[i]
            cost += 0.5 * x_i.T @ Q @ x_i + 0.5 * u_i.T @ R @ u_i + x_i.T @ S.T @ u_i + q.T @ x_i + r.T @ u_i
        # terminal cost
        Q_e = self.qp.terminal.Q_e
        q_e = self.qp.terminal.q_e
        x_N = self.solution.x[-1]
        cost += 0.5 * x_N.T @ Q_e @ x_N + q_e.T @ x_N
        return cost

    def print_statistics(self):
        # print residual table
        print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tbarrier\t\talpha_pri\talpha_dual')
        for i in range(len(self.res_dict['res_stat'])):
            res_stat_norm = self.res_dict['res_stat'][i]
            res_eq_norm = self.res_dict['res_eq'][i]
            res_ineq_norm = self.res_dict['res_ineq'][i]
            res_comp_norm = self.res_dict['res_comp'][i]
            barrier = self.res_dict['barrier'][i]
            alpha_pri = self.alpha_pri_list[i] if i < len(self.alpha_pri_list) else 0.0
            alpha_dual = self.alpha_dual_list[i] if i < len(self.alpha_dual_list) else 0.0
            print(f"{i}\t{res_stat_norm:.2e}\t{res_eq_norm:.2e}\t{res_ineq_norm:.2e}\t{res_comp_norm:.2e}\t{barrier:.2e}\t{alpha_pri:.2e}\t{alpha_dual:.2e}")