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

class QP_solver:

    def __init__(self, qp:QP):
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
        self.misc_dict = {
            'alpha_pri_list' : [],
            'alpha_dual_list' : [],
            'alpha_list' : [],
            'barrier_list' : [],
        }
        self.solution = iterate(
            x = np.zeros((qp.N + 1, qp.nx)),
            u =  np.zeros((qp.N, qp.nu)),
            pi = np.zeros((qp.N, self.qp.nx)),
            lam0= np.zeros((qp.nh, )),
            lam = np.zeros((qp.N-1, self.qp.nh)),
            s0 = np.zeros((self.qp.nh, )),
            s = np.zeros((qp.N-1, self.qp.nh)),
        )

    def _initilize_solution(self, x0:np.ndarray)-> iterate:
        self.solution.x = np.vstack([x0 for i in range(self.qp.N+1)])
        self.solution.u = np.zeros((self.qp.N, self.qp.nu))
        init_s = self.qp.stages[1].d - self.qp.stages[1].C @ self.solution.x[1] - self.qp.stages[1].D @ self.solution.u[1]
        self.solution.s = np.vstack([init_s for _ in range(self.qp.N-1)])
        self.solution.lam = self.tau / self.solution.s
        self.solution.pi = np.zeros((self.qp.N, self.qp.nx))

        C0 = np.vstack((-np.eye(self.qp.nx), np.eye(self.qp.nx), self.qp.stages[0].C))
        D0 = np.vstack((0*np.zeros((self.qp.nx, self.qp.nu)), 0*np.zeros((self.qp.nx, self.qp.nu)), self.qp.stages[0].D))
        init_d0 =  np.hstack(( -(x0-1e-1)*np.ones((self.qp.nx,)), (x0+1e-1)*np.ones((self.qp.nx,)), self.qp.stages[0].d))
        init_s0 = init_d0 - C0 @ self.solution.x[0] - D0 @ self.solution.u[0]
        d0 = np.hstack(( -(x0)*np.ones((self.qp.nx,)), (x0)*np.ones((self.qp.nx,)), self.qp.stages[0].d))
        self.qp.stages[0] = QP_stage(
            Q = self.qp.stages[0].Q,
            R = self.qp.stages[0].R,
            S = self.qp.stages[0].S,
            q = self.qp.stages[0].q,
            r = self.qp.stages[0].r,
            A = self.qp.stages[0].A,
            B = self.qp.stages[0].B,
            b = self.qp.stages[0].b,
            C = C0,
            D = D0,
            d = d0,
        )
        self.solution.s0 = init_s0
        self.solution.lam0 = self.tau / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=20, tol:float=1e-6, formulation: str = "delta")-> iterate:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        if x0 is not None:
            self._initilize_solution(x0)
        for i in range(max_iter):
            prev_sol = copy.deepcopy(self.solution)
            res = self._update_rhs(prev_sol, formulation)
            if res < tol:
                print(f"\033[32mConverged!\033[0m after {i} with KKT residual {res:.2e} and barrier parameter {self.misc_dict['barrier_list'][-1]:.2e}")
                return self.solution
            self._backward_sweep(prev_sol, formulation)
            self._forward_sweep(prev_sol, formulation)
            self.tau = max(self.tau*0.1, 1e-8)
        print(f"\033[31mNot Converged!\033[0m after {max_iter} iterations with KKT residual {res:.2e} and barrier parameter {self.misc_dict['barrier_list'][-1]:.2e}")
        return self.solution

    def _update_rhs(self, this_iter:iterate, formulation: str)-> np.ndarray:
        nx = self.qp.nx
        nu = self.qp.nu
        N = self.qp.N
        stage0 = self.qp.stages[0]

        if formulation == 'delta':
            res_stat = np.zeros((N+1, nx+nu))
            res_eq = np.zeros((N, nx))
            res_ineq0 = np.zeros((1, stage0.C.shape[0]))
            res_ineq = np.zeros((N-1, self.qp.stages[1].C.shape[0]))
            res_comp0 = np.zeros((1, stage0.C.shape[0]))
            res_comp = np.zeros((N-1, self.qp.stages[1].C.shape[0]))
            Q, R, S, q, r, A, B, b, C, D, d = self.qp.stages[1]
            C0 = self.qp.stages[0].C
            D0 = self.qp.stages[0].D

            for i in range(len(self.qp.stages)):
                # stationarity residual
                if i == 0:
                    q0 = stage0.q
                    r0 = stage0.r
                    x0_i = this_iter.x[0]
                    u0_i = this_iter.u[0]
                    res_stat[i][nx:nx+nu] = (
                        stage0.R @ u0_i + stage0.S @ x0_i + r0
                        + stage0.B.T @ this_iter.pi[0] + D0.T @ this_iter.lam0
                    )
                else:
                    x_i = this_iter.x[i]
                    u_i = this_iter.u[i]
                    res_stat[i][:nx] = (
                        Q @ x_i + stage0.S.T @ u_i + q
                        +  A.T @ this_iter.pi[i] + C.T @ this_iter.lam[i-1] - this_iter.pi[i-1]
                    )
                    res_stat[i][nx:nx+nu] = (
                        R @ u_i + S @ x_i + r
                        + B.T @ this_iter.pi[i] + D.T @ this_iter.lam[i-1]
                    )

                #equality residual
                x_i = this_iter.x[i]
                u_i = this_iter.u[i]
                x_next = this_iter.x[i+1]
                res_eq[i] = (
                    A @ x_i + B @ u_i - x_next - b
                )

                #inequality residual
                if i == 0:
                    x0_i = this_iter.x[0]
                    u0_i = this_iter.u[0]
                    res_ineq0 =  C0 @ x0_i + D0 @ u0_i + this_iter.s0 - stage0.d
                else:
                    x_i = this_iter.x[i]
                    u_i = this_iter.u[i]
                    res_ineq[i-1] = (C @ x_i + D @ u_i + this_iter.s[i-1]) - d

                #complementarity residual
                if i == 0:
                    res_comp0 = (
                this_iter.lam0 * this_iter.s0 - self.tau * np.ones((len(this_iter.lam0), ))
                )
                else:
                    res_comp[i-1] = (np.diag(this_iter.lam[i-1]) @ this_iter.s[i-1] - self.tau * np.ones((len(this_iter.lam[i-1]), ))).flatten()

            # terminal cost
                Q_N = self.qp.terminal.Q_e
                q_N = self.qp.terminal.q_e
                x_N = this_iter.x[N]
                res_stat[N][:nx] = (
                    Q_N @ x_N + q_N - this_iter.pi[N-1]
                )

            self.rhs_dict['rg'] = res_stat
            self.rhs_dict['rb'] = res_eq
            self.rhs_dict['rd0'] = res_ineq0
            self.rhs_dict['rd'] = res_ineq
            self.rhs_dict['rm0'] = res_comp0
            self.rhs_dict['rm'] = res_comp
            self.misc_dict['barrier_list'].append(self.tau)
            res = np.linalg.norm(np.hstack((res_stat.flatten(), res_eq.flatten(), res_ineq0.flatten(), res_ineq.flatten(), res_comp0.flatten(), res_comp.flatten())))

        return res

    def _forward_sweep(self, this_iter:iterate, formulation: str)-> 'iterate':
        x_old = copy.deepcopy(this_iter.x)
        u_old = copy.deepcopy(this_iter.u)
        pi_old = copy.deepcopy(this_iter.pi)
        lam0_old = copy.deepcopy(this_iter.lam0)
        lam_old = copy.deepcopy(this_iter.lam)
        s0_old = copy.deepcopy(this_iter.s0)
        s_old = copy.deepcopy(this_iter.s)
        param = copy.deepcopy(this_iter.p)

        # Solve for the next iterate
        delta_dict = self._solve_KKT_system(this_iter, formulation=formulation)

        alpha_pri, alpha_dual, alpha = self._fraction_to_boundary(this_iter, delta_dict)

        # Update the solution with the step size alpha
        x_aff = x_old + alpha_pri * delta_dict['delta_x']
        u_aff = u_old + alpha_pri * delta_dict['delta_u']
        pi_aff = pi_old + alpha_dual * delta_dict['delta_pi']
        lam0_aff = lam0_old + alpha_dual * delta_dict['delta_lam0']
        lam_aff = lam_old + alpha_dual * delta_dict['delta_lam']
        s0_aff = s0_old + alpha_pri * delta_dict['delta_s0']
        s_aff = s_old + alpha_pri * delta_dict['delta_s']

        self.solution = iterate(x_aff, u_aff, pi_aff, lam0_aff, lam_aff, s0_aff, s_aff, param)

    def _solve_KKT_system(self, this_iter:iterate, formulation)-> iterate:
        delta_x = copy.deepcopy(this_iter.x)
        delta_u = copy.deepcopy(this_iter.u)
        delta_pi = copy.deepcopy(this_iter.pi)
        delta_lam0 = copy.deepcopy(this_iter.lam0)
        delta_lam = copy.deepcopy(this_iter.lam)
        delta_s0 = copy.deepcopy(this_iter.s0)
        delta_s = copy.deepcopy(this_iter.s)
        Q, R, S, q, r, A, B, b, C, D, d = self.qp.stages[1]
        stage0 = self.qp.stages[0]

        if formulation == 'delta':
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
                    rhs_d =  stage0.C @ this_iter.x[i] + stage0.D @ this_iter.u[i] + this_iter.s0 - stage0.d
                    delta_s0 = -(stage0.C @delta_x[0] + stage0.D @delta_u[0]) - rhs_d
                else:
                    rhs_d = C @ this_iter.x[i] + D @ this_iter.u[i] + this_iter.s[i-1] - d
                    delta_s[i-1] = -(C @delta_x[i] + D @delta_u[i]) - rhs_d
                # recover lam
                if i == 0:
                    rhs_m = (this_iter.lam0 * this_iter.s0 - self.tau * np.ones(len(this_iter.lam0)))
                    rhs_m_modified = -rhs_m - (this_iter.lam0 * delta_s0) 
                    delta_lam0 = (np.linalg.solve(np.diag(this_iter.s0), rhs_m_modified)).flatten()
                else:
                    rhs_m = (this_iter.lam[i-1] * this_iter.s[i-1] - self.tau * np.ones(len(this_iter.lam[i-1])))
                    rhs_lam = -rhs_m - (this_iter.lam[i-1] * delta_s[i-1])
                    delta_lam[i-1] = (np.linalg.solve(np.diag(this_iter.s[i-1]), rhs_lam)).flatten()
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

        elif formulation == 'absolute':
            for i in range(len(self.qp.stages)):
                # recover x and u
                rhs_b = b
                delta_u[i] = self.feedback.compute_control(i, delta_x[i])
                delta_x[i+1] = A@delta_x[i] + B@delta_u[i] + rhs_b
                # recover s
                if i == 0:
                    rhs_d = -stage0.d
                    delta_s0 = -(stage0.C @delta_x[0] - stage0.D @delta_u[0]) - rhs_d
                else:
                    rhs_d = -d
                    delta_s[i-1] = -(C @delta_x[i] + D @delta_u[i]) - rhs_d
                # recover lam
                if i == 0:
                    rhs_m = (-this_iter.lam0 * this_iter.s0 - self.tau * np.ones(len(this_iter.lam0)))
                    rhs_m_modified = -rhs_m - (this_iter.lam0 * delta_s0) 
                    delta_lam0 = (np.linalg.solve(np.diag(this_iter.s0), rhs_m_modified)).flatten()
                else:
                    rhs_m = (-this_iter.lam[i-1] * this_iter.s[i-1] - self.tau * np.ones(len(this_iter.lam[i-1])))
                    rhs_lam = -rhs_m - (this_iter.lam[i-1] * delta_s[i-1])
                    delta_lam[i-1] = (np.linalg.solve(np.diag(this_iter.s[i-1]), rhs_lam)).flatten()
                # recover pi
                delta_pi[i] =  self.feedback.P[i+1] @ delta_x[i+1] + self.feedback.p[i+1]

            delta_dict = {
                'delta_x': delta_x - this_iter.x,
                'delta_u': delta_u - this_iter.u,
                'delta_pi': delta_pi - this_iter.pi,
                'delta_lam0': delta_lam0 - this_iter.lam0,
                'delta_lam': delta_lam - this_iter.lam,
                'delta_s0': delta_s0 - this_iter.s0,
                'delta_s': delta_s - this_iter.s,
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

        self.misc_dict['alpha_pri_list'].append(alpha_pri)
        self.misc_dict['alpha_dual_list'].append(alpha_dual)
        self.misc_dict['alpha_list'].append(alpha)
        return alpha_pri, alpha_dual, alpha

    def _backward_sweep(self, prev_sol:iterate, formulation: str):
        horizon = len(self.qp.stages)
        nu, nx = self.qp.stages[0].S.shape
        modified_QP = self._modify_QP_by_elimination(prev_sol, formulation)

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

    def _modify_QP_by_elimination(self, this_iter:iterate, formulation: str):
        modified_QP = copy.deepcopy(self.qp)

        if formulation == 'delta':
            for i in range(len(self.qp.stages)):
                if i == 0:
                    C_full = np.concatenate((self.qp.stages[0].C, self.qp.stages[0].D), axis=1)
                    T_mat = np.diag(this_iter.s0)
                    Lam_mat = np.diag(this_iter.lam0)
                    rd = self.rhs_dict['rd0']
                    rm = self.rhs_dict['rm0']
                else:
                    C_full = np.concatenate((self.qp.stages[i].C, self.qp.stages[i].D), axis=1)
                    T_mat = np.diag(this_iter.s[i-1])
                    Lam_mat = np.diag(this_iter.lam[i-1])
                    rd = self.rhs_dict['rd'][i-1]
                    rm = self.rhs_dict['rm'][i-1]

                T_mat_inv = np.linalg.inv(T_mat)
                lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
                rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

                Q_tilt = lhs_modification[:self.qp.nx, :self.qp.nx]
                R_tilt = lhs_modification[self.qp.nx:, self.qp.nx:]
                S_tilt = lhs_modification[:self.qp.nu, self.qp.nx:]
                modified_QP.stages[i] = QP_stage(
                    Q = self.qp.stages[i].Q + Q_tilt,
                    R = self.qp.stages[i].R + R_tilt,
                    S = self.qp.stages[i].S + S_tilt,
                    q = self.qp.stages[i].q, r = self.qp.stages[i].r,
                    A = self.qp.stages[i].A, B = self.qp.stages[i].B, b = self.qp.stages[i].b,
                    C = self.qp.stages[i].C, D = self.qp.stages[i].D, d = self.qp.stages[i].d,
                )

                self.rhs_dict['rg'][i][:self.qp.nx] += rhs_modification[:self.qp.nx]
                self.rhs_dict['rg'][i][self.qp.nx:] += rhs_modification[self.qp.nx:]
            # terminal stage has no modification
        return modified_QP

    def _riccati_recursion(self,
        qp_stage:QP_stage, qp_terminal:Feedback, rg, rb
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''The core step of Riccati recursion.'''
        Q, R, S, q, r, A, B, b, C, D, d = qp_stage
        P, p = qp_terminal
        q = rg[:self.qp.nx]
        r = rg[self.qp.nx:]
        b = rb

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