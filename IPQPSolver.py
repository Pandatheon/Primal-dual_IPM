from typing import Tuple, TypeVar
from RiccatiQP_const import QP, QP_stage, iterate

import casadi as ca
import numpy as np
import copy

T = TypeVar('T', ca.SX, np.ndarray)
"""it can be 1D or 2D, symbol (ca.SX) or value (np.ndarray)."""

class QP_solver:

    def __init__(self, qp:QP):
        self.qp = qp
        self.feedback = None
        self.tau = 2
        self.res_dict = {
            'res_stat': [],
            'res_eq': [],
            'res_ineq': [],
            'res_comp': [],
            'barrier': [],
        }
        self.alpha_pri_list = []
        self.alpha_dual_list = []
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

    def solve(self, x0:np.ndarray=None, max_iter:int=20, tol:float=1e-6)-> iterate:
        if x0 is not None:
            self._initilize_solution(x0)
        for iter in range(max_iter):
            prev_sol = copy.deepcopy(self.solution)
            KKT_mat, res_vec = self._establish_KKT_system(prev_sol)
            if np.linalg.norm(res_vec) < tol:
                print(f"\033[32mConverged!\033[0m after {iter} with KKT residual {np.linalg.norm(res_vec):.2e} and barrier parameter {self.res_dict['barrier'][-1]:.2e}")
                return self.solution
            delta_z = np.linalg.solve(KKT_mat, -res_vec)
            delta_dict = self._rearrange_delta_vector(delta_z)
            alpha_pri, alpha_dual, alpha = self._fraction_to_boundary(prev_sol,delta_dict)
            self._update_solution(prev_sol, delta_dict, alpha_pri, alpha_dual, alpha)
            self.tau = max(0.1*self.tau, 1e-8)
            self.res_dict['barrier'].append(self.tau)
        print(f"\033[31mNot Converged!\033[0m after {max_iter} iterations with KKT residual {np.linalg.norm(res_vec):.2e} and barrier parameter {self.res_dict['barrier'][-1]:.2e}")
        return self.solution

    def _establish_KKT_system(self, this_iter:iterate) -> Tuple[np.ndarray, np.ndarray]:
        # with x0 elimination
        nx = self.qp.nx
        nu = self.qp.nu
        N = self.qp.N

        dimension_H = N*(nx+nu) + nx
        H_matrix = np.zeros((dimension_H, dimension_H))
        for i in range(N):
            # stage cost
            Q_i = self.qp.stages[i].Q
            S_i = self.qp.stages[i].S
            R_i = self.qp.stages[i].R
            H_matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu):i*(nx+nu)+nx] = Q_i
            H_matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = R_i
            H_matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = S_i.T
            H_matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu):i*(nx+nu)+nx] = S_i

        # terminal cost
        Q_N = self.qp.terminal.Q_e
        H_matrix[N*(nx+nu):N*(nx+nu)+nx, N*(nx+nu):N*(nx+nu)+nx] = Q_N

        dimension_A = N*nx
        A_matrix = np.zeros((dimension_A, dimension_H))
        for i in range(N):
            A_matrix[i*nx:i*nx+nx, i*(nx+nu):i*(nx+nu)+nx] = self.qp.stages[i].A
            A_matrix[i*nx:i*nx+nx, i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.qp.stages[i].B
            A_matrix[i*nx:i*nx+nx, (i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = -np.eye(nx)

        dimension_C0 = self.qp.stages[0].C.shape[0]
        dimension_C =  dimension_C0 + (N-1)*self.qp.stages[1].C.shape[1]
        C_matrix = np.zeros((dimension_C, dimension_H))
        C0 = self.qp.stages[0].C
        D0 = self.qp.stages[0].D
        C = self.qp.stages[1].C
        D = self.qp.stages[1].D

        C_matrix[0:dimension_C0, 0:nx] = C0
        C_matrix[0:dimension_C0, nx:nx+nu] = D0
        for i in range(1, N):
            C_matrix[dimension_C0+(i-1)*nx:dimension_C0+i*nx, i*(nx+nu):i*(nx+nu)+nx] = C
            C_matrix[dimension_C0+(i-1)*nx:dimension_C0+i*nx, i*(nx+nu)+nx:(i+1)*(nx+nu)] = D

        dimension_T = dimension_C
        slacks = np.concatenate([this_iter.s0, this_iter.s.flatten()], axis=0)
        lambdas = np.concatenate([this_iter.lam0, this_iter.lam.flatten()], axis=0)
        T_matrix = np.diag(slacks)
        lambda_matrix = np.diag(lambdas)

        # combine all
        dimension = H_matrix.shape[0] + A_matrix.shape[0] + C_matrix.shape[0] + T_matrix.shape[0]
        KKT_matrix = np.block([
            [H_matrix, A_matrix.T, C_matrix.T, np.zeros((dimension_H, dimension_T))],
            [A_matrix, np.zeros((dimension_A, dimension_A + dimension_C + dimension_T))],
            [C_matrix, np.zeros((dimension_C, dimension_A + dimension_C)), np.eye(dimension_C)],
            [np.zeros((dimension_T, dimension_H + dimension_A)), T_matrix, lambda_matrix],
        ])

        res_stat = np.zeros((dimension_H, ))
        rest_eq = np.zeros((dimension_A, ))
        res_ineq = np.zeros((dimension_C, ))
        res_comp = np.zeros((dimension_T, ))

        # stationarity
        stage0 = self.qp.stages[0]
        q0 = stage0.q
        r0 = stage0.r
        x0_i = this_iter.x[0]
        u0_i = this_iter.u[0]
        res_stat[0:nx] = (
            stage0.Q @ x0_i + stage0.S.T @ u0_i + q0
            + stage0.A.T @ this_iter.pi[0] + C0.T @ this_iter.lam0
        )
        res_stat[nx:nx+nu] = (
            stage0.R @ u0_i + stage0.S @ x0_i + r0
            + stage0.B.T @ this_iter.pi[0] + D0.T @ this_iter.lam0
        )
        for i in range(1, N):
            stage = self.qp.stages[i]
            q_i = stage.q
            r_i = stage.r
            x_i = this_iter.x[i]
            u_i = this_iter.u[i]
            res_stat[i*(nx+nu):i*(nx+nu)+nx] = (
                stage.Q @ x_i + stage.S.T @ u_i + q_i
                + stage.A.T @ this_iter.pi[i] + C.T @ this_iter.lam[i-1] - this_iter.pi[i-1]
            )
            res_stat[i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = (
                stage.R @ u_i + stage.S @ x_i + r_i
                + stage.B.T @ this_iter.pi[i] + D.T @ this_iter.lam[i-1]
            )

        # terminal cost
        Q_N = self.qp.terminal.Q_e
        q_N = self.qp.terminal.q_e
        x_N = this_iter.x[N]
        res_stat[N*(nx+nu):N*(nx+nu)+nx] = (
            Q_N @ x_N + q_N - this_iter.pi[N-1]
        )

        # dynamics equality
        for i in range(N):
            stage = self.qp.stages[i]
            x_i = this_iter.x[i]
            u_i = this_iter.u[i]
            x_next = this_iter.x[i+1]
            rest_eq[i*nx:i*nx+nx] = (
                stage.A @ x_i + stage.B @ u_i - x_next - stage.b
            )

        # inequality constraints
        stage0 = self.qp.stages[0]
        x0_i = this_iter.x[0]
        u0_i = this_iter.u[0]
        res_ineq[0:dimension_C0] = (
            stage0.C @ x0_i + stage0.D @ u0_i + this_iter.s0 - stage0.d
        )
        for i in range(1, N):
            stage = self.qp.stages[i]
            x_i = this_iter.x[i]
            u_i = this_iter.u[i]
            res_ineq[dimension_C0+(i-1)*nx:dimension_C0+i*nx] = (
                C @ x_i + D @ u_i + this_iter.s[i-1] - self.qp.stages[i].d
            )

        # complementarity
        res_comp[0:dimension_C0] = (
            this_iter.lam0 * this_iter.s0 - self.tau * np.ones((len(this_iter.lam0), ))
        )
        for i in range(1, N):
            res_comp[dimension_C0+(i-1)*nx:dimension_C0+i*nx] = (
                this_iter.lam[i-1] * this_iter.s[i-1] - self.tau * np.ones((nx, ))
            )

        # combine all residuals
        rhs_vector = np.zeros((dimension, ))
        rhs_vector[0:dimension_H] = res_stat
        self.res_dict['res_stat'].append(np.linalg.norm(res_stat, ord=np.inf))
        rhs_vector[dimension_H:dimension_H + dimension_A] = rest_eq
        self.res_dict['res_eq'].append(np.linalg.norm(rest_eq, ord=np.inf))
        rhs_vector[dimension_H + dimension_A:dimension_H + dimension_A + dimension_C] = res_ineq
        self.res_dict['res_ineq'].append(np.linalg.norm(res_ineq, ord=np.inf))
        rhs_vector[dimension_H + dimension_A + dimension_C:dimension] = res_comp
        self.res_dict['res_comp'].append(np.linalg.norm(res_comp, ord=np.inf))
        return KKT_matrix, rhs_vector

    def _fraction_to_boundary(self, prev_sol: iterate, step_dict: dict) -> iterate:
        s0_old = prev_sol.s0
        s_old = prev_sol.s
        lam0_old = prev_sol.lam0
        lam_old = prev_sol.lam
        delta_s0 = step_dict['delta_s0']
        delta_s = step_dict['delta_s']
        delta_lam0 = step_dict['delta_lam0']
        delta_lam = step_dict['delta_lam']

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
        self.alpha_pri_list.append(alpha_pri)
        self.alpha_dual_list.append(alpha_dual)

        alpha = max(alpha_pri, alpha_dual)
        return alpha_pri, alpha_dual, alpha
 
    def _rearrange_delta_vector(self, delta_z: np.ndarray) -> dict:
        delta_dict = {
            'delta_x': [],
            'delta_u': [],
            'delta_pi': [],
            'delta_lam0': [],
            'delta_lam': [],
            'delata_s0': [],
            'delta_s': []
        }
        delta_x = np.zeros((self.qp.N + 1, self.qp.nx))
        delta_u = np.zeros((self.qp.N, self.qp.nu))
        for i in range(self.qp.N):
            index = i * (self.qp.nx + self.qp.nu)
            delta_x[i, :] = delta_z[index:index + self.qp.nx]
            delta_u[i, :] = delta_z[index + self.qp.nx:index + self.qp.nx + self.qp.nu]
        delta_x[self.qp.N, :] = delta_z[self.qp.N * (self.qp.nx + self.qp.nu):self.qp.N * (self.qp.nx + self.qp.nu) + self.qp.nx]
        delta_dict['delta_x'] = delta_x
        delta_dict['delta_u'] = delta_u
        index_after_xu = (self.qp.N + 1)*self.qp.nx + self.qp.N*self.qp.nu
        delta_pi = delta_z[index_after_xu:index_after_xu + (self.qp.N)*self.qp.nx]
        delta_pi = delta_pi.reshape((self.qp.N, self.qp.nx))
        delta_dict['delta_pi'] = delta_pi
        index_after_pi = index_after_xu + (self.qp.N)*self.qp.nx
        delta_lam0 =  delta_z[index_after_pi:index_after_pi + len(self.qp.stages[0].d)]
        delta_lam = delta_z[index_after_pi + len(self.qp.stages[0].d):index_after_pi + len(self.qp.stages[0].d)+(self.qp.N-1)*self.qp.nx]
        delta_lam = delta_lam.reshape((self.qp.N-1, self.qp.nx))
        delta_dict['delta_lam0'] = delta_lam0
        delta_dict['delta_lam'] = delta_lam
        index_after_lam = index_after_pi + len(self.qp.stages[0].d)+(self.qp.N-1)*self.qp.nx
        delta_s0 = delta_z[index_after_lam:index_after_lam + len(self.qp.stages[0].d)]
        delta_s = delta_z[index_after_lam + len(self.qp.stages[0].d):index_after_lam + len(self.qp.stages[0].d)+(self.qp.N-1)*self.qp.nx]
        delta_s = delta_s.reshape((self.qp.N-1, self.qp.nx))
        delta_dict['delta_s0'] = delta_s0
        delta_dict['delta_s'] = delta_s
        return delta_dict

    def _update_solution(self, prev_sol: iterate, step_dict: dict, alpha_pri: float, alpha_dual: float, alpha: float):
            self.solution.x = prev_sol.x + step_dict['delta_x'] * alpha_pri
            self.solution.u = prev_sol.u + step_dict['delta_u'] * alpha_pri
            self.solution.pi = prev_sol.pi + step_dict['delta_pi'] * alpha_dual
            self.solution.lam0 = prev_sol.lam0 + step_dict['delta_lam0'] * alpha_dual
            self.solution.lam = prev_sol.lam + step_dict['delta_lam'] * alpha_dual
            self.solution.s0 = prev_sol.s0 + step_dict['delta_s0'] * alpha_pri
            self.solution.s = prev_sol.s + step_dict['delta_s'] * alpha_pri

    def get_cost(self) -> float:
        cost = 0.0
        for i in range(self.qp.N):
            x_i = self.solution.x[i]
            u_i = self.solution.u[i]
            stage = self.qp.stages[i]
            cost += 0.5 * x_i.T @ stage.Q @ x_i + 0.5 * u_i.T @ stage.R @ u_i + x_i.T @ stage.S.T @ u_i + stage.q.T @ x_i + stage.r.T @ u_i
        # terminal cost
        x_N = self.solution.x[self.qp.N]
        Q_e = self.qp.terminal.Q_e
        q_e = self.qp.terminal.q_e
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