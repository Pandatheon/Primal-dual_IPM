from typing import Tuple

from QP_formulation import Dense_QP, iterate
import numpy as np
import copy


class QP_solver:

    def __init__(self, qp:Dense_QP):
        self.qp = qp
        self.feedback = None
        self.tau = 2.0
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
        init_s = self.qp.d_list[1] - self.qp.C_list[1] @ self.solution.x[1] - self.qp.D_list[1] @ self.solution.u[1]
        self.solution.s = np.vstack([init_s for _ in range(self.qp.N-1)])
        self.solution.lam = self.tau / self.solution.s
        self.solution.pi = np.zeros((self.qp.N, self.qp.nx))

        init_d0 =  np.hstack(( -(x0-1e-1)*np.ones((self.qp.nx,)), (x0+1e-1)*np.ones((self.qp.nx,)), self.qp.d_list[0]))
        init_s0 = init_d0 - self.qp.C0 @ self.solution.x[0] - self.qp.D0 @ self.solution.u[0]
        d0 = np.hstack(( -(x0)*np.ones((self.qp.nx,)), (x0)*np.ones((self.qp.nx,)), self.qp.d_list[0]))
        self.qp.d0 = d0
        self.solution.s0 = init_s0
        self.solution.lam0 = self.tau / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=50, tol:float=1e-6)-> iterate:
        if x0 is not None:
            self._initilize_solution(x0)
        for iter in range(max_iter):
            prev_sol = copy.deepcopy(self.solution)
            KKT_mat, res_vec = self._establish_KKT_system(prev_sol)
            if np.linalg.norm(res_vec) < tol:
                print(f"\033[32mConverged!\033[0m after {iter} with KKT residual {np.linalg.norm(res_vec):.2e} and barrier parameter {self.statistics_dict['barrier_list'][-1]:.2e}")
                return self.solution
            delta_z = np.linalg.solve(KKT_mat, -res_vec)
            delta_dict = self._rearrange_delta_vector(delta_z)
            alpha_pri, alpha_dual, alpha = self._fraction_to_boundary(prev_sol,delta_dict)
            self._update_solution(prev_sol, delta_dict, alpha_pri, alpha_dual, alpha)
            self.tau = max(0.1*self.tau, 1e-8)
            self.statistics_dict['barrier_list'].append(self.tau)
            self.statistics_dict['iter_count'] += 1
        print(f"\033[31mNot Converged!\033[0m after {max_iter} iterations with KKT residual {np.linalg.norm(res_vec):.2e} and barrier parameter {self.statistics_dict['barrier_list'][-1]:.2e}")
        return self.solution

    def _establish_KKT_system(self, this_iter:iterate) -> Tuple[np.ndarray, np.ndarray]:
        nx = self.qp.nx
        nu = self.qp.nu
        N = self.qp.N

        dimension_H = self.qp.dimension_H
        H_matrix = self.qp.H_matrix

        dimension_A = self.qp.dimension_A
        A_matrix = self.qp.A_matrix

        dimension_C = self.qp.dimension_C
        C_matrix = self.qp.C_matrix

        dimension_T = self.qp.dimension_T
        slacks = np.concatenate([this_iter.s0, this_iter.s.flatten()], axis=0)
        lambdas = np.concatenate([this_iter.lam0, this_iter.lam.flatten()], axis=0)
        T_matrix = np.diag(slacks)
        lambda_matrix = np.diag(lambdas)

        # combine all
        dimension = dimension_H + dimension_A + dimension_C + dimension_T
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
        res_stat[0:nx] = (
            self.qp.Q_list[0] @ this_iter.x[0] + self.qp.S_list[0].T @ this_iter.u[0] + self.qp.q_list[0]
            + self.qp.A_list[0].T @ this_iter.pi[0] + self.qp.C0.T @ this_iter.lam0
        )
        res_stat[nx:nx+nu] = (
            self.qp.R_list[0] @ this_iter.u[0] + self.qp.S_list[0] @ this_iter.x[0] + self.qp.r_list[0]
            + self.qp.B_list[0].T @ this_iter.pi[0] + self.qp.D0.T @ this_iter.lam0
        )
        for i in range(1, N):
            res_stat[i*(nx+nu):i*(nx+nu)+nx] = (
                self.qp.Q_list[i] @ this_iter.x[i] + self.qp.S_list[i].T @ this_iter.u[i] + self.qp.q_list[i]
                + self.qp.A_list[i].T @ this_iter.pi[i] + self.qp.C_list[i].T @ this_iter.lam[i-1] - this_iter.pi[i-1]
            )
            res_stat[i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = (
                self.qp.R_list[i] @ this_iter.u[i] + self.qp.S_list[i] @ this_iter.x[i] + self.qp.r_list[i]
                + self.qp.B_list[i].T @ this_iter.pi[i] + self.qp.D_list[i].T @ this_iter.lam[i-1]
            )

        # terminal cost
        res_stat[N*(nx+nu):N*(nx+nu)+nx] = (
            self.qp.Q_e @ this_iter.x[N] + self.qp.q_e - this_iter.pi[N-1]
        )

        # dynamics equality
        for i in range(N):
            x_next = this_iter.x[i+1]
            rest_eq[i*nx:i*nx+nx] = (
                self.qp.A_list[i] @ this_iter.x[i] + self.qp.B_list[i] @ this_iter.u[i] - x_next - self.qp.b_list[i]
            )

        # inequality constraints
        res_ineq[0:self.qp.nh0] = (
            self.qp.C0 @ this_iter.x[0] + self.qp.D0 @ this_iter.u[0] + this_iter.s0 - self.qp.d0
        )
        for i in range(1, N):
            res_ineq[self.qp.nh0+(i-1)*self.qp.nh:self.qp.nh0+i*self.qp.nh] = (
                self.qp.C_list[i] @ this_iter.x[i] + self.qp.D_list[i] @ this_iter.u[i] + this_iter.s[i-1] - self.qp.d_list[i]
            )

        # complementarity
        res_comp[0:self.qp.nh0] = (
            this_iter.lam0 * this_iter.s0 - self.tau * np.ones((len(this_iter.lam0), ))
        )
        for i in range(1, N):
            res_comp[self.qp.nh0+(i-1)*self.qp.nh:self.qp.nh0+i*self.qp.nh] = (
                this_iter.lam[i-1] * this_iter.s[i-1] - self.tau * np.ones((self.qp.nh, ))
            )

        # combine all residuals
        rhs_vector = np.zeros((dimension, ))
        rhs_vector[0:dimension_H] = res_stat
        self.statistics_dict['res_stat_list'].append(np.linalg.norm(res_stat, ord=np.inf))
        rhs_vector[dimension_H:dimension_H + dimension_A] = rest_eq
        self.statistics_dict['res_eq_list'].append(np.linalg.norm(rest_eq, ord=np.inf))
        rhs_vector[dimension_H + dimension_A:dimension_H + dimension_A + dimension_C] = res_ineq
        self.statistics_dict['res_ineq_list'].append(np.linalg.norm(res_ineq, ord=np.inf))
        rhs_vector[dimension_H + dimension_A + dimension_C:dimension] = res_comp
        self.statistics_dict['res_comp_list'].append(np.linalg.norm(res_comp, ord=np.inf))
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
        self.statistics_dict['alpha_pri_list'].append(alpha_pri)
        self.statistics_dict['alpha_dual_list'].append(alpha_dual)

        alpha = max(alpha_pri, alpha_dual)
        return alpha_pri, alpha_dual, alpha

    def _rearrange_delta_vector(self, delta_z: np.ndarray) -> dict:
        delta_dict = {
            'delta_x': [],
            'delta_u': [],
            'delta_pi': [],
            'delta_lam0': [],
            'delta_lam': [],
            'delta_s0': [],
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
        delta_lam0 =  delta_z[index_after_pi:index_after_pi + len(self.qp.d0)]
        delta_lam = delta_z[index_after_pi + len(self.qp.d0):index_after_pi + len(self.qp.d0)+(self.qp.N-1)*self.qp.nh]
        delta_lam = delta_lam.reshape((self.qp.N-1, self.qp.nh))
        delta_dict['delta_lam0'] = delta_lam0
        delta_dict['delta_lam'] = delta_lam

        index_after_lam = index_after_pi + len(self.qp.d0)+(self.qp.N-1)*self.qp.nh
        delta_s0 = delta_z[index_after_lam:index_after_lam + len(self.qp.d0)]
        delta_s = delta_z[index_after_lam + len(self.qp.d0):index_after_lam + len(self.qp.d0)+(self.qp.N-1)*self.qp.nh]
        delta_s = delta_s.reshape((self.qp.N-1, self.qp.nh))
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
            cost += 0.5 * self.solution.x[i].T @ self.qp.Q_list[i] @ self.solution.x[i] + 0.5 * self.solution.u[i].T @ self.qp.R_list[i] @ self.solution.u[i] \
            + self.solution.x[i].T @ self.qp.S_list[i].T @ self.solution.u[i] + self.qp.q_list[i].T @ self.solution.x[i] + self.qp.r_list[i].T @ self.solution.u[i]
        # terminal cost
        x_N = self.solution.x[self.qp.N]
        Q_e = self.qp.Q_e
        q_e = self.qp.q_e
        cost += 0.5 * x_N.T @ Q_e @ x_N + q_e.T @ x_N
        return cost

    def print_statistics(self):
        # print residual table
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