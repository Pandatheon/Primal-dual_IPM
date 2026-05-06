from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs
from solvers.Regularization_config import Regularizations_config
from solvers.Riccati_IPM import Riccati_IPM
import numpy as np
import copy

class QPSolver:

    def __init__(self, qp:BaseQP, reg:str='NO_REGULARIZE'):
        self.qp = qp
        self.status = -1
        self.centering_param = 1.0
        self.delta_aff_dict = None
        self.delta_dict = None
        self.res_max = {
            'res_stat' : 1e-8,
            'res_eq' : 1e-8,
            'res_ineq' : 1e-8,
            'res_comp' : 1e-8,
        }
        # statistics for analysis
        self.statistics_dict = {
            'res_stat_list' : [],
            'res_eq_list' : [],
            'res_ineq_list' : [],
            'res_comp_list' : [],
            'alpha_pri_list' : [],
            'alpha_dual_list' : [],
            'alpha_list' : [],
            'duality_measure_list' : [],
            'iter_count' : 0,
        }

        # sol and rhs initialization
        self.solution = iterate_sol(
            x =[],
            u = [],
            pi = [],
            lam = [],
            s = [],
        )
        self.rhs = iterate_rhs(
            # lazy intialization, will be updated in the first call of _update_rhs
            rg = [],
            rb = [],
            rd = [],
            rm = [],
            # tunable parameter, will be updated in each predictor step
            duality_measure = 10.0,
        )

        self.solver = Riccati_IPM(qp, Regularizations_config(method=reg, delta=1e-4))
        self.ineq_num = self.qp.nh * self.qp.N + 2*self.qp.nx

    def _initilize_solution(self, x0:np.ndarray):
        self.solution.x = [x0 for i in range(self.qp.N+1)]
        self.solution.u = [np.zeros((self.qp.nu,)) for i in range(self.qp.N)]
        init_s = self.qp.d[0] - self.qp.C[0] @ self.solution.x[0] - self.qp.D[0] @ self.solution.u[0]
        self.solution.s = [init_s for _ in range(self.qp.N)]
        self.solution.lam = [self.rhs.duality_measure / self.solution.s[i] for i in range(self.qp.N)]
        self.solution.pi = [np.zeros((self.qp.nx,)) for i in range(self.qp.N)]

        init_d0 =  np.hstack(( -(x0-1e-1)*np.ones((self.qp.nx,)), (x0+1e-1)*np.ones((self.qp.nx,)), self.qp.d[0]))
        C0 = np.vstack((-np.eye(self.qp.nx), np.eye(self.qp.nx), self.qp.C[0]))
        D0 = np.vstack((np.zeros((self.qp.nx, self.qp.nu)), np.zeros((self.qp.nx, self.qp.nu)), self.qp.D[0]))
        temp = C0 @ self.solution.x[0] - D0 @ self.solution.u[0]
        init_s0 = init_d0 - temp
        d0 = np.hstack(( -(x0)*np.ones((self.qp.nx,)), (x0)*np.ones((self.qp.nx,)), self.qp.d[0]))

        self.qp.C[0] = C0
        self.qp.D[0] = D0
        self.qp.d[0] = d0

        self.solution.s[0] = init_s0
        self.solution.lam[0] = self.rhs.duality_measure / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=200, reg:str='NO_REGULARIZE')-> iterate_sol:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        assert x0 is not None, "Please provide a non-zero initial state x0 to solve the QP."

        self._initilize_solution(x0)

        for i in range(max_iter+1):

            prev_sol = copy.deepcopy(self.solution)

            ### Mehrotra predictor-corrector step
            # predictor step
            self.delta_aff_dict = self._compute_step(prev_sol, step_type='predictor')
            mu_aff = self._compute_duality(prev_sol, self.delta_aff_dict)

            # check convergence based on predictor step
            if self._convergence_check():
                break

            # compute centering parameter sigma
            self.centering_param = (mu_aff/self.rhs.duality_measure)**3

            # predictor-corrector step
            self.delta_dict = self._compute_step(prev_sol, step_type='predictor-corrector')
            mu_pred = self._compute_duality(prev_sol, self.delta_dict, prob=True)

            ### Conditonal Mehrotra step
            if mu_pred > 2*mu_aff:
                self.delta_dict = self._compute_step(prev_sol, step_type='centering')

            self._compute_step_size_and_update_solution(prev_sol)

            self.statistics_dict['iter_count'] += 1
        self._quit(max_iter)
        return self.solution

    def _compute_step(self, prev_sol:iterate_sol, step_type:str)-> iterate_sol:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        self.rhs = self.solver.update_rhs(centering_param=self.centering_param, rhs=self.rhs, statistics_dict=self.statistics_dict, 
                                          prev_sol=prev_sol, delta_aff_dict=self.delta_aff_dict, step_type=step_type)
        feedback = self.solver.backward_sweep(self.rhs, prev_sol=prev_sol)
        delta_dict = self.solver.foward_sweep(self.rhs, prev_sol, feedback)
        return delta_dict

    def _compute_duality(self, prev_sol:iterate_sol, delta_dict:dict, prob:bool=False)-> float:
        tol = 1e-16 if prob else 1e-20
        alpha_pri_max, alpha_dual_max, alpha_max = self.fraction_to_boundary(prev_sol, delta_dict, boundary=prob)
        s_aff = np.maximum(np.concatenate(prev_sol.s) + alpha_pri_max * np.concatenate(delta_dict['delta_s']), tol)
        lam_aff = np.maximum(np.concatenate(prev_sol.lam) + alpha_dual_max * np.concatenate(delta_dict['delta_lam']), tol)
        mu = (s_aff.flatten() @ lam_aff.flatten())/self.ineq_num
        return mu

    def _compute_step_size_and_update_solution(self, prev_sol:iterate_sol):
        # compute step size
        alpha_pri, alpha_dual, alpha = self.fraction_to_boundary(prev_sol, self.delta_dict, boundary=True)
        # update solution
        x_delta = [prev_sol.x[i] + alpha_pri * self.delta_dict['delta_x'][i] for i in range(self.qp.N+1)]
        u_delta = [prev_sol.u[i] + alpha_pri * self.delta_dict['delta_u'][i] for i in range(self.qp.N)]
        pi_delta = [prev_sol.pi[i] + alpha_dual * self.delta_dict['delta_pi'][i] for i in range(self.qp.N)]
        lam_delta = [np.maximum(prev_sol.lam[i] + alpha_dual * self.delta_dict['delta_lam'][i], 1e-16) for i in range(self.qp.N)]
        s_delta = [np.maximum(prev_sol.s[i] + alpha_pri * self.delta_dict['delta_s'][i], 1e-16) for i in range(self.qp.N)]
        self.solution = iterate_sol(x_delta, u_delta, pi_delta, lam_delta, s_delta)

    def fraction_to_boundary(self, prev_sol: iterate_sol, delta_dict, boundary:bool=True) -> iterate_sol:
        s_i = np.concatenate(prev_sol.s)
        lam_i = np.concatenate(prev_sol.lam)
        delta_s = np.concatenate(delta_dict['delta_s'])
        delta_lam =  np.concatenate(delta_dict['delta_lam'])

        mask_lam = delta_lam < 0
        mask_s = delta_s < 0

        frac_pri = s_i[mask_s] / (-delta_s[mask_s])
        frac_dual = lam_i[mask_lam] / (-delta_lam[mask_lam])
        # take the minimum positive step size
        try:
            alpha_pri_max = np.min(frac_pri)
        except ValueError:
            alpha_pri_max = 1e-12
        try:
            alpha_dual_max = np.min(frac_dual)
        except ValueError:
            alpha_dual_max = 1e-12

        if boundary:
            # take 0.99 of the maximum step size
            alpha_pri = min(0.99 * alpha_pri_max, 1.0)
            alpha_dual = min(0.99 * alpha_dual_max, 1.0)
            alpha = min(alpha_pri, alpha_dual)

            self.statistics_dict['alpha_pri_list'].append(alpha_pri)
            self.statistics_dict['alpha_dual_list'].append(alpha_dual)
            self.statistics_dict['alpha_list'].append(alpha)
            return alpha_pri, alpha_dual, alpha
        else:
            return alpha_pri_max, alpha_dual_max, min(alpha_pri_max, alpha_dual_max)

    def _convergence_check(self, relative_tol:float=0)-> bool:
        '''Check convergence based on KKT residuals.'''
        return ((self.statistics_dict['res_stat_list'][-1] < self.res_max['res_stat'] + relative_tol*self.res_max['res_stat']) and
                (self.statistics_dict['res_eq_list'][-1] < self.res_max['res_eq'] + relative_tol*self.res_max['res_eq']) and
                (self.statistics_dict['res_ineq_list'][-1] < self.res_max['res_ineq'] + relative_tol*self.res_max['res_ineq']) and
                (self.statistics_dict['res_comp_list'][-1] < self.res_max['res_comp'] + relative_tol*self.res_max['res_comp']))

    def _quit(self, max_iter):
        res = max(self.statistics_dict['res_stat_list'][-1], 
                  self.statistics_dict['res_eq_list'][-1], 
                  self.statistics_dict['res_ineq_list'][-1], 
                  self.statistics_dict['res_comp_list'][-1])
        if self.statistics_dict['iter_count'] >= max_iter:
            print(f"\033[31mNot Converged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
        else:
            print(f"\033[32mConverged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
            self.status = 0

    def get_cost(self):
        cost = 0.0
        for i in range(self.qp.N):
            x_i = self.solution.x[i]
            u_i = self.solution.u[i]
            cost += 0.5 * x_i.T @ self.qp.Q[i] @ x_i + 0.5 * u_i.T @ self.qp.R[i] @ u_i + x_i.T @ self.qp.S[i].T @ u_i + self.qp.q[i].T @ x_i + self.qp.r[i].T @ u_i
        # terminal cost
        x_N = self.solution.x[-1]
        cost += 0.5 * x_N.T @ self.qp.Q_e @ x_N + self.qp.q_e.T @ x_N
        return cost

    def print_statistics(self):
        print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tduality_measure\talpha_pri\talpha_dual')
        for i in range(self.statistics_dict['iter_count']):
            res_stat_norm = self.statistics_dict['res_stat_list'][i]
            res_eq_norm = self.statistics_dict['res_eq_list'][i]
            res_ineq_norm = self.statistics_dict['res_ineq_list'][i]
            res_comp_norm = self.statistics_dict['res_comp_list'][i]
            duality_measure = self.statistics_dict['duality_measure_list'][i]
            alpha_pri = self.statistics_dict['alpha_pri_list'][i] if i < len(self.statistics_dict['alpha_pri_list']) else 0.0
            alpha_dual = self.statistics_dict['alpha_dual_list'][i] if i < len(self.statistics_dict['alpha_dual_list']) else 0.0
            print(f"{i}\t{res_stat_norm:.2e}\t{res_eq_norm:.2e}\t{res_ineq_norm:.2e}\t{res_comp_norm:.2e}\t{duality_measure:.2e}\t{alpha_pri:.2e}\t{alpha_dual:.2e}")