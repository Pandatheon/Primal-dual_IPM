from typing import Tuple
from dataclasses import dataclass
from QP_template.OCP_QP import OCP_QP_varying, QP_stage, QP_Terminal, iterate
from solvers.Riccati_recursion import Feedback, riccati_recursion_with_stage_regularization
from Regularizations import Regularizations_config
from solvers.Regularization import regularize_QP
from solvers.regularization_recorder import Recorder
from solvers.Riccati_IPM import update_rhs, backward_sweep, foward_sweep, fraction_to_boundary
import numpy as np
import copy

class QPSolver:

    def __init__(self, qp:OCP_QP_varying):
        self.qp = qp
        self.status = -1
        self.feedback = None
        self.centering_param = 1.0
        self.delta_aff_dict = None
        self.delta_dict = None
        self.res_max = {
            'res_stat' : 1e-9,
            'res_eq' : 1e-9,
            'res_ineq' : 1e-9,
            'res_comp' : 1e-9,
        }
        # rhs of KKT system, will be updated in each iteration
        self.rhs_dict = {
            # lazy intialization, will be updated in the first call of _update_rhs
            'rg' : np.ones((qp.N+1, qp.nx+qp.nu)),
            'rg_': np.ones((qp.N+1, qp.nx+qp.nu)),
            'rb' : np.ones((qp.N, qp.nx)),
            'rd0' : np.ones((qp.nh0, )),
            'rd' : np.ones((qp.N-1, qp.nh)),
            'rm0' : np.ones((qp.nh0, )),
            'rm' : np.ones((qp.N-1, qp.nh)),
            # tunable parameter, will be updated in each predictor step
            'duality_measure' : 10.0,
        }
        # To check convergence
        self.res_dict = {
            'res_stat' : None,
            'res_eq' : None,
            'res_ineq' : None,
            'res_comp' : None,
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
        self.solution = iterate(
            x = np.zeros((qp.N + 1, qp.nx)),
            u =  np.zeros((qp.N, qp.nu)),
            pi = np.zeros((qp.N, self.qp.nx)),
            lam0= np.zeros((qp.nh0, )),
            lam = np.zeros((qp.N-1, self.qp.nh)),
            s0 = np.zeros((self.qp.nh0, )),
            s = np.zeros((qp.N-1, self.qp.nh)),
        )
        self.ineq_num = self.solution.lam0.size + self.solution.lam.size
        self.reg_config = Regularizations_config()
        #debug
        self.recorder = Recorder(self.qp.N)
        self.curvature_list = []

    def _initilize_solution(self, x0:np.ndarray)-> iterate:
        self.solution.x = np.vstack([x0 for i in range(self.qp.N+1)])
        self.solution.u = np.zeros((self.qp.N, self.qp.nu))
        init_s = self.qp.stages[1].d - self.qp.stages[1].C @ self.solution.x[1] - self.qp.stages[1].D @ self.solution.u[1]
        self.solution.s = np.vstack([init_s for _ in range(self.qp.N-1)])
        self.solution.lam = self.rhs_dict['duality_measure'] / self.solution.s
        self.solution.pi = np.zeros((self.qp.N, self.qp.nx))

        init_d0 =  np.hstack(( -(x0-1e-1)*np.ones((self.qp.nx,)), (x0+1e-1)*np.ones((self.qp.nx,)), self.qp.stages[0].d))
        init_s0 = init_d0 -  self.qp.stages[0].C @ self.solution.x[0] - self.qp.stages[0].D @ self.solution.u[0]
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
            C = self.qp.stages[0].C,
            D = self.qp.stages[0].D,
            d = d0,
        )
        self.solution.s0 = init_s0
        self.solution.lam0 = self.rhs_dict['duality_measure'] / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=200, reg:str='NO_REGULARIZE')-> iterate:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        assert x0 is not None, "Please provide a non-zero initial state x0 to solve the QP."

        self._initilize_solution(x0)

        for i in range(max_iter+1):

            prev_sol = copy.deepcopy(self.solution)

            ### Mehrotra predictor-corrector step
            # predictor step
            self.delta_aff_dict = self._compute_step(prev_sol, step_type='predictor', reg=reg)
            mu_aff = self._compute_duality(prev_sol, self.delta_aff_dict)

            # check convergence based on predictor step
            self._record_res()
            if self._convergence_check():
                break

            # compute centering parameter sigma
            self.centering_param = (mu_aff/self.rhs_dict['duality_measure'])**3
            current_reg_count = self.recorder.reg_count

            # predictor-corrector step
            self.delta_dict = self._compute_step(prev_sol, step_type='predictor-corrector', reg=reg)
            mu_pred = self._compute_duality(prev_sol, self.delta_dict)

            ### Conditonal Mehrotra step
            if mu_pred > 2*mu_aff:
                self.recorder.reg_count = current_reg_count
                self.delta_dict = self._compute_step(prev_sol, step_type='centering', reg=reg)

            self._compute_step_size_and_update_solution(prev_sol)

            self.statistics_dict['iter_count'] += 1
        self._quit(max_iter)
        return self.solution

    def _quit(self, max_iter):
        res = max(self.res_dict['res_stat'], self.res_dict['res_eq'], self.res_dict['res_ineq'], self.res_dict['res_comp'])
        if self.statistics_dict['iter_count'] >= max_iter:
            print(f"\033[31mNot Converged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
        else:
            print(f"\033[32mConverged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
            self.status = 0

        self.recorder.finalize()

    def _update_res_dict(self):
        self.res_dict['res_stat'] = np.linalg.norm(self.rhs_dict['rg'].flatten(), ord=np.inf)
        self.res_dict['res_eq'] = np.linalg.norm(self.rhs_dict['rb'].flatten(), ord=np.inf)
        self.res_dict['res_ineq'] = np.linalg.norm(np.hstack((self.rhs_dict['rd0'],self.rhs_dict['rd'].flatten())), ord=np.inf)
        self.res_dict['res_comp'] = np.linalg.norm(np.hstack((self.rhs_dict['rm0'],self.rhs_dict['rm'].flatten())), ord=np.inf)

    def _convergence_check(self, relative_tol:float=0)-> bool:
        '''Check convergence based on KKT residuals.'''
        return ((self.res_dict['res_stat'] < self.res_max['res_stat'] + relative_tol*self.res_max['res_stat']) and
                (self.res_dict['res_eq'] < self.res_max['res_eq'] + relative_tol*self.res_max['res_eq']) and
                (self.res_dict['res_ineq'] < self.res_max['res_ineq'] + relative_tol*self.res_max['res_ineq']) and
                (self.res_dict['res_comp'] < self.res_max['res_comp'] + relative_tol*self.res_max['res_comp']))

    def _record_res(self):
        '''Record the KKT residuals from predictor step.'''
        self._update_res_dict()
        self.statistics_dict['res_stat_list'].append(self.res_dict['res_stat'])
        self.statistics_dict['res_eq_list'].append(self.res_dict['res_eq'])
        self.statistics_dict['res_ineq_list'].append(self.res_dict['res_ineq'])
        self.statistics_dict['res_comp_list'].append(self.res_dict['res_comp'])

    def _compute_step(self, prev_sol:iterate, step_type:str , reg:str='NO_REGULARIZE')-> iterate:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        self.rhs_dict = update_rhs(qp=self.qp, centering_param=self.centering_param, rhs_dict=self.rhs_dict, statistics_dict=self.statistics_dict, 
                                   ineq_num=self.ineq_num, prev_sol=prev_sol, delta_aff_dict=self.delta_aff_dict, step_type=step_type)
        feedback = backward_sweep(self.qp, self.rhs_dict, self.recorder, 
                                  self.reg_config, prev_sol=prev_sol, reg=reg)
        delta_dict = foward_sweep(self.qp, self.rhs_dict,
                                  prev_sol, feedback)
        return delta_dict

    def _compute_duality(self, prev_sol:iterate, delta_dict:dict)-> float:
        alpha_pri_max, alpha_dual_max, alpha_max = fraction_to_boundary(prev_sol, delta_dict, self.statistics_dict, boundary=False)
        s0_aff = np.maximum(prev_sol.s0 + alpha_pri_max * self.delta_aff_dict['delta_s0'], 1e-20)
        s_aff = np.maximum(prev_sol.s + alpha_pri_max * self.delta_aff_dict['delta_s'], 1e-20)
        lam0_aff = np.maximum(prev_sol.lam0 + alpha_dual_max * self.delta_aff_dict['delta_lam0'], 1e-20)
        lam_aff = np.maximum(prev_sol.lam + alpha_dual_max * self.delta_aff_dict['delta_lam'], 1e-20)
        mu = ((s0_aff @ lam0_aff) + (s_aff.flatten() @ lam_aff.flatten()))/self.ineq_num
        return mu

    def _compute_step_size_and_update_solution(self, prev_sol:iterate):
        # compute step size
        alpha_pri, alpha_dual, alpha = fraction_to_boundary(prev_sol, self.delta_dict, self.statistics_dict, boundary=True)
        # update solution
        x_delta = prev_sol.x + alpha_pri * self.delta_dict['delta_x']
        u_delta = prev_sol.u + alpha_pri * self.delta_dict['delta_u']
        pi_delta = prev_sol.pi + alpha_dual * self.delta_dict['delta_pi']
        lam0_delta = np.maximum(prev_sol.lam0 + alpha_dual * self.delta_dict['delta_lam0'], 1e-16)
        lam_delta = np.maximum(prev_sol.lam + alpha_dual * self.delta_dict['delta_lam'], 1e-16)
        s0_delta = np.maximum(prev_sol.s0 + alpha_pri * self.delta_dict['delta_s0'], 1e-16)
        s_delta = np.maximum(prev_sol.s + alpha_pri * self.delta_dict['delta_s'], 1e-16)
        self.solution = iterate(x_delta, u_delta, pi_delta, lam0_delta, lam_delta, s0_delta, s_delta)

    def get_cost(self):
        cost = 0.0
        for i in range(len(self.qp.stages)):
            x_i = self.solution.x[i]
            u_i = self.solution.u[i]
            cost += 0.5 * x_i.T @ self.qp.stages[i].Q @ x_i + 0.5 * u_i.T @ self.qp.stages[i].R @ u_i + x_i.T @ self.qp.stages[i].S.T @ u_i + self.qp.stages[i].q.T @ x_i + self.qp.stages[i].r.T @ u_i
        # terminal cost
        x_N = self.solution.x[-1]
        cost += 0.5 * x_N.T @ self.qp.terminal.Q_e @ x_N + self.qp.terminal.q_e.T @ x_N
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