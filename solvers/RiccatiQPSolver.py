from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs
from solvers.Regularization_config import Regularizations_config
from solvers.Riccati_IPM import Riccati_IPM
from solvers.Condensed_IPM import Condensed_IPM
import numpy as np
import copy

class QPSolver:

    def __init__(self, qp:BaseQP, reg:str='NO_REGULARIZE', type:str='Riccati'):
        self.qp = qp
        self.status = -1
        self.centering_param = 1.0
        self.max_iter = 200
        self.delta_aff_dict = None
        self.delta_dict = None
        self.res_max = {
            'res_stat' : 1e-6,
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

        self.ineq_num = 2*(self.qp.nb_0 + self.qp.nh_0 +
                           (self.qp.nh + self.qp.nb) * (self.qp.N - 1) +
                           self.qp.nb_e + self.qp.nh_e)
        if type == 'Riccati':
            self.solver = Riccati_IPM(qp,
                                    Regularizations_config(method=reg, delta=1e-4),
                                    ineq_num=self.ineq_num)
        elif type == 'Condensed':
            self.solver = Condensed_IPM(qp,
                                    Regularizations_config(method=reg, delta=1e-4),
                                    ineq_num=self.ineq_num)
        else:
            raise ValueError(f"Invalid solver type {type}. Valid types are 'Riccati' and 'Condensed'.")

    def _initilize_solution(self, x0:np.ndarray):
        if x0 is None:
            x0 = np.zeros((self.qp.nx,))
        self.solution.x = [x0 for i in range(self.qp.N+1)]
        self.solution.u = [np.zeros((self.qp.nu,)) for _ in range(self.qp.N)]
        self.solution.pi = [np.zeros((self.qp.nx,)) for _ in range(self.qp.N+1)]
        self._initailize_lam_and_s()

    def _initailize_lam_and_s(self):
        self.solution.s = []
        self.solution.lam = []
        threshold = 1e-1
        for k in range(self.qp.N):
            nb = self.qp.nb_0 if k == 0 else (self.qp.nb_e if k == self.qp.N else self.qp.nb)
            nh = self.qp.nh_0 if k == 0 else (self.qp.nh_e if k == self.qp.N else self.qp.nh)
            end_lb = nb + nh

            Jbx_k = self.qp.Jbx[k]
            Jbu_k = self.qp.Jbu[k]
            d_k = self.qp.d[k]
            C_k = self.qp.C[k][2*nb:]
            D_k = self.qp.D[k][2*nb:]

            bounded_x = Jbx_k @ self.solution.x[k] if Jbx_k.shape[0] > 0 else np.array([])
            bounded_u = Jbu_k @ self.solution.u[k] if Jbu_k.shape[0] > 0 else np.array([])
            bounded_ux = np.concatenate((bounded_u, bounded_x))

            d_lb_box = d_k[:nb]
            d_ub_box = d_k[end_lb:end_lb+nb]

            s_lb_box = bounded_ux + d_lb_box
            s_ub_box = d_ub_box - bounded_ux

            mask_lb = s_lb_box < threshold
            mask_ub = s_ub_box < threshold
            mask_both = mask_lb & mask_ub

            if np.any(mask_both):
                bounded_ux[mask_both] = 0.5 * (d_ub_box[mask_both] - d_lb_box[mask_both])
                s_lb_box[mask_both] = threshold
                s_ub_box[mask_both] = threshold

            mask_only_lb = mask_lb & ~mask_both
            if np.any(mask_only_lb):
                s_lb_box[mask_only_lb] = threshold
                bounded_ux[mask_only_lb] = -d_lb_box[mask_only_lb] + threshold

            mask_only_ub = mask_ub & ~mask_both
            if np.any(mask_only_ub):
                s_ub_box[mask_only_ub] = threshold
                bounded_ux[mask_only_ub] = d_ub_box[mask_only_ub] - threshold

            self.solution.u[k] = bounded_ux[:self.qp.nu] if bounded_ux[:self.qp.nu].size > 0 else self.solution.u[k]
            self.solution.x[k] = bounded_ux[self.qp.nu:] if bounded_ux[self.qp.nu:].size > 0 else self.solution.x[k]

            d_lb_gen = d_k[nb:end_lb]
            d_ub_gen = d_k[end_lb+nb:end_lb+nb+nh]

            y_gen = C_k @ self.solution.x[k] + D_k @ self.solution.u[k]

            s_lb_gen_raw = y_gen - d_lb_gen
            s_ub_gen_raw = d_ub_gen - y_gen

            s_lb_gen = np.maximum(s_lb_gen_raw, threshold)
            s_ub_gen = np.maximum(s_ub_gen_raw, threshold)

            s_lb_full = np.concatenate((s_lb_box, s_lb_gen))
            s_ub_full = np.concatenate((s_ub_box, s_ub_gen))
            s_k_full = np.concatenate((s_lb_full, s_ub_full))

            self.solution.s.append(s_k_full)
            self.solution.lam.append(self.rhs.duality_measure / s_k_full)

    def solve(self, x0:np.ndarray=None, max_iter:int=200)-> iterate_sol:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''

        self._initilize_solution(x0)
        self.max_iter = max_iter

        while True:

            prev_sol = copy.deepcopy(self.solution)

            ### Mehrotra predictor-corrector step
            # predictor step
            self.delta_aff_dict = self._compute_step(prev_sol, step_type='predictor')

            # print delta_aff for debug
            for k in range(self.qp.N+1):
                # ux_k = np.concatenate((self.delta_aff_dict['delta_u'][k] if k < self.qp.N else np.array([]),
                #                        self.delta_aff_dict['delta_x'][k]))
                # print(f"Stage {k}:ux = {ux_k}")
                if k < self.qp.N:
                   s_k = self.delta_aff_dict['delta_s'][k]
                   print(f"Stage {k}: delta_s = {s_k}")

            mu_aff = self._compute_duality(prev_sol, self.delta_aff_dict)
            
            # print for debug
            print(f"Predictor step: mu_aff = {mu_aff}")

            # check convergence based on predictor step
            if self._convergence_check() or (self.statistics_dict['iter_count'] == self.max_iter):
                self._quit()
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

            return self.solution

    def _compute_step(self, prev_sol:iterate_sol, step_type:str)-> iterate_sol:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        self.rhs = self.solver.update_rhs(centering_param=self.centering_param, rhs=self.rhs, statistics_dict=self.statistics_dict,
                                          prev_sol=prev_sol, delta_aff_dict=self.delta_aff_dict, step_type=step_type)
        feedback = self.solver.backward_sweep(self.rhs, prev_sol=prev_sol)
        delta_dict = self.solver.forward_sweep(self.rhs, prev_sol, feedback)
        return delta_dict

    def _compute_duality(self, prev_sol:iterate_sol, delta_dict:dict, prob:bool=False)-> float:
        tol = 1e-16
        alpha_pri_max, alpha_dual_max, alpha_max = self.fraction_to_boundary(prev_sol, delta_dict, boundary=prob)
        # print for debug
        print(f"alpha_pri_max = {alpha_pri_max}, alpha_dual_max = {alpha_dual_max}, alpha_max = {alpha_max}")
        s_aff = np.maximum(np.concatenate(prev_sol.s) + alpha_pri_max * np.concatenate(delta_dict['delta_s']), tol)
        lam_aff = np.maximum(np.concatenate(prev_sol.lam) + alpha_dual_max * np.concatenate(delta_dict['delta_lam']), tol)
        mu = (s_aff.flatten() @ lam_aff.flatten())/self.ineq_num
        return mu

    def _compute_step_size_and_update_solution(self, prev_sol:iterate_sol):
        # compute step size
        alpha_pri, alpha_dual, alpha = self.fraction_to_boundary(prev_sol, self.delta_dict, boundary=True)
        # update solution
        x_sol = [prev_sol.x[i] + alpha_pri * self.delta_dict['delta_x'][i] for i in range(self.qp.N+1)]
        u_sol = [prev_sol.u[i] + alpha_pri * self.delta_dict['delta_u'][i] for i in range(self.qp.N)]
        pi_sol = [prev_sol.pi[i] + alpha_dual * self.delta_dict['delta_pi'][i] for i in range(self.qp.N+1)]
        lam_sol = [np.maximum(prev_sol.lam[i] + alpha_dual * self.delta_dict['delta_lam'][i], 1e-16) for i in range(self.qp.N)]
        s_sol = [np.maximum(prev_sol.s[i] + alpha_pri * self.delta_dict['delta_s'][i], 1e-16) for i in range(self.qp.N)]
        self.solution = iterate_sol(x=x_sol, u=u_sol, pi=pi_sol, lam=lam_sol, s=s_sol)

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
            alpha_pri_max = 1.0
        try:
            alpha_dual_max = np.min(frac_dual)
        except ValueError:
            alpha_dual_max = 1.0

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

    def _quit(self):
        res = max(self.statistics_dict['res_stat_list'][-1],
                  self.statistics_dict['res_eq_list'][-1],
                  self.statistics_dict['res_ineq_list'][-1],
                  self.statistics_dict['res_comp_list'][-1])
        if self.statistics_dict['iter_count'] >= self.max_iter:
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

    def get(self, stage:int, field:str):

        if field == 'x':
            return self.solution.x[stage]
        elif field == 'u':
            return self.solution.u[stage]
        elif field == 'pi':
            return self.solution.pi[stage]
        elif field == 'lam':
            return self.solution.lam[stage]
        elif field == 's':
            return self.solution.s[stage]
        else:
            raise ValueError(f"Invalid field {field}. Valid fields are 'x', 'u', 'pi', 'lam', 's'.")

    def print_statistics(self):
        print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tduality_measure\talpha_pri\talpha_dual')
        for i in range(self.statistics_dict['iter_count']+1):
            res_stat_norm = self.statistics_dict['res_stat_list'][i]
            res_eq_norm = self.statistics_dict['res_eq_list'][i]
            res_ineq_norm = self.statistics_dict['res_ineq_list'][i]
            res_comp_norm = self.statistics_dict['res_comp_list'][i]
            duality_measure = self.statistics_dict['duality_measure_list'][i]
            alpha_pri = self.statistics_dict['alpha_pri_list'][i] if i < len(self.statistics_dict['alpha_pri_list']) else 0.0
            alpha_dual = self.statistics_dict['alpha_dual_list'][i] if i < len(self.statistics_dict['alpha_dual_list']) else 0.0
            print(f"{i}\t{res_stat_norm:.2e}\t{res_eq_norm:.2e}\t{res_ineq_norm:.2e}\t{res_comp_norm:.2e}\t{duality_measure:.2e}\t{alpha_pri:.2e}\t{alpha_dual:.2e}")