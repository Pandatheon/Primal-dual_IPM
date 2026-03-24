from typing import Tuple
from dataclasses import dataclass
from QP_template.OCP_QP import OCP_QP_varying, QP_stage, QP_Terminal, iterate
from solvers.Riccati_recursion import Feedback, riccati_recursion_with_stage_regularization
from Regularizations import Regularizations_config
from solvers.Regularization import regularize_QP
from solvers.regularization_recorder import Recorder
import numpy as np
import copy

class QPSolver:

    def __init__(self, qp:OCP_QP_varying):
        self.qp = qp
        self.status = -1
        self.feedback = None
        self.duality_measure = 10.0
        self.centering_param = 1.0
        self.delta_aff_dict = None
        self.delta_dict = None
        self.res_max = {
            'res_stat' : 1e-9,
            'res_eq' : 1e-9,
            'res_ineq' : 1e-9,
            'res_comp' : 1e-9,
        }
        self.rhs_dict = {
            'rg' : np.zeros((qp.N+1, qp.nx+qp.nu)),
            'rg_': np.zeros((qp.N+1, qp.nx+qp.nu)),
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
        self.solution.lam = self.duality_measure / self.solution.s
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
        self.solution.lam0 = self.duality_measure / init_s0

    def solve(self, x0:np.ndarray=None, max_iter:int=200, reg:str='NO_REGULARIZE')-> iterate:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''
        assert x0 is not None, "Please provide a non-zero initial state x0 to solve the QP."
        self._initilize_solution(x0)
        iter_res_dict = self._record_res()
        for i in range(max_iter+1):
            prev_sol = copy.deepcopy(self.solution)

            if self._convergence_check(iter_res_dict):
                self._quit(iter_res_dict, reg, max_iter)
                return self.solution

            ### Mehrotra predictor-corrector step
            # predictor step
            self.delta_aff_dict = self._compute_step(prev_sol, step_type='predictor', reg=reg)
            mu_aff = self._compute_duality(prev_sol, self.delta_aff_dict)

            # compute centering parameter sigma
            self.centering_param = (mu_aff/self.duality_measure)**3
            current_reg_count = self.recorder.reg_count

            # predictor-corrector step
            self.delta_dict = self._compute_step(prev_sol, step_type='predictor-corrector', reg=reg)
            mu_pred = self._compute_duality(prev_sol, self.delta_dict)

            ### Conditonal Mehrotra step
            if mu_pred > 2*mu_aff:
                self.recorder.reg_count = current_reg_count
                self.delta_dict = self._compute_step(prev_sol, step_type='centering', reg=reg)

            self._compute_step_size_and_update_solution(prev_sol)

            iter_res_dict = self._record_res()
            self.statistics_dict['iter_count'] += 1
        self._quit(iter_res_dict, reg, max_iter)
        return self.solution

    def _quit(self, iter_res_dict, reg, max_iter):
        self._update_rhs(self.solution)
        res = max(iter_res_dict['res_stat'], iter_res_dict['res_eq'], iter_res_dict['res_ineq'], iter_res_dict['res_comp'])
        if self.statistics_dict['iter_count'] >= max_iter:
            print(f"\033[31mNot Converged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
        else:
            print(f"\033[32mConverged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.2e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.2e}")
            self.status = 0

        self.recorder.finalize()

    def _convergence_check(self, iter_res_dict, relative_tol:float=0)-> bool:
        '''Check convergence based on KKT residuals.'''
        return ((iter_res_dict['res_stat'] < self.res_max['res_stat'] + relative_tol*self.res_max['res_stat']) and
                (iter_res_dict['res_eq'] < self.res_max['res_eq'] + relative_tol*self.res_max['res_eq']) and
                (iter_res_dict['res_ineq'] < self.res_max['res_ineq'] + relative_tol*self.res_max['res_ineq']) and
                (iter_res_dict['res_comp'] < self.res_max['res_comp'] + relative_tol*self.res_max['res_comp']))

    def _record_res(self):
        '''Record the initial KKT residuals.'''
        self._update_rhs(self.solution, record=True)
        iter_res_dict = {
            'res_stat' : np.linalg.norm(self.rhs_dict['rg'].flatten(), ord=np.inf),
            'res_eq' : np.linalg.norm(self.rhs_dict['rb'].flatten(), ord=np.inf),
            'res_ineq' : np.linalg.norm(np.hstack((self.rhs_dict['rd0'],self.rhs_dict['rd'].flatten())), ord=np.inf),
            'res_comp' : np.linalg.norm(np.hstack((self.rhs_dict['rm0'],self.rhs_dict['rm'].flatten())), ord=np.inf),
        }
        self.statistics_dict['res_stat_list'].append(iter_res_dict['res_stat'])
        self.statistics_dict['res_eq_list'].append(iter_res_dict['res_eq'])
        self.statistics_dict['res_ineq_list'].append(iter_res_dict['res_ineq'])
        self.statistics_dict['res_comp_list'].append(iter_res_dict['res_comp'])
        return iter_res_dict

    def _compute_step(self, prev_sol:iterate, step_type:str , reg:str='NO_REGULARIZE')-> iterate:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        self._update_rhs(prev_sol, self.delta_aff_dict, step_type=step_type)
        feedback = self._backward_sweep(prev_sol, reg=reg)
        delta_dict = self._foward_sweep(prev_sol, feedback)
        return delta_dict

    def _compute_duality(self, prev_sol:iterate, delta_dict:dict)-> float:
        alpha_pri_max, alpha_dual_max, alpha_max = self._fraction_to_boundary(prev_sol, delta_dict, boundary=False)
        s0_aff = np.maximum(prev_sol.s0 + alpha_pri_max * self.delta_aff_dict['delta_s0'], 1e-20)
        s_aff = np.maximum(prev_sol.s + alpha_pri_max * self.delta_aff_dict['delta_s'], 1e-20)
        lam0_aff = np.maximum(prev_sol.lam0 + alpha_dual_max * self.delta_aff_dict['delta_lam0'], 1e-20)
        lam_aff = np.maximum(prev_sol.lam + alpha_dual_max * self.delta_aff_dict['delta_lam'], 1e-20)
        mu = ((s0_aff @ lam0_aff) + (s_aff.flatten() @ lam_aff.flatten()))/self.ineq_num
        return mu

    def _compute_step_size_and_update_solution(self, prev_sol:iterate):
        # compute step size
        alpha_pri, alpha_dual, alpha = self._fraction_to_boundary(prev_sol, self.delta_dict, boundary=True)
        # update solution
        x_delta = prev_sol.x + alpha_pri * self.delta_dict['delta_x']
        u_delta = prev_sol.u + alpha_pri * self.delta_dict['delta_u']
        pi_delta = prev_sol.pi + alpha_dual * self.delta_dict['delta_pi']
        lam0_delta = np.maximum(prev_sol.lam0 + alpha_dual * self.delta_dict['delta_lam0'], 1e-16)
        lam_delta = np.maximum(prev_sol.lam + alpha_dual * self.delta_dict['delta_lam'], 1e-16)
        s0_delta = np.maximum(prev_sol.s0 + alpha_pri * self.delta_dict['delta_s0'], 1e-16)
        s_delta = np.maximum(prev_sol.s + alpha_pri * self.delta_dict['delta_s'], 1e-16)
        self.solution = iterate(x_delta, u_delta, pi_delta, lam0_delta, lam_delta, s0_delta, s_delta)

    def _update_rhs(self, prev_sol:iterate, delta_aff_dict:dict = None, record:bool=False, step_type:str='predictor')-> np.ndarray:
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
        duality_measure = 0.0

        for i in range(len(self.qp.stages)):
            # stationarity residual
            if i == 0:
                res_stat[i][nx:] = self.qp.stages[0].R @ prev_sol.u[0] + self.qp.stages[0].r + self.qp.stages[0].B.T @ prev_sol.pi[0] + self.qp.stages[0].D.T @ prev_sol.lam0
            else:
                res_stat[i][:nx] = self.qp.stages[i].Q @ prev_sol.x[i] + self.qp.stages[i].S.T @ prev_sol.u[i] + self.qp.stages[i].q +  self.qp.stages[i].A.T @ prev_sol.pi[i] + self.qp.stages[i].C.T @ prev_sol.lam[i-1] - prev_sol.pi[i-1]
                res_stat[i][nx:] = self.qp.stages[i].R @ prev_sol.u[i] + self.qp.stages[i].S @ prev_sol.x[i] + self.qp.stages[i].r + self.qp.stages[i].B.T @ prev_sol.pi[i] + self.qp.stages[i].D.T @ prev_sol.lam[i-1]

            #equality residual
            res_eq[i] = self.qp.stages[i].A @ prev_sol.x[i] + self.qp.stages[i].B @ prev_sol.u[i] - prev_sol.x[i+1] - self.qp.stages[i].b

            #inequality residual
            if i == 0:
                res_ineq0 =  self.qp.stages[0].C @ prev_sol.x[0] + self.qp.stages[0].D @ prev_sol.u[0] + prev_sol.s0 - self.qp.stages[0].d
            else:
                res_ineq[i-1] = (self.qp.stages[i].C @ prev_sol.x[i] + self.qp.stages[i].D @ prev_sol.u[i] + prev_sol.s[i-1]) - self.qp.stages[i].d

            #complementarity residual
            if i == 0:
                if step_type == 'predictor':
                    res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten()
                    duality_measure += (prev_sol.lam0 @ prev_sol.s0)
                elif step_type == 'centering':
                    res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten() - self.centering_param * self.duality_measure * np.ones(len(prev_sol.lam0))
                    duality_measure += (prev_sol.lam0 @ prev_sol.s0)
                elif step_type == 'predictor-corrector':
                    res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten() + (delta_aff_dict['delta_lam0'] * delta_aff_dict['delta_s0']).flatten() - self.centering_param * self.duality_measure * np.ones(len(prev_sol.lam0))
            else:
                if step_type == 'predictor':
                    res_comp[i-1] = (np.diag(prev_sol.lam[i-1]) @ prev_sol.s[i-1]).flatten()
                    duality_measure += (prev_sol.lam[i-1] @ prev_sol.s[i-1])
                elif step_type == 'centering':
                    res_comp[i-1] = (prev_sol.lam[i-1] * prev_sol.s[i-1]).flatten() - self.centering_param * self.duality_measure * np.ones(len(prev_sol.lam[i-1]))
                    duality_measure += (prev_sol.lam[i-1] @ prev_sol.s[i-1])
                elif step_type == 'predictor-corrector':
                    res_comp[i-1] = (prev_sol.lam[i-1] * prev_sol.s[i-1]).flatten() + (delta_aff_dict['delta_lam'][i-1] * delta_aff_dict['delta_s'][i-1]) - self.centering_param * self.duality_measure * np.ones(len(prev_sol.lam[i-1]))

        # terminal residual
        res_stat[N][:nx] = self.qp.terminal.Q_e @ prev_sol.x[N] + self.qp.terminal.q_e - prev_sol.pi[N-1]

        self.rhs_dict['rg'] = res_stat
        self.rhs_dict['rb'] = res_eq
        self.rhs_dict['rd0'] = res_ineq0
        self.rhs_dict['rd'] = res_ineq
        self.rhs_dict['rm0'] = res_comp0
        self.rhs_dict['rm'] = res_comp
        if step_type == 'predictor' and not record:
            self.duality_measure = max(duality_measure / self.ineq_num, 1e-16)
            self.statistics_dict['duality_measure_list'].append(self.duality_measure)

    def _foward_sweep(self, prev_sol:iterate, feedback:Feedback)-> dict:
        delta_x = copy.deepcopy(prev_sol.x)
        delta_u = copy.deepcopy(prev_sol.u)
        delta_pi = copy.deepcopy(prev_sol.pi)
        delta_lam0 = copy.deepcopy(prev_sol.lam0)
        delta_lam = copy.deepcopy(prev_sol.lam)
        delta_s0 = copy.deepcopy(prev_sol.s0)
        delta_s = copy.deepcopy(prev_sol.s)

        # recover delta_x and delta_u, delta_pi first with modified KKT system
        # then recover delta_s and delta_lam with original KKT system
        delta_x[0] = 0
        for i in range(len(self.qp.stages)):
            # recover x and u
            rhs_b = self.rhs_dict['rb'][i]
            delta_u[i] = feedback.compute_control(i, delta_x[i])
            delta_x[i+1] = self.qp.stages[i].A@delta_x[i] + self.qp.stages[i].B@delta_u[i] + rhs_b

            # recover s
            if i == 0:
                rhs_d =  self.rhs_dict['rd0']
                delta_s0 = -(self.qp.stages[0].C @delta_x[0] + self.qp.stages[0].D @delta_u[0]) - rhs_d
            else:
                rhs_d = self.rhs_dict['rd'][i-1]
                delta_s[i-1] = -(self.qp.stages[i].C @delta_x[i] + self.qp.stages[i].D @delta_u[i]) - rhs_d
            # recover lam
            if i == 0:
                rhs_m = self.rhs_dict['rm0']
                rhs_m_modified = -rhs_m - (prev_sol.lam0 * delta_s0)
                delta_lam0 = (rhs_m_modified/prev_sol.s0).flatten()
            else:
                rhs_m = self.rhs_dict['rm'][i-1]
                rhs_m_modified = -rhs_m - (prev_sol.lam[i-1] * delta_s[i-1])
                delta_lam[i-1] = (rhs_m_modified/prev_sol.s[i-1]).flatten()
            # recover pi
            delta_pi[i] =  feedback.P[i+1] @ delta_x[i+1] + feedback.p[i+1]

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

    def _backward_sweep(self, prev_sol:iterate, reg:str ='NO_REGULARIZE')-> Feedback:
        horizon = len(self.qp.stages)
        nu, nx = self.qp.stages[0].S.shape

        if reg == 'PROJECT' or reg == 'MIRROR':
            regularized_QP = regularize_QP(self.qp, recorder=self.recorder,
                                           reg_config= self.reg_config, method=reg)
        else:
            regularized_QP = copy.deepcopy(self.qp)
        modified_QP_stages = self._modify_QP(regularized_QP ,prev_sol)

        self.recorder.reset_stage_arrays()

        feedback = Feedback(
            P = np.zeros((horizon+1, nx, nx), dtype=float),
            p = np.zeros((horizon+1, nx), dtype=float),
            K = np.zeros((horizon, nu, nx), dtype=float),
            k = np.zeros((horizon, nu), dtype=float)
        )
        feedback.P[-1] = modified_QP_stages[-1].Q_e
        feedback.p[-1] = self.rhs_dict['rg_'][-1][:nx]
        for i in reversed(range(horizon)):
            feedback.P[i], feedback.p[i], feedback.K[i], feedback.k[i] = riccati_recursion_with_stage_regularization(
            qp_stage=modified_QP_stages[i],
            qp_terminal=(feedback.P[i+1], feedback.p[i+1]),
            nx = nx,
            rg=self.rhs_dict['rg_'][i], rb=self.rhs_dict['rb'][i],
            reg_config=self.reg_config, recorder=self.recorder,
            reg=reg, index=i
            )
        return feedback

    def _fraction_to_boundary(self, prev_sol: iterate, delta_dict, boundary:bool=True) -> iterate:
        s_old = np.hstack((prev_sol.s0, prev_sol.s.flatten()))
        lam_old = np.hstack((prev_sol.lam0, prev_sol.lam.flatten()))
        delta_s = np.hstack((delta_dict['delta_s0'], delta_dict['delta_s'].flatten()))
        delta_lam =  np.hstack((delta_dict['delta_lam0'], delta_dict['delta_lam'].flatten()))

        mask_lam = delta_lam < 0
        mask_s = delta_s < 0
        frac_pri = s_old[mask_s] / (-delta_s[mask_s])
        frac_dual = lam_old[mask_lam] / (-delta_lam[mask_lam])
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

    def _modify_QP(self, qp:OCP_QP_varying, prev_sol:iterate):
        modified_QP_stages = []

        for i in range(len(self.qp.stages)):
            if i == 0:
                C_full = np.concatenate((qp.stages[0].C, qp.stages[0].D), axis=1)
                T_mat = np.diag(prev_sol.s0)
                Lam_mat = np.diag(prev_sol.lam0)
                rd = self.rhs_dict['rd0']
                rm = self.rhs_dict['rm0']
            else:
                C_full = np.concatenate((qp.stages[i].C, qp.stages[i].D), axis=1)
                T_mat = np.diag(prev_sol.s[i-1])
                Lam_mat = np.diag(prev_sol.lam[i-1])
                rd = self.rhs_dict['rd'][i-1]
                rm = self.rhs_dict['rm'][i-1]

            T_mat_inv = np.linalg.inv(T_mat)
            lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
            rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

            Q_tilt = lhs_modification[:qp.nx, :qp.nx]
            R_tilt = lhs_modification[qp.nx:, qp.nx:]
            self.curvature_list.append(R_tilt)
            S_tilt = lhs_modification[qp.nx:, :qp.nx]
            modified_QP_stages.append(QP_stage(
                Q = qp.stages[i].Q + Q_tilt,
                 # in order to make R_stage and project the same when only R is negative definite
                 # R_tilt has to be removed before riccati recursion to avoid possible successful cholesky decomposition
                 # which is due to the curvature provided by constraints (which often happens!!!)
                 # and add R_tilt back during recursion regularization after R is modified
                R = qp.stages[i].R + R_tilt,
                S = qp.stages[i].S + S_tilt,
                q = qp.stages[i].q, r = qp.stages[i].r,
                A = qp.stages[i].A, B = qp.stages[i].B, b = qp.stages[i].b,
                C = qp.stages[i].C, D = qp.stages[i].D, d = qp.stages[i].d,
            ))
            self.rhs_dict['rg_'][i][:qp.nx] = self.rhs_dict['rg'][i][:qp.nx] + rhs_modification[:qp.nx]
            self.rhs_dict['rg_'][i][qp.nx:] = self.rhs_dict['rg'][i][qp.nx:] + rhs_modification[qp.nx:]
        modified_QP_stages.append(qp.terminal)
        self.rhs_dict['rg_'][-1] = self.rhs_dict['rg'][-1]
        return modified_QP_stages

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
        for i in range(self.statistics_dict['iter_count']+1):
            res_stat_norm = self.statistics_dict['res_stat_list'][i]
            res_eq_norm = self.statistics_dict['res_eq_list'][i]
            res_ineq_norm = self.statistics_dict['res_ineq_list'][i]
            res_comp_norm = self.statistics_dict['res_comp_list'][i]
            duality_measure = self.statistics_dict['duality_measure_list'][i]
            alpha_pri = self.statistics_dict['alpha_pri_list'][i] if i < len(self.statistics_dict['alpha_pri_list']) else 0.0
            alpha_dual = self.statistics_dict['alpha_dual_list'][i] if i < len(self.statistics_dict['alpha_dual_list']) else 0.0
            print(f"{i}\t{res_stat_norm:.2e}\t{res_eq_norm:.2e}\t{res_ineq_norm:.2e}\t{res_comp_norm:.2e}\t{duality_measure:.2e}\t{alpha_pri:.2e}\t{alpha_dual:.2e}")