from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs
from solvers.Regularization_config import Regularizations_config
from solvers.Riccati_IPM import Riccati_IPM
from solvers.Condensed_IPM import Condensed_IPM
import numpy as np
import copy

class QpSolver:
    """
    Prototype solver for QPs using primal-dual interior point method with Riccati recursion.
    """
    def __init__(self, qp:BaseQP, reg:str='NO_REGULARIZE', type:str='Riccati', mehrotra:bool=True, globalization:bool=True):
        self.qp = qp
        self.status = -1
        self.centering_param = 1.0
        self.max_iter = 400
        self.delta_aff_dict = None
        self.delta_dict = None
        self.Mehrotra_step = mehrotra
        self.globalization = globalization
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
            # lazy intialization, will be updated in the first call of _compute_rhs
            rg = [],
            rb = [],
            rd = [],
            rm = [],
            # tunable parameter, will be updated in each predictor step
            duality_measure = 10.0,
        )
        self.backup_rm = None
        self.alpha_prim = 1.0
        self.alpha_dual = 1.0
        self.alpha = 1.0
        self.theta_max = 0.0

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
        x_0 = np.zeros((self.qp.nx,))
        self.solution.x = [x_0] + [x0 for i in range(self.qp.N)]
        self.solution.u = [np.zeros((self.qp.nu,)) for _ in range(self.qp.N)]
        self.solution.pi = [np.zeros((self.qp.nx,)) for _ in range(self.qp.N+1)]
        self._initailize_lam_and_s()
        _, theta_max = self._compute_objective_and_constraints(self.solution)
        self.theta_max = max(1.25 * theta_max, 100)

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
            C_k = self.qp.C[k][end_lb+nb:]
            D_k = self.qp.D[k][end_lb+nb:]

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

            s_lb_gen_raw = y_gen + d_lb_gen
            s_ub_gen_raw = d_ub_gen - y_gen

            s_lb_gen = np.maximum(s_lb_gen_raw, threshold)
            s_ub_gen = np.maximum(s_ub_gen_raw, threshold)

            s_lb_full = np.concatenate((s_lb_box, s_lb_gen))
            s_ub_full = np.concatenate((s_ub_box, s_ub_gen))
            s_k_full = np.concatenate((s_lb_full, s_ub_full))

            self.solution.s.append(s_k_full)
            self.solution.lam.append(self.rhs.duality_measure / s_k_full)


    def compute_rhs(self, prev_sol:iterate_sol)-> np.ndarray:
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
            temp_res_stat[:nx] = Q_k @ x[k] + S_k.T @ u[k] + q_k + A_k.T @ pi[k+1] + C_k.T @ lam[k] - pi[k]
            temp_res_stat[nx:] = R_k @ u[k] + S_k @ x[k] + r_k + B_k.T @ pi[k+1] + D_k.T @ lam[k]
            res_stat.append(temp_res_stat)

            #equality residual
            res_eq.append(A_k @ x[k] + B_k @ u[k] - x[k+1] + b_k)

            #inequality residual
            res_ineq.append((C_k @ x[k] + D_k @ u[k] + s[k]) - d_k)

            #complementarity residual
            temp_res_comp = (np.diag(lam[k]) @ s[k]).flatten()
            duality_measure += (lam[k] @ s[k])
            res_comp.append(temp_res_comp)

        # terminal residual
        temp_res_stat = np.zeros(nx+nu)
        Q_e = self.qp.Q_e
        q_e = self.qp.q_e
        temp_res_stat[:nx] = Q_e @ x[self.qp.N] + q_e - pi[self.qp.N]
        res_stat.append(temp_res_stat)

        self.rhs.rg = res_stat
        self.rhs.rb = res_eq
        self.rhs.rd = res_ineq
        self.rhs.rm = res_comp

        # # print for debug
        # for k in range(self.qp.N):
        #     b_k = self.qp.b[k]
        #     print(f"Stage {k}: b_k = {b_k}")
        # for k in range(self.qp.N):
        #     rq_k = np.concatenate((self.qp.r[k], self.qp.q[k]))
        #     print(f"Stage {k}: rq_k = {rq_k}")
        # for k in range(self.qp.N):
        #     # rg
        #     print(f"Stage {k}: rg = {res_stat[k]}")
        #     # rb
        #     print(f"Stage {k}: rb = {res_eq[k]}")
        #     # rd
        #     print(f"Stage {k}: rd = {res_ineq[k]}")
        #     # rm
        #     print(f"Stage {k}: rm = {res_comp[k]}")
        #     print("-" * 30)
        # breakpoint()

        self.backup_rm = copy.deepcopy(self.rhs.rm)
        self.rhs.duality_measure = max(duality_measure / self.ineq_num, 1e-16)
        self.statistics_dict['res_stat_list'].append(np.linalg.norm(np.concatenate(self.rhs.rg), ord=np.inf))
        self.statistics_dict['res_eq_list'].append(np.linalg.norm(np.concatenate(self.rhs.rb), ord=np.inf))
        self.statistics_dict['res_ineq_list'].append(np.linalg.norm(np.concatenate(self.rhs.rd), ord=np.inf))
        self.statistics_dict['res_comp_list'].append(np.linalg.norm(np.concatenate(self.rhs.rm), ord=np.inf))
        self.statistics_dict['duality_measure_list'].append(self.rhs.duality_measure)


    def update_rhs_comp(self, step_type:str='centering'):
        if step_type == 'centering':
            for k in range(self.qp.N):
                self.rhs.rm[k] = self.backup_rm[k] - self.centering_param * self.rhs.duality_measure * np.ones(len(self.rhs.rm[k]))
        elif step_type == 'predictor-corrector':
            for k in range(self.qp.N):
                self.rhs.rm[k] = self.backup_rm[k] + (self.delta_aff_dict['delta_lam'][k] * self.delta_aff_dict['delta_s'][k]) - self.centering_param * self.rhs.duality_measure * np.ones(len(self.backup_rm[k]))
        elif step_type == 'predictor':
            for k in range(self.qp.N):
                self.rhs.rm[k] = self.backup_rm[k] - 1e-16 * np.ones(len(self.backup_rm[k]))


    def _recover_solution(self):
        # enforce x0 constraint
        self.solution.x[0] = self.qp.x0
        # recover the lam at stage 0
        k = 0
        nx = self.qp.nx
        nu = self.qp.nu
        nb = self.qp.nb_0
        nh = self.qp.nh_0
        nbu = self.qp.nbu_0
        nbx = self.qp.nbx_0
        Q_k = self.qp.Q[k]
        S_k = self.qp.S[k]
        A_k = self.qp.A[k]
        q_k = self.qp.q[k]
        x, u, pi, lam, s = self.solution

        lam_lbx = np.zeros((nx,))
        lam_ubx = np.zeros((nx,))
        lam_0 = self.solution.lam[0]
        # stationarity residual
        temp_res_stat_x = np.zeros(nx)
        temp_res_stat_x = Q_k @ x[k] + S_k.T @ u[k] + q_k + A_k.T @ pi[k+1]
        lam_lbx = np.maximum(temp_res_stat_x, 1e-16)
        lam_ubx = np.maximum(-temp_res_stat_x, 1e-16)
        end_lower = nb + nh
        self.solution.lam[0] = np.concatenate((lam_0[:nbu], lam_lbx, lam_0[nbu+nx:nbu+nx+nbx],
                                                lam_0[end_lower:end_lower+nbu], lam_ubx, lam_0[end_lower+nbu+nx+nbx:]))


    def solve(self, x0:np.ndarray=None, max_iter:int=400)-> iterate_sol:
        '''
        Solve the QP using primal-dual interior point method with Riccati recursion.
        '''

        self._initilize_solution(x0)
        self.max_iter = max_iter

        prev_sol = copy.deepcopy(self.solution)
        self.compute_rhs(prev_sol=prev_sol)

        while True:

            # check convergence based on predictor step
            if self._convergence_check() or self.statistics_dict['iter_count'] == self.max_iter:
                self._quit()
                break

            self.statistics_dict['iter_count'] += 1
            self.statistics_dict['alpha_pri_list'].append(self.alpha_prim)
            self.statistics_dict['alpha_dual_list'].append(self.alpha_dual)
            self.statistics_dict['alpha_list'].append(self.alpha)

            ### Mehrotra predictor-corrector step
            ## predictor step
            self.delta_aff_dict = self._compute_step(prev_sol)
            self.alpha_prim, self.alpha_dual, self.alpha = self.step_size_to_boundary(prev_sol, self.delta_aff_dict)
            mu_aff = self._compute_duality(prev_sol, self.delta_aff_dict)
            self.centering_param = (mu_aff/self.rhs.duality_measure)**3

            if self.Mehrotra_step:
                ## predictor-corrector step
                self.update_rhs_comp(step_type='predictor-corrector')
                self.delta_dict = self._compute_step(prev_sol)
                self.alpha_prim, self.alpha_dual, self.alpha = self.step_size_to_boundary(prev_sol, self.delta_dict)
                mu_pred = self._compute_duality(prev_sol, self.delta_dict)

                ### Conditonal Mehrotra step
                if mu_pred > 2*mu_aff:
                    self.update_rhs_comp(step_type='centering')
                    self.delta_dict = self._compute_step(prev_sol)
                    self.alpha_prim, self.alpha_dual, self.alpha = self.step_size_to_boundary(prev_sol, self.delta_dict)
            else:
                self.update_rhs_comp(step_type='centering')
                self.delta_dict = self._compute_step(prev_sol)
                self.alpha_prim, self.alpha_dual, self.alpha = self.step_size_to_boundary(prev_sol, self.delta_dict)

            self.update_solution(prev_sol)
            self.statistics_dict['alpha_pri_list'][-1] = self.alpha_prim
            self.statistics_dict['alpha_dual_list'][-1] = self.alpha_dual
            self.statistics_dict['alpha_list'][-1] = self.alpha

            prev_sol = copy.deepcopy(self.solution)
            self.compute_rhs(prev_sol=prev_sol)

        self._recover_solution()
        return self.solution


    def _compute_step(self, prev_sol:iterate_sol)-> iterate_sol:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        feedback = self.solver.backward_sweep(self.rhs, prev_sol=prev_sol)
        delta_dict = self.solver.forward_sweep(prev_sol, feedback)
        return delta_dict


    def _compute_duality(self, prev_sol:iterate_sol, delta_dict:dict)-> float:
        tol = 1e-16
        s_aff = np.maximum(np.concatenate(prev_sol.s) + self.alpha_prim * np.concatenate(delta_dict['delta_s']), tol)
        lam_aff = np.maximum(np.concatenate(prev_sol.lam) + self.alpha_dual * np.concatenate(delta_dict['delta_lam']), tol)
        mu = (s_aff.flatten() @ lam_aff.flatten())/self.ineq_num
        return mu


    def update_solution(self, prev_sol:iterate_sol):
        self.alpha_prim, self.alpha_dual, _ = self._funnel_line_search(prev_sol, self.delta_dict)
        x_sol = [prev_sol.x[i] + self.alpha_prim * self.delta_dict['delta_x'][i] for i in range(self.qp.N+1)]
        u_sol = [prev_sol.u[i] + self.alpha_prim * self.delta_dict['delta_u'][i] for i in range(self.qp.N)]
        pi_sol = [prev_sol.pi[i] + self.alpha_dual * self.delta_dict['delta_pi'][i] for i in range(self.qp.N+1)]
        lam_sol = [np.maximum(prev_sol.lam[i] + self.alpha_dual * self.delta_dict['delta_lam'][i], 1e-16) for i in range(self.qp.N)]
        s_sol = [np.maximum(prev_sol.s[i] + self.alpha_prim * self.delta_dict['delta_s'][i], 1e-16) for i in range(self.qp.N)]
        self.solution = iterate_sol(x=x_sol, u=u_sol, pi=pi_sol, lam=lam_sol, s=s_sol)


    def step_size_to_boundary(self, prev_sol: iterate_sol, delta_dict) -> iterate_sol:
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
            alpha_pri_max = np.min((np.min(frac_pri), 1e0))
        except ValueError:
            alpha_pri_max = 1.0
        try:
            alpha_dual_max = np.min((np.min(frac_dual), 1e0))
        except ValueError:
            alpha_dual_max = 1.0
        alpha = min(alpha_pri_max, alpha_dual_max)

        return alpha_pri_max, alpha_dual_max, alpha


    def _funnel_line_search(self, prev_sol, delta_dict, beta=0.5, max_ls=5):

        # FTB
        alpha_p = self.alpha_prim * ((1.0 - self.alpha_prim) * 0.99 + self.alpha_prim * 0.9999999) if self.alpha_prim < 1.0 else self.alpha_prim
        alpha_d = self.alpha_dual * ((1.0 - self.alpha_dual) * 0.99 + self.alpha_dual * 0.9999999) if self.alpha_dual < 1.0 else self.alpha_dual

        if self.globalization:
            objective, norm_c = self._compute_objective_and_constraints(prev_sol)
            obj_grad = self._compute_objective_grad(prev_sol)

            direction_prim = np.concatenate(
                delta_dict['delta_x'] + 
                delta_dict['delta_u'] + 
                delta_dict['delta_s']
            )
            grad_direc_prod = obj_grad.T @ direction_prim

            for i in range(max_ls):

                cand = iterate_sol(
                    x   = [prev_sol.x[i]   + alpha_p * delta_dict['delta_x'][i]   for i in range(self.qp.N + 1)],
                    u   = [prev_sol.u[i]   + alpha_p * delta_dict['delta_u'][i]   for i in range(self.qp.N)],
                    pi  = [prev_sol.pi[i]  + alpha_d * delta_dict['delta_pi'][i]  for i in range(self.qp.N + 1)],
                    lam = [np.maximum(prev_sol.lam[i] + alpha_d * delta_dict['delta_lam'][i], 1e-16) for i in range(self.qp.N)],
                    s   = [np.maximum(prev_sol.s[i]   + alpha_p * delta_dict['delta_s'][i],   1e-16) for i in range(self.qp.N)],
                )

                objective_cand, norm_c_cand = self._compute_objective_and_constraints(cand)
                # check acceptablity
                if norm_c_cand <= self.theta_max:
                    # check f type condition
                    if alpha_p*(-grad_direc_prod)>1e-4*norm_c**2:
                        # check Armijo condition
                        if objective_cand <= objective + 1e-8*alpha_p*grad_direc_prod:
                            break
                        else:
                            alpha_p *= beta
                    else:
                        # check theta type condition
                        if norm_c_cand <= 0.9999*self.theta_max:
                            # update theta_max
                            self.theta_max = (1-0.5)*self.theta_max + 0.5*norm_c_cand
                            break
                        else:
                            alpha_p *= beta
                else:
                    alpha_p *= beta
        else:
            pass

        return alpha_p, alpha_d, min(alpha_p, alpha_d)


    def _compute_objective_grad(self, sol):
        x, u, pi, lam, s = sol

        grad_x = []
        grad_u = []
        grad_s = []
        tau = self.centering_param*self.rhs.duality_measure

        for k in range(self.qp.N):
            Q_k = self.qp.Q[k];  R_k = self.qp.R[k];  S_k = self.qp.S[k]
            q_k = self.qp.q[k];  r_k = self.qp.r[k]

            # Gradient of the actual objective function f(x, u)
            grad_x_k = Q_k @ x[k] + S_k.T @ u[k] + q_k
            grad_u_k = R_k @ u[k] + S_k @ x[k] + r_k
            
            # Gradient of the barrier term: -tau / s
            grad_s_k = -tau / s[k]

            grad_x.append(grad_x_k)
            grad_u.append(grad_u_k)
            grad_s.append(grad_s_k)

        # Terminal cost gradient
        Q_e = self.qp.Q_e
        q_e = self.qp.q_e
        grad_x_N = Q_e @ x[self.qp.N] + q_e
        grad_x.append(grad_x_N)

        # Return a single concatenated 1D numpy array
        return np.concatenate(grad_x + grad_u + grad_s)


    def _compute_objective_and_constraints(self, sol: iterate_sol) -> float:
        x, u, pi, lam, s = sol

        obj = 0.0
        barrier = 0.0
        res_eq_sq = 0.0
        res_ineq_sq = 0.0
        tau = self.centering_param*self.rhs.duality_measure

        for k in range(self.qp.N):
            Q_k = self.qp.Q[k];  R_k = self.qp.R[k];  S_k = self.qp.S[k]
            A_k = self.qp.A[k];  B_k = self.qp.B[k];  b_k = self.qp.b[k]
            C_k = self.qp.C[k];  D_k = self.qp.D[k];  d_k = self.qp.d[k]
            q_k = self.qp.q[k];  r_k = self.qp.r[k]

            obj += 0.5 * x[k] @ Q_k @ x[k] + 0.5 * u[k] @ R_k @ u[k] + u[k] @ S_k @ x[k] + q_k @ x[k] + r_k @ u[k]

            barrier += np.sum(np.log(s[k]))

            r_eq = A_k @ x[k] + B_k @ u[k] - x[k+1] + b_k
            res_eq_sq += r_eq @ r_eq

            r_ineq = C_k @ x[k] + D_k @ u[k] + s[k] - d_k
            res_ineq_sq += r_ineq @ r_ineq

        obj += 0.5 * x[self.qp.N] @ self.qp.Q_e @ x[self.qp.N] + self.qp.q_e @ x[self.qp.N]
        norm_c = np.sqrt(res_eq_sq + res_ineq_sq)

        return obj - tau * barrier, norm_c


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
            print(f"\033[31mNot Converged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.6e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.6e}")
        else:
            print(f"\033[32mConverged!\033[0m after {self.statistics_dict['iter_count']} with KKT residual {res:.6e} and barrier parameter {self.statistics_dict['duality_measure_list'][-1]:.6e}")
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

    def get_stats(self, field:str):
        if field == 'iter':
            return self.statistics_dict['iter_count']

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
            print(f"{i}\t{res_stat_norm:.6e}\t{res_eq_norm:.6e}\t{res_ineq_norm:.6e}\t{res_comp_norm:.6e}\t{duality_measure:.6e}\t{alpha_pri:.6e}\t{alpha_dual:.6e}")