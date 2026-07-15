from QP_template.QP_base import BaseQP
from QP_template.iterate_base import iterate_sol, iterate_rhs
from solvers.Regularization_config import Regularizations_config, Iteration_based_reg, Block_Hessian_reg
from solvers.Riccati_IPM import Riccati_IPM
# from solvers.Riccati_IPM_sqrt import Riccati_IPM
from solvers.Riccati_IPM_reg import Riccati_IPM as Riccati_IPM_reg
import numpy as np
import copy

class QpSolver:
    """
    Prototype solver for QPs using primal-dual interior point method with Riccati recursion.
    """
    def __init__(self, qp:BaseQP, reg:str='NO_REGULARIZE', type:str='Riccati', globalization:bool=True):
        self.qp = qp
        self.status = -1
        self.max_iter = 400
        self.delta_dict = None
        self.globalization = globalization

        # ---- Fiacco-McCormick (SUMT) barrier settings ----
        self.tol = 1e-16
        self.kappa_mu = 0.2
        self.theta_mu = 1.5
        self.inner_kappa = 1e1
        self.max_inner = 10
        self.ftb_tau_min = 0.995
        self.s_max = 100.0
        self.mu = 1e1

        self.res_max = {
            'res_stat' : 1e-6,
            'res_eq' : 1e-8,
            'res_ineq' : 1e-8,
            'res_comp' : 5e-6,
        }

        # statistics for analysis
        self.stat_dict = {
            'res_stat_list' : [],
            'res_eq_list' : [],
            'res_ineq_list' : [],
            'res_comp_list' : [],
            'inf_primal' : [],
            'inf_dual' : [],
            'alpha_pri_list' : [],
            'alpha_dual_list' : [],
            'alpha_list' : [],
            'duality_measure_list' : [],
            'obj' : [],
            'ls' : [],
            'iter_count' : 0,
            'decomp_flop' : 0,
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
            if reg in Iteration_based_reg + Block_Hessian_reg:
                self.solver = Riccati_IPM_reg(qp,
                                            Regularizations_config(method=reg, delta=1e-4),
                                            ineq_num=self.ineq_num,
                                            stat_dict=self.stat_dict)
            else:
                self.solver = Riccati_IPM(qp,
                                        Regularizations_config(method=reg, delta=1e-4),
                                        ineq_num=self.ineq_num,
                                        stat_dict=self.stat_dict)
        else:
            raise ValueError(f"Invalid solver type {type}. Valid types are 'Riccati' and 'Condensed'.")


    def _initilize_solution(self, x0:np.ndarray):
        self.solution.x = [np.zeros(self.qp.nx[k]) for k in range(self.qp.N+1)]
        self.solution.u = [np.zeros(self.qp.nu) for _ in range(self.qp.N)]
        self.solution.pi = [np.zeros((self.qp.nx[k],)) for k in range(self.qp.N+1)]
        self._initialize_lam_and_s()

        if self.globalization:
            _, theta_max = self._compute_objective_and_constraints(self.solution)
            self.theta_max = max(1.25 * theta_max, 100)


    def _initialize_lam_and_s(self):
        self.solution.s = []
        self.solution.lam = []
        thr = 1e-1
        mu0 = self.mu
        for k in range(self.qp.N + 1):
            if k == 0:
                nb, nh = self.qp.nb_0, self.qp.nh_0
            elif k == self.qp.N:
                nb, nh = self.qp.nb_e, self.qp.nh_e
            else:
                nb, nh = self.qp.nb, self.qp.nh
            end_lb = nb + nh

            Jbx_k, Jbu_k = self.qp.Jbx[k], self.qp.Jbu[k]
            d_k = self.qp.d[k]
            C_k = self.qp.C[k][end_lb + nb:]
            D_k = self.qp.D[k][end_lb + nb:]
            idxbu = np.nonzero(Jbu_k)[1] if Jbu_k.shape[0] > 0 else np.empty(0, dtype=int)
            idxbx = np.nonzero(Jbx_k)[1] if Jbx_k.shape[0] > 0 else np.empty(0, dtype=int)
            nbu = idxbu.size

            v = np.concatenate((
                self.solution.u[k][idxbu] if nbu > 0 else np.empty(0),
                self.solution.x[k][idxbx] if idxbx.size > 0 else np.empty(0),
            ))
            lb = -d_k[:nb]
            ub = d_k[end_lb:end_lb + nb]

            t_lb = v - lb
            t_ub = ub - v
            small_lb = t_lb < thr
            small_ub = t_ub < thr
            both    = small_lb &  small_ub
            only_lb = small_lb & ~small_ub
            only_ub = small_ub & ~small_lb

            v[both]    = 0.5 * (lb[both] + ub[both]); t_lb[both] = thr; t_ub[both] = thr
            v[only_lb] = lb[only_lb] + thr;           t_lb[only_lb] = thr
            v[only_ub] = ub[only_ub] - thr;           t_ub[only_ub] = thr

            s_lb_box, s_ub_box = t_lb, t_ub
            if nbu > 0:
                self.solution.u[k][idxbu] = v[:nbu]
            if idxbx.size > 0:
                self.solution.x[k][idxbx] = v[nbu:]

            if k < self.qp.N:
                y = C_k @ self.solution.x[k] + D_k @ self.solution.u[k]
            else:
                y = C_k @ self.solution.x[k]
            s_lb_gen = np.maximum(y + d_k[nb:end_lb], thr)
            s_ub_gen = np.maximum(d_k[end_lb + nb:end_lb + nb + nh] - y, thr)

            s_k = np.concatenate((s_lb_box, s_lb_gen, s_ub_box, s_ub_gen))
            self.solution.s.append(s_k)
            self.solution.lam.append(mu0 / s_k)


    def compute_rhs(self, prev_sol:iterate_sol)-> np.ndarray:
        nu = self.qp.nu

        res_stat = []
        res_eq = []
        res_ineq = []
        vio_ineq = []
        res_comp = []

        for k in range(self.qp.N+1):
            x, u, pi, lam, s = prev_sol
            nx = self.qp.nx[k]
            if k < self.qp.N:
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

                # stationarity residual
                temp_res_stat = np.zeros(nx + nu)
                temp_res_stat[:nx] = Q_k @ x[k] + S_k.T @ u[k] + q_k + A_k.T @ pi[k+1] + C_k.T @ lam[k] - pi[k]
                temp_res_stat[nx:] = R_k @ u[k] + S_k @ x[k] + r_k + B_k.T @ pi[k+1] + D_k.T @ lam[k]
                res_stat.append(temp_res_stat)

                #equality residual
                res_eq.append(A_k @ x[k] + B_k @ u[k] - x[k+1] + b_k)

                #inequality residual
                res_ineq.append((C_k @ x[k] + D_k @ u[k] + s[k]) - d_k)
                vio_ineq.append(np.maximum(0, (C_k @ x[k] + D_k @ u[k]) - d_k))

                #complementarity residual
                temp_res_comp = (np.diag(lam[k]) @ s[k]).flatten()
                res_comp.append(temp_res_comp)
            else:
                Q_e = self.qp.Q_e
                q_e = self.qp.q_e
                C_k = self.qp.C[k]
                d_k = self.qp.d[k]

                # stationarity residual
                temp_res_stat = np.zeros(nx)
                temp_res_stat = Q_e @ x[k] + q_e + C_k.T @ lam[k] - pi[k]
                res_stat.append(temp_res_stat)

                #inequality residual
                res_ineq.append((C_k @ x[k] + s[k]) - d_k)
                vio_ineq.append(np.maximum(0, (C_k @ x[k]) - d_k))

                #complementarity residual
                temp_res_comp = (np.diag(lam[k]) @ s[k]).flatten()
                res_comp.append(temp_res_comp)

        self.rhs.rg = res_stat
        self.rhs.rb = res_eq
        self.rhs.rd = res_ineq
        self.rhs.rm = res_comp

        self.backup_rm = copy.deepcopy(self.rhs.rm)
        sd, _ = self._scaling_factors()
        self.stat_dict['res_stat_list'].append(np.linalg.norm(np.concatenate(self.rhs.rg), ord=np.inf))
        self.stat_dict['res_eq_list'].append(np.linalg.norm(np.concatenate(self.rhs.rb), ord=np.inf))
        self.stat_dict['res_ineq_list'].append(np.linalg.norm(np.concatenate(self.rhs.rd), ord=np.inf))
        self.stat_dict['res_comp_list'].append(np.linalg.norm(np.concatenate(self.rhs.rm), ord=np.inf))

        vio_ineq_norm = np.linalg.norm(np.concatenate(vio_ineq), ord=np.inf)
        self.stat_dict['inf_primal'].append(max(self.stat_dict['res_eq_list'][-1], vio_ineq_norm))
        self.stat_dict['inf_dual'].append(self.stat_dict['res_stat_list'][-1] / sd)


    def update_rhs_comp(self):
        '''Set the barrier-perturbed complementarity residual for the current
        barrier parameter tau:  rm = Lam*S*e - tau*e  (target Lam*S = tau*e).'''
        for k in range(self.qp.N+1):
            self.rhs.rm[k] = self.backup_rm[k] - self.mu * np.ones(len(self.backup_rm[k]))


    def _recover_solution(self):
        k = 0
        nx  = self.qp.nx[1]
        nbu = self.qp.nbu_0
        nbx = self.qp.nbx_0
        nh  = self.qp.nh_0
        elim = self.qp.idxe_0
        free = self.qp.free_idx_0

        x, u, pi, lam, s = self.solution
        lam_0 = self.solution.lam[0]

        x0_full = np.zeros(nx)
        x0_full[free] = x[k]
        x0_full[elim] = self.qp.x0
        self.solution.x[0] = x0_full

        res_e = (self.qp.Q0e @ x0_full + self.qp.S0e.T @ u[k]
                + self.qp.q0e + self.qp.A0e.T @ pi[k+1])
        lam_lbx = np.zeros(nx); lam_ubx = np.zeros(nx)
        lam_lbx[elim] = np.maximum( res_e, 1e-16)
        lam_ubx[elim] = np.maximum(-res_e, 1e-16)

        end_lower = nbu + nbx + nh
        lam_lbx[free] = lam_0[nbu:nbu+nbx]
        lam_ubx[free] = lam_0[end_lower+nbu:end_lower+nbu+nbx]

        self.solution.lam[0] = np.concatenate((
            lam_0[:nbu],
            lam_lbx,
            lam_0[nbu+nbx:nbu+nbx+nh],
            lam_0[end_lower:end_lower+nbu],
            lam_ubx,
            lam_0[end_lower+nbu+nbx+nh:],
        ))

        # recover matrix by block concatenation
        Q_0 = np.zeros((nx, nx))
        Q_0[np.ix_(free, free)] = self.qp.Q[k]
        Q_0[np.ix_(elim, free)] = self.qp.Q0e[:, free]
        Q_0[np.ix_(free, elim)] = self.qp.Q0e[:, free].T
        Q_0[np.ix_(elim, elim)] = self.qp.Q0e[:, elim]
        self.qp.Q[k] = Q_0

        A_0 = np.zeros((nx, nx))
        A_0[:, free] = self.qp.A[k]
        A_0[:, elim] = self.qp.A0e
        self.qp.A[k] = A_0

        S_0 = np.zeros((self.qp.nu, nx))
        S_0[:, free] = self.qp.S[k]
        S_0[:, elim] = self.qp.S0e
        self.qp.S[k] = S_0

        q_0 = np.zeros(nx)
        q_0[free] = self.qp.q[k] - self.qp.Q0e[:, free].T @ self.qp.x0
        q_0[elim] = self.qp.q0e
        self.qp.q[k] = q_0

        self.qp.r[k] = self.qp.r[k] - self.qp.S0e @ self.qp.x0
        self.qp.b[k] = self.qp.b[k] - self.qp.A0e @ self.qp.x0
        
        self.qp.nx[k] = nx


    def solve(self, x0:np.ndarray=None, max_iter:int=100)-> iterate_sol:
        self._initilize_solution(x0)
        self.max_iter = max_iter

        prev_sol = copy.deepcopy(self.solution)

        inner_iter = 0

        while True:
            self.compute_rhs(prev_sol=prev_sol)
            self.stat_dict['duality_measure_list'].append(self.mu)


            if self._inf_check():
                self.status = 3
                self._quit()
                break
            if self._convergence_check():
                self.status = 0
                self._quit()
                break
            if self.stat_dict['iter_count'] >= self.max_iter:
                self.status = 2
                self._quit()
                break

            if (self._optimality_error(self.mu) <= self.inner_kappa * self.mu or inner_iter >= self.max_inner):
                self.mu = max(self.tol / 10.0,  min(self.kappa_mu * self.mu, self.mu ** self.theta_mu))
                inner_iter = 0

            self.stat_dict['iter_count'] += 1
            self.stat_dict['obj'].append(self.get_cost())
            inner_iter += 1


            if self.stat_dict['iter_count'] == 11:
                print('Debug')

            # fixed-tau centering Newton step:  rm = Lam*S*e - tau*e
            self.update_rhs_comp()
            self.delta_dict = self._compute_step(prev_sol)

            if self.delta_dict is None:
                self.status = 1; self._quit(); break

            self.alpha_prim, self.alpha_dual, self.alpha = self.step_size_to_boundary(prev_sol, self.delta_dict)
            self.update_solution(prev_sol)
            self.stat_dict['alpha_pri_list'].append(self.alpha_prim)
            self.stat_dict['alpha_dual_list'].append(self.alpha_dual)
            self.stat_dict['alpha_list'].append(self.alpha)

            prev_sol = copy.deepcopy(self.solution)

        self._recover_solution()
        return self.solution


    def _scaling_factors(self):
        lam_1 = np.linalg.norm(np.concatenate(self.solution.lam), ord=1)
        pi_1  = np.linalg.norm(np.concatenate(self.solution.pi),  ord=1)
        n = self.ineq_num
        m = sum(len(p) for p in self.solution.pi)
        s_d = max(self.s_max, (pi_1 + lam_1) / max(m + n, 1)) / self.s_max
        s_c = max(self.s_max, lam_1 / max(n, 1)) / self.s_max
        return s_d, s_c


    def _optimality_error(self, mu:float) -> float:
        s_d, s_c = self._scaling_factors()
        res_stat = np.linalg.norm(np.concatenate(self.rhs.rg), ord=np.inf)
        res_eq = np.linalg.norm(np.concatenate(self.rhs.rb), ord=np.inf)
        res_comp = np.linalg.norm(np.concatenate(self.backup_rm) - mu, ord=np.inf)
        return max(res_stat / s_d,
                   res_eq,
                   res_comp / s_c)


    def _compute_step(self, prev_sol:iterate_sol)-> iterate_sol:
        '''Compute the search direction by solving the KKT system with Riccati recursion.'''
        feedback = self.solver.backward_sweep(self.rhs, prev_sol=prev_sol)
        delta_dict = self.solver.forward_sweep(prev_sol, feedback) if feedback is not None else None
        # if delta_dict is not None and self.mu < 1e-5:
        #     self._probe_direction_quality(prev_sol, delta_dict)
        return delta_dict

    def _probe_direction_quality(self, prev_sol, dd):
        x, u, pi, lam, s = prev_sol
        lin_comp = []
        for k in range(self.qp.N + 1):
            row = (s[k] * dd['delta_lam'][k]
                + lam[k] * dd['delta_s'][k]
                + lam[k] * s[k] - self.mu)
            lin_comp.append(row)
        lin_comp = np.concatenate(lin_comp)
        rhs_comp = np.concatenate([lam[k]*s[k] - self.mu for k in range(self.qp.N+1)])
        rel = np.linalg.norm(lin_comp, np.inf) / max(np.linalg.norm(rhs_comp, np.inf), 1e-30)
        print(f"[probe] mu={self.mu:.2e}  comp_lin_rel={rel:.2e}  "
              f'min_lam={min(np.min(li) for li in lam):.2e}  '
            f"min_s={min(np.min(si) for si in s):.2e}  "
            f"max_lam_over_s={max(np.max(l/si) for l,si in zip(lam,s)):.2e}")


    def update_solution(self, prev_sol:iterate_sol):
        tol_s = 1e-11
        tol_lam = 1e-12
        self.alpha_prim, self.alpha_dual, _ = self._funnel_line_search(prev_sol, self.delta_dict)
        x_sol = [prev_sol.x[i] + self.alpha_prim * self.delta_dict['delta_x'][i] for i in range(self.qp.N+1)]
        u_sol = [prev_sol.u[i] + self.alpha_prim * self.delta_dict['delta_u'][i] for i in range(self.qp.N)]
        pi_sol = [prev_sol.pi[i] + self.alpha_dual * self.delta_dict['delta_pi'][i] for i in range(self.qp.N+1)]
        lam_sol = [np.maximum(prev_sol.lam[i] + self.alpha_dual * self.delta_dict['delta_lam'][i], tol_lam) for i in range(self.qp.N+1)]
        s_sol = [np.maximum(prev_sol.s[i] + self.alpha_prim * self.delta_dict['delta_s'][i], tol_s) for i in range(self.qp.N+1)]
        self.solution = iterate_sol(x=x_sol, u=u_sol, pi=pi_sol, lam=lam_sol, s=s_sol)


    def step_size_to_boundary(self, prev_sol: iterate_sol, delta_dict) -> iterate_sol:
        tau_ftb = max(self.ftb_tau_min, 1.0 - self.mu)
        s_i = np.concatenate(prev_sol.s)
        lam_i = np.concatenate(prev_sol.lam)
        delta_s = np.concatenate(delta_dict['delta_s'])
        delta_lam = np.concatenate(delta_dict['delta_lam'])

        mask_s = delta_s < 0
        mask_lam = delta_lam < 0
        frac_pri  = s_i[mask_s]/(-delta_s[mask_s])
        frac_dual = lam_i[mask_lam]/(-delta_lam[mask_lam])
        alpha_pri_max  = min(1.0, tau_ftb * np.min(frac_pri)) if frac_pri.size else 1.0
        alpha_dual_max = min(1.0, tau_ftb * np.min(frac_dual)) if frac_dual.size else 1.0
        return alpha_pri_max, alpha_dual_max, min(alpha_pri_max, alpha_dual_max)


    def _funnel_line_search(self, prev_sol, delta_dict, beta=0.5, max_ls=5):

        alpha_p = self.alpha_prim
        alpha_d = self.alpha_dual

        self.stat_dict['ls'].append("--")
        if self.globalization:
            objective, norm_c = self._compute_objective_and_constraints(prev_sol)
            obj_grad = self._compute_objective_grad(prev_sol)
            ls = 0

            direction_prim = np.concatenate(
                delta_dict['delta_x'] +
                delta_dict['delta_u'] +
                delta_dict['delta_s']
            )
            grad_direc_prod = obj_grad.T @ direction_prim

            for i in range(max_ls):
                ls += 1

                cand = iterate_sol(
                    x   = [prev_sol.x[i] + alpha_p * delta_dict['delta_x'][i] for i in range(self.qp.N + 1)],
                    u   = [prev_sol.u[i] + alpha_p * delta_dict['delta_u'][i] for i in range(self.qp.N)],
                    pi  = [prev_sol.pi[i] + alpha_d * delta_dict['delta_pi'][i] for i in range(self.qp.N + 1)],
                    lam = [np.maximum(prev_sol.lam[i] + alpha_d * delta_dict['delta_lam'][i], 1e-12) for i in range(self.qp.N+1)],
                    s   = [np.maximum(prev_sol.s[i] + alpha_p * delta_dict['delta_s'][i], 1e-12) for i in range(self.qp.N+1)],
                )

                objective_cand, norm_c_cand = self._compute_objective_and_constraints(cand)
                # check acceptablity
                if norm_c_cand <= self.theta_max:
                    # check f type condition
                    if alpha_p*(-grad_direc_prod)>1e-4*norm_c**2:
                        # check Armijo condition
                        if objective_cand <= objective + 1e-4*alpha_p*grad_direc_prod:
                            self.stat_dict['ls'][-1] = str(ls)+"f"
                            break
                        else:
                            alpha_p *= beta
                            if i == max_ls - 1:
                                self.stat_dict['ls'][-1] = '5x'
                    else:
                        # check theta type condition
                        if norm_c_cand <= 0.9999*self.theta_max:
                            # update theta_max
                            self.theta_max = max((1-0.5)*self.theta_max + 0.5*norm_c_cand, self.theta_max*0.9999)
                            self.stat_dict['ls'][-1] = str(ls)+"h"
                            break
                        else:
                            alpha_p *= beta
                            if i == max_ls - 1:
                                self.stat_dict['ls'][-1] = '5x'
                else:
                    alpha_p *= beta
                    if i == max_ls - 1:
                        self.stat_dict['ls'][-1] = '5x'

        else:
            pass

        return alpha_p, alpha_d, min(alpha_p, alpha_d)


    def _compute_objective_grad(self, sol):
        x, u, pi, lam, s = sol

        grad_x = []
        grad_u = []
        grad_s = []
        tau = self.mu

        for k in range(self.qp.N+1):
            if k < self.qp.N:
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
            else:
                # Terminal cost gradient
                Q_e = self.qp.Q_e
                q_e = self.qp.q_e

                grad_x_N = Q_e @ x[self.qp.N] + q_e
                grad_x.append(grad_x_N)

                grad_s_k = -tau / s[k]
                grad_s.append(grad_s_k)

        # Return a single concatenated 1D numpy array
        return np.concatenate(grad_x + grad_u + grad_s)


    def _compute_objective_and_constraints(self, sol: iterate_sol) -> float:
        x, u, pi, lam, s = sol

        obj = 0.0
        barrier = 0.0
        res_eq_sq = 0.0
        res_ineq_sq = 0.0
        tau = self.mu

        for k in range(self.qp.N+1):
            if k < self.qp.N:
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
            else:
                obj += 0.5 * x[self.qp.N] @ self.qp.Q_e @ x[self.qp.N] + self.qp.q_e @ x[self.qp.N]
                barrier += np.sum(np.log(s[k]))

                C_k = self.qp.C[k];  d_k = self.qp.d[k]
                r_ineq = C_k @ x[k] + s[k] - d_k
                res_ineq_sq += r_ineq @ r_ineq

        norm_c = np.sqrt(res_eq_sq + res_ineq_sq)

        return obj - tau * barrier, norm_c


    def _convergence_check(self, relative_tol:float=0)-> bool:
        '''Check convergence based on KKT residuals.'''
        return ((self.stat_dict['res_stat_list'][-1] < self.res_max['res_stat'] + relative_tol*self.res_max['res_stat']) and
                (self.stat_dict['res_eq_list'][-1] < self.res_max['res_eq'] + relative_tol*self.res_max['res_eq']) and
                (self.stat_dict['res_ineq_list'][-1] < self.res_max['res_ineq'] + relative_tol*self.res_max['res_ineq']) and
                (self.stat_dict['res_comp_list'][-1] < self.res_max['res_comp'] + relative_tol*self.res_max['res_comp']))


    def _inf_check(self):
        res = max(self.stat_dict['res_stat_list'][-1],
                  self.stat_dict['res_eq_list'][-1],
                  self.stat_dict['res_ineq_list'][-1],
                  self.stat_dict['res_comp_list'][-1])
        return np.isinf(res) or np.isnan(res)


    def _quit(self):
        res_dict = {
            'stat': self.stat_dict['res_stat_list'][-1],
            'eq': self.stat_dict['res_eq_list'][-1],
            'ineq': self.stat_dict['res_ineq_list'][-1],
            'comp': self.stat_dict['res_comp_list'][-1]
        }
        max_name, res = max(res_dict.items(), key=lambda item: item[1])
        if self.status == 0:
            print(f"\033[32mConverged!\033[0m after {self.stat_dict['iter_count']} with KKT residual {res:.6e} (from {max_name}) and barrier parameter {self.stat_dict['duality_measure_list'][-1]:.6e}")
            return
        if self.status == 1:
            print(f"\033[31mUnregularized!\033[0m after {self.stat_dict['iter_count']} iterations.")
            return
        if self.stat_dict['iter_count'] >= self.max_iter:
            print(f"\033[31mMax iters!\033[0m after {self.stat_dict['iter_count']} with KKT residual {res:.6e} (from {max_name}) and barrier parameter {self.stat_dict['duality_measure_list'][-1]:.6e}")
            self.status = 2
            return
        if self.status == 3:
            print(f"\033[31mDiverged!\033[0m after {self.stat_dict['iter_count']} iterations.")
            return


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
            return self.stat_dict['iter_count']


    def print_statistics(self):
        print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tbarrier\t\talpha_prim\talpha_dual\tobj\t\tls')
        for i in range(self.stat_dict['iter_count']+1):
            res_stat_norm = self.stat_dict['res_stat_list'][i] if i < len(self.stat_dict['res_stat_list']) else 0.0
            res_eq_norm = self.stat_dict['res_eq_list'][i] if i < len(self.stat_dict['res_eq_list']) else 0.0
            res_ineq_norm = self.stat_dict['res_ineq_list'][i] if i < len(self.stat_dict['res_ineq_list']) else 0.0
            res_comp_norm = self.stat_dict['res_comp_list'][i] if i < len(self.stat_dict['res_comp_list']) else 0.0
            duality_measure = self.stat_dict['duality_measure_list'][i] if i < len(self.stat_dict['duality_measure_list']) else 0.0
            alpha_pri = self.stat_dict['alpha_pri_list'][i] if i < len(self.stat_dict['alpha_pri_list']) else 0.0
            alpha_dual = self.stat_dict['alpha_dual_list'][i] if i < len(self.stat_dict['alpha_dual_list']) else 0.0
            cost = self.stat_dict['obj'][i] if i < len(self.stat_dict['obj']) else 0.0
            ls = self.stat_dict['ls'][i] if i < len(self.stat_dict['ls']) else "--"
            print(f"{i}\t{res_stat_norm:.6e}\t{res_eq_norm:.6e}\t{res_ineq_norm:.6e}\t{res_comp_norm:.6e}\t{duality_measure:.6e}\t{alpha_pri:.6e}\t{alpha_dual:.6e}\t{cost:.6e}\t{ls}")