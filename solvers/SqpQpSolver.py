from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager, AcadosOcpQp

import casadi as ca
import numpy as np


class SqpQpSolver(AcadosOcpSolver):

    def __init__(self, qp: AcadosOcpQp, nlp_max_iter=100, mode='nonconvex'):

        self.qp = qp
        self.mode = mode
        parametric_ocp, self.param_manager = self.formulate_parametric_OCP_QP(nlp_max_iter)
        super().__init__(parametric_ocp, verbose=False)

    def solve(self):
        self._populate_ocp_solver()
        status = super().solve()
        if status != 0:
            print(f"ocp_solver returned status {status}.")
        return status

    def _populate_ocp_solver(self):
        N = self.qp.N
        nx = self.qp.A[0].shape[0]
        nu = self.qp.B[0].shape[1]

        nh_0 = self.qp.C[0].shape[0]
        nh = self.qp.C[1].shape[0]
        nh_e = self.qp.C[N].shape[0]

        nbx_0 = self.qp.lbx[0].shape[0]
        nbx = self.qp.lbx[1].shape[0]
        nbx_e = self.qp.lbx[N].shape[0]

        nbu = self.qp.lbu[0].shape[0]

        # Set parameters value in param_manager
        for i in range(N+1):
            self.param_manager.set_value(name="Q", value=self.qp.Q[i], stage=i)
            self.param_manager.set_value(name="q", value=self.qp.q[i], stage=i)
            if i < N:
                self.param_manager.set_value(name="A", value=self.qp.A[i], stage=i)
                self.param_manager.set_value(name="B", value=self.qp.B[i], stage=i)
                self.param_manager.set_value(name="b", value=self.qp.b[i], stage=i)
                self.param_manager.set_value(name="R", value=self.qp.R[i], stage=i)
                self.param_manager.set_value(name="S", value=self.qp.S[i], stage=i)
                self.param_manager.set_value(name="r", value=self.qp.r[i], stage=i)

                if nbu > 0:
                    self.param_manager.set_value(name="lbu", value=self.qp.lbu[i], stage=i)
                    self.param_manager.set_value(name="ubu", value=self.qp.ubu[i], stage=i)
                else:
                    self.param_manager.set_value(name="lbu", value=-1e4*np.ones((nu, 1)), stage=i)
                    self.param_manager.set_value(name="ubu", value=1e4*np.ones((nu, 1)), stage=i)

            nbx_ = nbx_0 if i == 0 else nbx_e if i == N else nbx
            lbx_value = self.qp.lbx[i] if self.qp.lbx[i].shape[0] > 0 else -1e4*np.ones((nbx_, 1))
            ubx_value = self.qp.ubx[i] if self.qp.ubx[i].shape[0] > 0 else 1e4*np.ones((nbx_, 1))
            self.param_manager.set_value(name="lbx", value=lbx_value, stage=i)
            self.param_manager.set_value(name="ubx", value=ubx_value, stage=i)

            nh_ = nh_0 if i == 0 else nh_e if i == N else nh
            C_value = self.qp.C[i] if self.qp.C[i].shape[0] > 0 else np.zeros((nh_, nx))
            D_value = self.qp.D[i] if self.qp.D[i].shape[0] > 0 else np.zeros((nh_, nu))
            lg_value = self.qp.lg[i] if self.qp.lg[i].shape[0] > 0 else -1e4*np.ones((nh_, 1))
            ug_value = self.qp.ug[i] if self.qp.ug[i].shape[0] > 0 else 1e4*np.ones((nh_, 1))
            self.param_manager.set_value(name="C", value=C_value, stage=i)
            if i < N:
                self.param_manager.set_value(name="D", value=D_value, stage=i)
            self.param_manager.set_value(name="lg", value=lg_value, stage=i)
            self.param_manager.set_value(name="ug", value=ug_value, stage=i)

        # Set parameters value in the solver by param_manager
        for i in range(N+1):
            param_value = self.param_manager.get_p_stagewise_values(i)
            param_dyna_cost_values = param_value[:self.param_manager.p_dyna_num + self.param_manager.p_cost_num]
            param_cons_values = param_value[self.param_manager.p_dyna_num + self.param_manager.p_cost_num:]

            # Set dynamic and cost parameters
            self.set(i, 'p', param_dyna_cost_values)

            # Set constraints in bu, bx, b
            nbx_ = nbx_0 if i == 0 else nbx_e if i == N else nbx
            lbx = param_cons_values[:nbx_].flatten()
            ubx = param_cons_values[nbx_:2*nbx_].flatten()
            if nbx_ > 0:
                self.constraints_set(i, 'lbx', lbx)
                self.constraints_set(i, 'ubx', ubx)

            end_bx = 2 * nbx_
            lbu = param_cons_values[end_bx:end_bx+nu].flatten()
            ubu = param_cons_values[end_bx+nu:end_bx+2*nu].flatten()
            if nbu > 0 and i < N:
                self.constraints_set(i, 'lbu', lbu)
                self.constraints_set(i, 'ubu', ubu)

            end_nb = end_bx + 2*nu
            nh_ = nh_0 if i == 0 else nh_e if i == N else nh
            C_k = param_cons_values[end_nb:end_nb+nh_*nx].reshape((nh_, nx), order='F')
            D_k = param_cons_values[end_nb+nh_*nx:end_nb+nh_*nx+nh_*nu].reshape((nh_, nu), order='F')
            end_ng = end_nb + nh_*nx + nh_*nu
            lg = param_cons_values[end_ng:end_ng+nh_].flatten()
            ug = param_cons_values[end_ng+nh_:end_ng+2*nh_].flatten()
            if nh_ > 0:
                self.constraints_set(i, 'C', C_k)
                if i < N:
                    self.constraints_set(i, 'D', D_k)
                self.constraints_set(i, 'lg', lg)
                self.constraints_set(i, 'ug', ug)

    def formulate_parametric_OCP_QP(self, nlp_max_iter):
        ocp = AcadosOcp()
        ocp.model.name = 'parametric_OCP_QP'

        N = self.qp.N
        nx = self.qp.A[0].shape[0]
        nu = self.qp.B[0].shape[1]

        nh_0 = self.qp.C[0].shape[0]
        nh = self.qp.C[1].shape[0]
        nh_e = self.qp.C[N].shape[0]

        nbx_0 = self.qp.lbx[0].shape[0]
        nbx = self.qp.lbx[1].shape[0]
        nbx_e = self.qp.lbx[N].shape[0]

        nbx_max = max(nbx_0, nbx, nbx_e)
        nbu = self.qp.lbu[0].shape[0]
        nh_max = max(nh_0, nh, nh_e)

        # params
        Q_param = AcadosParam("Q", np.zeros((nx, nx)))
        q_param = AcadosParam("q", np.zeros((nx, 1)))
        A_param = AcadosParam("A", np.zeros((nx, nx)))
        B_param = AcadosParam("B", np.zeros((nx, nu)))
        b_param = AcadosParam("b", np.zeros((nx, 1)))
        R_param = AcadosParam("R", np.zeros((nu, nu)))
        S_param = AcadosParam("S", np.zeros((nu, nx)))
        r_param = AcadosParam("r", np.zeros((nu, 1)))
        lbu_param = AcadosParam("lbu", np.zeros((nbu, 1)))
        ubu_param = AcadosParam("ubu", np.zeros((nbu, 1)))
        lbx_param = AcadosParam("lbx", np.zeros((nbx_max, 1)))
        ubx_param = AcadosParam("ubx", np.zeros((nbx_max, 1)))
        C_param = AcadosParam("C", np.zeros((nh_max, nx)))
        D_param = AcadosParam("D", np.zeros((nh_max, nu)))
        lg_param = AcadosParam("lg", np.zeros((nh_max, 1)))
        ug_param = AcadosParam("ug", np.zeros((nh_max, 1)))

        param_manager = AcadosParamManager([A_param, B_param, b_param,
                                            Q_param, R_param, S_param, q_param, r_param,
                                            lbx_param, ubx_param, lbu_param, ubu_param,
                                            C_param, D_param, lg_param, ug_param])

        param_manager.N_horizon = N
        param_manager.p_dyna_num = nx*nx + nu*nx + nx
        param_manager.p_cost_num = nx*nx + nu*nu +  nx*nu + nx + nu

        # set up model
        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.p = param_manager.get_p_stagewise_expression()[:param_manager.p_dyna_num + param_manager.p_cost_num]
        ocp.parameter_values = param_manager.get_p_stagewise_values(stage=0)[:param_manager.p_dyna_num + param_manager.p_cost_num]

        # dynamics
        A_expr = param_manager.get_expression("A")
        B_expr = param_manager.get_expression("B")
        b_expr = param_manager.get_expression("b")
        f_expl = ca.mtimes(A_expr, x) + ca.mtimes(B_expr, u) + b_expr
        ocp.model.disc_dyn_expr = f_expl

        # get matrices
        Q = param_manager.get_expression("Q")
        R = param_manager.get_expression("R")
        S = param_manager.get_expression("S")
        q = param_manager.get_expression("q")
        r = param_manager.get_expression("r")

        # set up cost
        x_res = ocp.model.x
        u_res = ocp.model.u
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u_res.T @ R @ u_res + 2 * u_res.T @ S @ x_res) + q.T @ x_res + r.T @ u_res
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res) + q.T @ x_res

        # constraints with placeholders
        # initial state constraints
        ocp.constraints.x0 = np.zeros((nx, 1))
        ocp.constraints.idxbx_0 = self.qp.idxb[0][nbu:] - nbu if nbx_0 > 0 else []
        # state constraints
        ocp.constraints.lbx = -1e4*np.ones((nbx, 1))
        ocp.constraints.ubx = 1e4*np.ones((nbx, 1))
        ocp.constraints.idxbx = self.qp.idxb[1][nbu:] - nbu if nbx > 0 else []
        # terminal state constraints
        ocp.constraints.lbx_e = -1e4*np.ones((nbx_e, 1))
        ocp.constraints.ubx_e = 1e4*np.ones((nbx_e, 1))
        ocp.constraints.idxbx_e = self.qp.idxb[N] if nbx_e > 0 else []
        # control constraints
        ocp.constraints.lbu = -1e4*np.ones((nbu, 1))
        ocp.constraints.ubu = 1e4*np.ones((nbu, 1))
        ocp.constraints.idxbu = self.qp.idxb[0][:nbu] if nbu > 0 else []

        # general linear constraints
        ocp.constraints.C = np.zeros((nh_max, nx))
        ocp.constraints.D = np.zeros((nh_max, nu))
        ocp.constraints.lg = -1e4*np.ones((nh_max, 1))
        ocp.constraints.ug = 1e4*np.ones((nh_max, 1))

        ocp.constraints.C_e = np.zeros((nh_e, nx))
        ocp.constraints.lg_e = -1e4*np.ones((nh_e, 1))
        ocp.constraints.ug_e = 1e4*np.ones((nh_e, 1))
        # NLP solver options
        if self.mode == 'convex':
            ocp.solver_options.globalization = 'FIXED_STEP'  # 'FIXED_STEP', 'MERIT_BACKTRACKING', 'FILTER'
            ocp.solver_options.regularize_method = "NO_REGULARIZE"  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
        else:
            ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'  # 'FIXED_STEP', 'MERIT_BACKTRACKING', 'FILTER'
            ocp.solver_options.regularize_method = 'PROJECT'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'

        ocp.solver_options.tf = N
        ocp.solver_options.N_horizon = N
        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = nlp_max_iter
        ocp.solver_options.eval_residual_at_max_iter = True
        ocp.solver_options.nlp_qp_tol_strategy = 'FIXED_QP_TOL'

        # QP solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
        ocp.solver_options.qp_solver_tol_stat = 1e-6
        ocp.solver_options.qp_solver_tol_eq = 1e-8
        ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.qp_solver_tol_comp = 1e-6
        ocp.solver_options.qp_solver_iter_max = 30
        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_mu0 = 1e1
        ocp.solver_options.tau_min = 1e-16
        ocp.solver_options.hpipm_mode = 'CUSTOMIZED'  # 'DEFAULT', 'CUSTOMIZED'

        return ocp, param_manager
