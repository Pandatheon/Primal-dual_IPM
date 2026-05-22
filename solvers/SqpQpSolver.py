from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager, AcadosOcpQp
import casadi as ca
import numpy as np

class SqpQpSolver(AcadosOcpSolver):
    def __init__(self, qp: AcadosOcpQp, reg = 'NO_REGULARIZE', qp_max_iter=15):
        self.qp = qp
        parametric_ocp, self.param_manager = self.formulate_parametric_OCP_QP(reg, qp_max_iter)

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
        nh_e = self.qp.C[-1].shape[0]
        
        # Set parameters value in param_manager
        for i in range(N):
            self.param_manager.set_value(name="A", value=self.qp.A[i], stage=i)
            self.param_manager.set_value(name="B", value=self.qp.B[i], stage=i)
            self.param_manager.set_value(name="b", value=self.qp.b[i], stage=i)
            self.param_manager.set_value(name="Q", value=self.qp.Q[i], stage=i)
            self.param_manager.set_value(name="R", value=self.qp.R[i], stage=i)
            self.param_manager.set_value(name="S", value=self.qp.S[i], stage=i)
            self.param_manager.set_value(name="q", value=self.qp.q[i], stage=i)
            self.param_manager.set_value(name="r", value=self.qp.r[i], stage=i)
            self.param_manager.set_value(name="lbu", value=self.qp.lbu[i], stage=i)
            self.param_manager.set_value(name="ubu", value=self.qp.ubu[i], stage=i)
            if i == 0:
                self.param_manager.set_value(name="lbx", value=self.qp.lbx[i], stage=i)
                self.param_manager.set_value(name="ubx", value=self.qp.ubx[i], stage=i)

            if i == 0 and (nh_0 or nh):
                C_0 = self.qp.C[0] if nh_0 else np.zeros((1, nx))
                D_0 = self.qp.D[0] if nh_0 else np.zeros((1, nu))
                lg_0 = self.qp.lg[0] if nh_0 else -1e4*np.ones((1, 1))
                ug_0 = self.qp.ug[0] if nh_0 else 1e4*np.ones((1, 1))
                self.param_manager.set_value(name="C", value=C_0, stage=i)
                self.param_manager.set_value(name="D", value=D_0, stage=i)
                self.param_manager.set_value(name="lg", value=lg_0, stage=i)
                self.param_manager.set_value(name="ug", value=ug_0, stage=i)
            elif nh and i > 0:
                self.param_manager.set_value(name="C", value=self.qp.C[i], stage=i)
                self.param_manager.set_value(name="D", value=self.qp.D[i], stage=i)
                self.param_manager.set_value(name="lg", value=self.qp.lg[i], stage=i)
                self.param_manager.set_value(name="ug", value=self.qp.ug[i], stage=i)

        self.param_manager.set_value(name="Q", value=self.qp.Q[-1], stage=N)
        self.param_manager.set_value(name="q", value=self.qp.q[-1], stage=N)

        if nh_e:
            self.param_manager.set_value(name="C", value=self.qp.C[-1], stage=N)
            self.param_manager.set_value(name="lg", value=self.qp.lg[-1], stage=N)
            self.param_manager.set_value(name="ug", value=self.qp.ug[-1], stage=N)


        # Set parameters value in the solver
        for i in range(N):
            param_value = self.param_manager.get_p_stagewise_values(i)
            param_dyna_cost_values = param_value[:self.param_manager.p_dyna_num + self.param_manager.p_cost_num]
            param_cons_values = param_value[self.param_manager.p_dyna_num + self.param_manager.p_cost_num:]

            # Set dynamic and cost parameters
            self.set(i, 'p', param_dyna_cost_values)

            # Set constraints
            end_bx = 2 * nx
            lbu = param_cons_values[end_bx:end_bx+nu].flatten()
            ubu = param_cons_values[end_bx+nu:end_bx+2*nu].flatten()

            self.constraints_set(i, 'lbu', lbu)
            self.constraints_set(i, 'ubu', ubu)

            end_nb = end_bx + 2*nu
            if i == 0 and (nh_0 or nh):
                nh_0 = nh_0 if nh_0 else 1
                C_k = param_cons_values[end_nb:end_nb+nh_0*nx].reshape((nh_0, nx))
                D_k = param_cons_values[end_nb+nh_0*nx:end_nb+nh_0*nx+nh_0*nu].reshape((nh_0, nu))
                end_ng = end_nb + nh_0*nx + nh_0*nu
                lg = param_cons_values[end_ng:end_ng+nh_0].flatten()
                ug = param_cons_values[end_ng+nh_0:end_ng+2*nh_0].flatten()
                self.constraints_set(i, 'C', C_k)
                self.constraints_set(i, 'D', D_k)
                self.constraints_set(i, 'lg', lg)
                self.constraints_set(i, 'ug', ug)
            elif i > 0 and nh:
                C_k = param_cons_values[end_nb:end_nb+nh*nx].reshape((nh, nx))
                D_k = param_cons_values[end_nb+nh*nx:end_nb+nh*nx+nh*nu].reshape((nh, nu))
                end_ng = end_nb + nh*nx + nh*nu
                lg = param_cons_values[end_ng:end_ng+nh].flatten()
                ug = param_cons_values[end_ng+nh:end_ng+2*nh].flatten()
                self.constraints_set(i, 'C', C_k)
                self.constraints_set(i, 'D', D_k)
                self.constraints_set(i, 'lg', lg)
                self.constraints_set(i, 'ug', ug)

            if i == 0:
                lbx = param_cons_values[:nx].flatten()
                ubx = param_cons_values[nx:2*nx].flatten()
                self.constraints_set(i, 'lbx', lbx)
                self.constraints_set(i, 'ubx', ubx)

        param_value_N = self.param_manager.get_p_stagewise_values(N)
        param_dyna_cost_values_N = param_value_N[:self.param_manager.p_dyna_num + self.param_manager.p_cost_num]
        # Set terminal node parameters
        self.set(N, 'p', param_dyna_cost_values_N)

        if nh_e:
            C_e = param_cons_values[end_nb:end_nb+nh_e*nx].reshape((nh_e, nx))
            lg_e = param_cons_values[end_nb+nh_e*nx:end_nb+nh_e*nx+nh_e].flatten()
            ug_e = param_cons_values[end_nb+nh_e*nx+nh_e:end_nb+nh_e*nx+2*nh_e].flatten()
            self.constraints_set(N, 'C', C_e)
            self.constraints_set(N, 'lg', lg_e)
            self.constraints_set(N, 'ug', ug_e)

    def formulate_parametric_OCP_QP(self, reg, qp_max_iter):
        ocp = AcadosOcp()
        ocp.model.name = 'parametric_OCP_QP'

        N = self.qp.N
        nx = self.qp.A[0].shape[0]
        nu = self.qp.B[0].shape[1]

        nh_0 = self.qp.C[0].shape[0]
        nh = self.qp.C[1].shape[0]
        nh_e = self.qp.C[-1].shape[0]

        # params
        A_param = AcadosParam("A", np.zeros((nx, nx)))
        B_param = AcadosParam("B", np.zeros((nx, nu)))
        b_param = AcadosParam("b", np.zeros((nx, 1)))
        Q_param = AcadosParam("Q", np.zeros((nx, nx)))
        R_param = AcadosParam("R", np.zeros((nu, nu)))
        S_param = AcadosParam("S", np.zeros((nu, nx)))
        q_param = AcadosParam("q", np.zeros((nx, 1)))
        r_param = AcadosParam("r", np.zeros((nu, 1)))
        lbx_param = AcadosParam("lbx", np.zeros((nx, 1)))
        ubx_param = AcadosParam("ubx", np.zeros((nx, 1)))
        lbu_param = AcadosParam("lbu", np.zeros((nu, 1)))
        ubu_param = AcadosParam("ubu", np.zeros((nu, 1)))

        nh_max = max(nh_0, nh, nh_e)
        if nh_max:
            C_param = AcadosParam("C", np.zeros((nh_max, nx)))
            D_param = AcadosParam("D", np.zeros((nh_max, nu)))
            lg_param = AcadosParam("lg", np.zeros((nh_max, 1)))
            ug_param = AcadosParam("ug", np.zeros((nh_max, 1)))
            param_manager = AcadosParamManager([A_param, B_param, b_param,
                                                Q_param, R_param, S_param, q_param, r_param,
                                                lbx_param, ubx_param, lbu_param, ubu_param,
                                                C_param, D_param, lg_param, ug_param])
        else:
            param_manager = AcadosParamManager([A_param, B_param, b_param,
                                                Q_param, R_param, S_param, q_param, r_param,
                                                lbx_param, ubx_param, lbu_param, ubu_param])

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
        ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
        # control constraints
        ocp.constraints.lbu = -np.zeros((nu, 1))
        ocp.constraints.ubu = np.zeros((nu, 1))
        ocp.constraints.idxbu = np.arange(nu).tolist()

        if nh:
            ocp.constraints.C = np.zeros((nh, nx))
            ocp.constraints.D = np.zeros((nh, nu))
            ocp.constraints.lg = np.zeros((nh, 1))
            ocp.constraints.ug = np.zeros((nh, 1))
        elif nh_e:
            ocp.constraints.C_e = np.zeros((nh_e, nx))
            ocp.constraints.lg_e = np.zeros((nh_e, 1))
            ocp.constraints.ug_e = np.zeros((nh_e, 1))

        # NLP solver options
        ocp.solver_options.tf = N
        ocp.solver_options.N_horizon = N
        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.regularize_method = reg  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = 30
        ocp.solver_options.eval_residual_at_max_iter = True
        ocp.solver_options.nlp_qp_tol_strategy = 'FIXED_QP_TOL'

        # QP solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
        ocp.solver_options.qp_solver_tol_stat = 1e-6
        ocp.solver_options.qp_solver_tol_eq = 1e-8
        ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.qp_solver_tol_comp = 1e-8

        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_mu0 = 1e1
        ocp.solver_options.tau_min = 1e-16
        ocp.solver_options.qp_solver_iter_max = qp_max_iter
        ocp.solver_options.hpipm_mode = 'CUSTOMIZED'  # 'DEFAULT', 'CUSTOMIZED'

        return ocp, param_manager
