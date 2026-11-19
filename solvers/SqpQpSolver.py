from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager, AcadosOcpQp, AcadosOcpQpOptions, AcadosOcpQpHpipmOptions

import casadi as ca
import numpy as np
import hashlib
import os


class SqpQpSolver(AcadosOcpSolver):
    """
    Solve an AcadosOcpQp with the acados SQP solver.

    The QP data enters a parametric OCP: A, B, b, Q, R, S, q, r, C, D are model parameters and the
    bounds are set directly. Starting from the zero iterate, the first SQP iteration solves exactly
    the given QP, so with the same qp_opts it reproduces a standalone AcadosOcpQpSolver (HPIPM) solve.
    General constraints are modeled as h_0 / h / h_e, since acados only allows a stage-0 specific
    number of constraints for h, and the QP may have e.g. ng = 0 at stage 0 but ng > 0 afterwards.
    """

    def __init__(self, qp: AcadosOcpQp, nlp_max_iter=100, mode='nonconvex', qp_opts: AcadosOcpQpOptions = None):

        self.solver_created = False  # AcadosOcpSolver.__del__ needs it if the checks below raise
        self.qp = qp
        self.mode = mode
        self.qp_opts = qp_opts if qp_opts is not None else self.default_qp_opts()
        self._check_qp_structure()
        parametric_ocp, self.param_manager = self.formulate_parametric_OCP_QP(nlp_max_iter)
        super().__init__(parametric_ocp, verbose=False)

    @staticmethod
    def default_qp_opts() -> AcadosOcpQpHpipmOptions:
        opts = AcadosOcpQpHpipmOptions()
        opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        opts.hpipm_mode = 'BALANCE'
        opts.mu0 = 1e1
        opts.t0_init = 2
        opts.warm_start = 0
        opts.ric_alg = 0
        opts.cond_ric_alg = 0
        opts.tol_stat = 1e-6
        opts.tol_eq = 1e-8
        opts.tol_ineq = 1e-8
        opts.tol_comp = 1e-6
        opts.iter_max = 30
        return opts

    def solve(self):
        # start from the zero iterate, where the first SQP iteration solves exactly the given QP
        self.reset()
        self._populate_ocp_solver()
        status = super().solve()
        if status != 0:
            print(f"ocp_solver returned status {status}.")
        return status

    def _check_qp_structure(self):
        qp = self.qp
        dims = qp.dims
        N = qp.N

        if len(set(dims.nx)) != 1:
            raise ValueError(f"SqpQpSolver requires constant nx, got nx = {dims.nx}.")
        if len(set(dims.nu[:N])) != 1 or dims.nu[N] != 0:
            raise ValueError(f"SqpQpSolver requires constant nu for stages 0..N-1 and nu = 0 at stage N, got nu = {dims.nu}.")
        if any(dims.ns > 0):
            raise NotImplementedError("SqpQpSolver does not support slack variables.")
        if qp.has_masks():
            raise NotImplementedError("SqpQpSolver does not support masked constraints.")

        idxbu_0 = qp.idxb[0][:dims.nbu[0]]
        for i in range(1, N):
            if not np.array_equal(qp.idxb[i][:dims.nbu[i]], idxbu_0):
                raise ValueError(f"SqpQpSolver requires the same u bounds on stages 0..N-1, stage {i} differs from stage 0.")
            if not np.array_equal(qp.idxb[i], qp.idxb[1]) or dims.ng[i] != dims.ng[1]:
                raise ValueError(f"SqpQpSolver requires the same constraint structure on stages 1..N-1, stage {i} differs from stage 1.")
            if dims.nbxe[i] > 0:
                raise NotImplementedError(f"SqpQpSolver supports equality bounds (idxe) only at stage 0, got idxe at stage {i}.")
        if dims.nbxe[N] > 0:
            raise NotImplementedError(f"SqpQpSolver supports equality bounds (idxe) only at stage 0, got idxe at stage {N}.")

    def _idxbx(self, stage):
        # idxb indexes [u; x], the first nbu entries are the u bounds
        return self.qp.idxb[stage][self.qp.dims.nbu[stage]:] - self.qp.dims.nu[stage]

    def _qp_solver_options(self) -> dict:
        # map AcadosOcpQpOptions onto the QP solver options of the OCP solver
        opts = self.qp_opts
        solver_options = {
            'qp_solver': opts.qp_solver,
            'hpipm_mode': opts.hpipm_mode,
            'qp_solver_mu0': opts.mu0 if opts.mu0 is not None else 0.0,  # 0.0 -> HPIPM default
            'qp_solver_t0_init': opts.t0_init,
            'qp_solver_ric_alg': opts.ric_alg,
            'qp_solver_cond_ric_alg': opts.cond_ric_alg,
            'qp_solver_tol_stat': opts.tol_stat,
            'qp_solver_tol_eq': opts.tol_eq,
            'qp_solver_tol_ineq': opts.tol_ineq,
            'qp_solver_tol_comp': opts.tol_comp,
            'qp_solver_iter_max': opts.iter_max,
            'qp_solver_warm_start': opts.warm_start,
        }
        if opts.cond_N is not None:
            solver_options['qp_solver_cond_N'] = opts.cond_N
        if opts.cond_block_size is not None:
            solver_options['qp_solver_cond_block_size'] = opts.cond_block_size
        return solver_options

    def _model_name(self, nlp_max_iter, qp_solver_options):
        # The generated code only depends on the QP structure and the options. Using a distinct name and
        # export directory per structure prevents linking stale objects of a previously generated QP.
        dims = self.qp.dims
        signature = repr((dims.nx.tolist(), dims.nu.tolist(), dims.nbu.tolist(), dims.nbx.tolist(), dims.ng.tolist(),
                          [idxb.tolist() for idxb in self.qp.idxb],
                          [[] if idxe is None else idxe.tolist() for idxe in self.qp.idxe],
                          self.mode, nlp_max_iter, sorted((k, repr(v)) for k, v in qp_solver_options.items())))
        return 'parametric_OCP_QP_' + hashlib.md5(signature.encode()).hexdigest()[:10]

    def _populate_ocp_solver(self):
        qp = self.qp
        dims = qp.dims
        N = qp.N
        nx = dims.nx[0]
        nu = dims.nu[0]
        ng_max = max(dims.ng)

        for i in range(N+1):
            ng = dims.ng[i]

            # Set parameters value in param_manager
            self.param_manager.set_value(name="Q", value=qp.Q[i], stage=i)
            self.param_manager.set_value(name="q", value=qp.q[i].reshape((nx, 1)), stage=i)
            if i < N:
                self.param_manager.set_value(name="A", value=qp.A[i], stage=i)
                self.param_manager.set_value(name="B", value=qp.B[i], stage=i)
                self.param_manager.set_value(name="b", value=qp.b[i].reshape((nx, 1)), stage=i)
                self.param_manager.set_value(name="R", value=qp.R[i], stage=i)
                self.param_manager.set_value(name="S", value=qp.S[i], stage=i)
                self.param_manager.set_value(name="r", value=qp.r[i].reshape((nu, 1)), stage=i)

            # only the first ng rows of C, D enter h at this stage
            C_value = np.zeros((ng_max, nx))
            D_value = np.zeros((ng_max, nu))
            if ng > 0:
                C_value[:ng, :] = qp.C[i]
                if i < N:
                    D_value[:ng, :] = qp.D[i]
            self.param_manager.set_value(name="C", value=C_value, stage=i)
            self.param_manager.set_value(name="D", value=D_value, stage=i)

            self.set(i, 'p', self.param_manager.get_p_stagewise_values(i).flatten())

            # Set bounds
            if dims.nbu[i] > 0:
                self.constraints_set(i, 'lbu', qp.lbu[i])
                self.constraints_set(i, 'ubu', qp.ubu[i])
            if dims.nbx[i] > 0:
                self.constraints_set(i, 'lbx', qp.lbx[i])
                self.constraints_set(i, 'ubx', qp.ubx[i])
            if ng > 0:
                self.constraints_set(i, 'lh', qp.lg[i])
                self.constraints_set(i, 'uh', qp.ug[i])

    def formulate_parametric_OCP_QP(self, nlp_max_iter):
        ocp = AcadosOcp()
        qp_solver_options = self._qp_solver_options()
        ocp.model.name = self._model_name(nlp_max_iter, qp_solver_options)
        ocp.code_gen_opts.code_export_directory = os.path.join('c_generated_code', ocp.model.name)
        ocp.code_gen_opts.json_file = os.path.join(ocp.code_gen_opts.code_export_directory, f'{ocp.model.name}_ocp.json')
        os.makedirs(ocp.code_gen_opts.code_export_directory, exist_ok=True)

        qp = self.qp
        dims = qp.dims
        N = qp.N
        nx = dims.nx[0]
        nu = dims.nu[0]

        nbx_0 = dims.nbx[0]
        nbx = dims.nbx[1] if N > 1 else 0
        nbx_e = dims.nbx[N]
        nbu = dims.nbu[0]

        ng_0 = dims.ng[0]
        ng = dims.ng[1] if N > 1 else 0
        ng_e = dims.ng[N]
        ng_max = max(dims.ng)

        # params
        Q_param = AcadosParam("Q", np.zeros((nx, nx)))
        q_param = AcadosParam("q", np.zeros((nx, 1)))
        A_param = AcadosParam("A", np.zeros((nx, nx)))
        B_param = AcadosParam("B", np.zeros((nx, nu)))
        b_param = AcadosParam("b", np.zeros((nx, 1)))
        R_param = AcadosParam("R", np.zeros((nu, nu)))
        S_param = AcadosParam("S", np.zeros((nu, nx)))
        r_param = AcadosParam("r", np.zeros((nu, 1)))
        C_param = AcadosParam("C", np.zeros((ng_max, nx)))
        D_param = AcadosParam("D", np.zeros((ng_max, nu)))

        param_manager = AcadosParamManager([A_param, B_param, b_param,
                                            Q_param, R_param, S_param, q_param, r_param,
                                            C_param, D_param])
        param_manager.N_horizon = N

        # set up model
        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.p = param_manager.get_p_stagewise_expression()
        ocp.parameter_values = param_manager.get_p_stagewise_values(stage=0).flatten()

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
        C = param_manager.get_expression("C")
        D = param_manager.get_expression("D")

        # set up cost
        x_res = ocp.model.x
        u_res = ocp.model.u
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u_res.T @ R @ u_res + 2 * u_res.T @ S @ x_res) + q.T @ x_res + r.T @ u_res
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res) + q.T @ x_res

        # bounds with placeholders, values are set in _populate_ocp_solver
        # initial state constraints, the equalities (idxe) index the bounds [u; x]
        ocp.constraints.idxbx_0 = self._idxbx(0)
        ocp.constraints.lbx_0 = -1e5*np.ones((nbx_0,))
        ocp.constraints.ubx_0 = 1e5*np.ones((nbx_0,))
        ocp.constraints.idxbxe_0 = qp.idxe[0] - nbu if dims.nbxe[0] > 0 else np.array([])
        # state constraints
        if nbx > 0:
            ocp.constraints.idxbx = self._idxbx(1)
            ocp.constraints.lbx = -1e5*np.ones((nbx,))
            ocp.constraints.ubx = 1e5*np.ones((nbx,))
        # terminal state constraints
        if nbx_e > 0:
            ocp.constraints.idxbx_e = self._idxbx(N)
            ocp.constraints.lbx_e = -1e5*np.ones((nbx_e,))
            ocp.constraints.ubx_e = 1e5*np.ones((nbx_e,))
        # control constraints
        if nbu > 0:
            ocp.constraints.idxbu = qp.idxb[0][:nbu]
            ocp.constraints.lbu = -1e5*np.ones((nbu,))
            ocp.constraints.ubu = 1e5*np.ones((nbu,))

        # general linear constraints lg <= C x + D u <= ug, stage-wise number of rows
        if ng_0 > 0:
            ocp.model.con_h_expr_0 = C[:ng_0, :] @ x + D[:ng_0, :] @ u
            ocp.constraints.lh_0 = -1e5*np.ones((ng_0,))
            ocp.constraints.uh_0 = 1e5*np.ones((ng_0,))
        if ng > 0:
            ocp.model.con_h_expr = C[:ng, :] @ x + D[:ng, :] @ u
            ocp.constraints.lh = -1e5*np.ones((ng,))
            ocp.constraints.uh = 1e5*np.ones((ng,))
        if ng_e > 0:
            ocp.model.con_h_expr_e = C[:ng_e, :] @ x
            ocp.constraints.lh_e = -1e5*np.ones((ng_e,))
            ocp.constraints.uh_e = 1e5*np.ones((ng_e,))

        # QP solver options, same as for a standalone AcadosOcpQpSolver with qp_opts
        for field, value in qp_solver_options.items():
            setattr(ocp.solver_options, field, value)

        # NLP solver options
        if self.mode == 'convex':
            ocp.solver_options.regularize_method = "NO_REGULARIZE"  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
            ocp.solver_options.hpipm_mode = "CUSTOMIZED"
        else:
            ocp.solver_options.regularize_method = 'PROJECT'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
        ocp.solver_options.reg_epsilon = 1e-4

        ocp.solver_options.globalization = 'FIXED_STEP'
        ocp.solver_options.tf = N
        ocp.solver_options.N_horizon = N
        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = nlp_max_iter
        ocp.solver_options.eval_residual_at_max_iter = True
        ocp.solver_options.nlp_qp_tol_strategy = 'FIXED_QP_TOL'

        ocp.solver_options.nlp_solver_tol_stat = 1e-6
        ocp.solver_options.nlp_solver_tol_eq = 1e-8
        ocp.solver_options.nlp_solver_tol_ineq = 1e-8
        ocp.solver_options.nlp_solver_tol_comp = 1e-6

        return ocp, param_manager
