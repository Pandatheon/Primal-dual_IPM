# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager, AcadosOcpQp
import casadi as ca
import numpy as np

import json

def formulate_parametric_OCP_QP(qp:AcadosOcpQp):
    ocp = AcadosOcp()
    ocp.model.name = 'parametric_OCP_QP'

    N = qp.N
    nx = qp.A[0].shape[0]
    nu = qp.B[0].shape[1]

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
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])
    ocp.constraints.lbu = -np.zeros((nu, 1))
    ocp.constraints.ubu = np.zeros((nu, 1))
    ocp.constraints.idxbu = np.arange(nu).tolist()

    # NLP solver options
    ocp.solver_options.tf = N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    # ocp.solver_options.regularize_method = 'PROJECT'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_max_iter = 100
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
    ocp.solver_options.qp_solver_iter_max = 30
    ocp.solver_options.hpipm_mode = 'CUSTOMIZED'  # 'DEFAULT', 'CUSTOMIZED'

    return ocp, param_manager

json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Pos_QP_json/mock_pendulum_qp_data_000.json'

def main():
    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    ocp, param_manager = formulate_parametric_OCP_QP(qp)
    N = qp.N
    nx = qp.A[0].shape[0]
    nu = qp.B[0].shape[1]
    for i in range(qp.N):
        param_manager.set_value(name="A", value=qp.A[i], stage=i)
        param_manager.set_value(name="B", value=qp.B[i], stage=i)
        param_manager.set_value(name="b", value=qp.b[i], stage=i)
        param_manager.set_value(name="Q", value=qp.Q[i], stage=i)
        param_manager.set_value(name="R", value=qp.R[i], stage=i)
        param_manager.set_value(name="S", value=qp.S[i], stage=i)
        param_manager.set_value(name="q", value=qp.q[i], stage=i)
        param_manager.set_value(name="r", value=qp.r[i], stage=i)
        if i == 0:
            param_manager.set_value(name="lbx", value=qp.lbx[i], stage=i)
            param_manager.set_value(name="ubx", value=qp.ubx[i], stage=i)
        param_manager.set_value(name="lbu", value=qp.lbu[i], stage=i)
        param_manager.set_value(name="ubu", value=qp.ubu[i], stage=i)

    param_manager.set_value(name="Q", value=qp.Q[-1], stage=N)
    param_manager.set_value(name="q", value=qp.q[-1], stage=N)

    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    for i in range(ocp.solver_options.N_horizon):
        param_value = param_manager.get_p_stagewise_values(i)
        param_dyna_cost_values = param_value[:param_manager.p_dyna_num + param_manager.p_cost_num]
        param_cons_values = param_value[param_manager.p_dyna_num + param_manager.p_cost_num:]

        # dynamic and cost
        ocp_solver.set(i, 'p', param_dyna_cost_values)

        # constraints
        end_bx = 2*nx
        lbu = param_cons_values[end_bx:end_bx+nu].flatten()
        ubu = param_cons_values[end_bx+nu:end_bx+2*nu].flatten()

        ocp_solver.constraints_set(i, 'lbu', lbu)
        ocp_solver.constraints_set(i, 'ubu', ubu)

        if i == 0:
            lbx = param_cons_values[:nx].flatten()
            ubx = param_cons_values[nx:2*nx].flatten()
            ocp_solver.constraints_set(i, 'lbx', lbx)
            ocp_solver.constraints_set(i, 'ubx', ubx)

    param_value_N = param_manager.get_p_stagewise_values(N)
    param_dyna_cost_values_N = param_value_N[:param_manager.p_dyna_num + param_manager.p_cost_num]
    ocp_solver.set(N, 'p', param_dyna_cost_values_N)

    status = ocp_solver.solve()
    if status != 0:
        print(f"ocp_solver returned status {status}.")
    ocp_solver.print_statistics()
    cost = ocp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    x = [ocp_solver.get(k, "x") for k in range(N+1)]
    u = [ocp_solver.get(k, "u") for k in range(N)]
    lam = [ocp_solver.get(k, "lam") for k in range(N)]
    for k in range(N):
        print(f"lam[{k}] = {lam[k]}")

if __name__ == "__main__":
    main()