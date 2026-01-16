from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager
import casadi as ca
import numpy as np
from utils import read_matrix_list


N = 100
json_path = "pendulum_caseF_qp_N=100.json"

def formulate_OCP(regularize_method='NO_REGULARIZE'):
    ocp = AcadosOcp()
    nx = 4
    nu = 1
    # params
    A_param = AcadosParam("A", np.zeros((nx, nx)))
    B_param = AcadosParam("B", np.zeros((nx, nu)))
    Q_param = AcadosParam("Q", np.zeros((nx, nx)))
    R_param = AcadosParam("R", np.zeros((nu, nu)))
    S_param = AcadosParam("S", np.zeros((nu, nx)))

    param_manager = AcadosParamManager([A_param, B_param, Q_param, R_param, S_param])
    param_manager.N_horizon = N

    ocp.model.name = 'parametric_QP_OCP'

    # set up model
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.p = param_manager.get_p_stagewise_expression()

    # dynamics
    A_expr = param_manager.get_expression("A")
    B_expr = param_manager.get_expression("B")
    f_expl = ca.mtimes(A_expr, x) + ca.mtimes(B_expr, u)
    ocp.model.disc_dyn_expr = f_expl

    # get matrices
    Q = param_manager.get_expression("Q")
    R = param_manager.get_expression("R")
    S = param_manager.get_expression("S")

    # set up cost
    x_res = ocp.model.x
    u_res = ocp.model.u
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u_res.T @ R @ u_res + 2 * u_res.T @ S @ x_res)
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q @ x_res)

    ocp.parameter_values = param_manager.get_p_stagewise_values(stage=0)

    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    ocp.constraints.x0 = x0

    uh = 80.0
    lh = -80.0
    # inequality constraintsL: h(x,u) = Cx + Du <= d
    C = np.zeros((1, nx))
    D = np.array([[1.0]])
    ocp.constraints.C = C
    ocp.constraints.D = D
    ocp.constraints.ug = np.array([uh])
    ocp.constraints.lg = np.array([lh])

    # set options
    ocp.solver_options.tf = N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.regularize_method = 'PROJECT'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.qp_solver_mu0 = 2.0
    ocp.solver_options.tau_min = 1e-8
    # ocp.solver_options.nlp_solver_max_iter = 1
    # ocp.solver_options.qp_solver_max_iter = 1
    # ocp.solver_options.print_level = 5
    return ocp, param_manager

def main():
    ocp, param_manager = formulate_OCP()
    A_list = read_matrix_list(json_path , "A")
    B_list = read_matrix_list(json_path, "B")
    Q_list = read_matrix_list(json_path, "Q")
    R_list = read_matrix_list(json_path, "R")
    S_list = read_matrix_list(json_path, "S")
    # Q_list = [np.array([[10.0, 0.0, 0.0, 0.0],
    #             [0.0, 20.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0, 0.0],
    #             [0.0, 0.0, 0.0, 1.0]]) for _ in range(N)]
    # R_list = [np.array([[0.1]]) for _ in range(N)]
    # S_list = [np.zeros((1, 4)) for _ in range(N)]

    for i in range(ocp.solver_options.N_horizon):
        param_manager.set_value(name="A", value=A_list[i], stage=i)
        param_manager.set_value(name="B", value=B_list[i], stage=i)
        param_manager.set_value(name="Q", value=Q_list[i], stage=i)
        param_manager.set_value(name="R", value=R_list[i], stage=i)
        param_manager.set_value(name="S", value=S_list[i], stage=i)
    param_manager.set_value(name="Q", value=Q_list[-1], stage=ocp.solver_options.N_horizon)
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    for i in range(ocp.solver_options.N_horizon + 1):
        param_value = param_manager.get_p_stagewise_values(i)
        ocp_solver.set(i, 'p', param_value)
    status = ocp_solver.solve()
    if status != 0:
        print(f"ocp_solver returned status {status}.")

    cost_acados = ocp_solver.get_cost()
    print(f"\n=== Predicted cost acados ===\n{cost_acados}")

    print("\n=== Predicted next state sequence (x) ===")
    for i in range(N+1):
        x_i = ocp_solver.get(i, "x")
        print(x_i)

    print("\n=== Predicted control sequence (u) ===")
    for i in range(N):
        u_i = ocp_solver.get(i, "u")
        print(u_i)

    ocp_solver.print_statistics()

if __name__ == "__main__":
    main()
