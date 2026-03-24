from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
from Simple_indefinite_case.case_indefinite_mats import QP_mats
from utils import read_matrix_list

N = 20

def formulate_OCP(regularize_method='NO_REGULARIZE'):
    model = AcadosModel()
    model.name = "case_positive"

    # set up model
    nx = 4
    nu = 1

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    model.x = x
    model.u = u

    # dynamics
    A = read_matrix_list("pendulum_caseF_qp.json", "A")[0]
    B = read_matrix_list("pendulum_caseF_qp.json", "B")[0]
    b = np.zeros((nx,))

    f_expl = ca.mtimes(A, x) + ca.mtimes(B, u) + b
    model.disc_dyn_expr = f_expl

    ocp = AcadosOcp()
    ocp.model = model

    # get matrices
    Q = read_matrix_list("pendulum_caseF_qp.json", "Q")[0]
    R = read_matrix_list("pendulum_caseF_qp.json", "R")[0]
    S = read_matrix_list("pendulum_caseF_qp.json", "S")[0]
    q = np.zeros((nx,))
    r = np.zeros((nu,))
    Q_e = read_matrix_list("pendulum_caseF_qp.json", "Q")[-1]
    q_e = np.zeros((nx,))
    # set up cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = ca.diagcat(Q, R).full()
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros((nx + nu,))
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.W_e = Q_e
    ocp.cost.yref_e = np.zeros((nx,))

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
    ocp.solver_options.tf = 1.0*N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.nlp_solver_tol_stat = 1e-5
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.nlp_solver_max_iter = 1
    # ocp.solver_options.qp_solver_iter_max = 0
    # ocp.solver_options.print_level = 5
    return ocp

def main():
    ocp = formulate_OCP()
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    status = ocp_solver.solve()
    ocp_solver.dump_last_qp_to_json("qp_json/acados_ocp_indefinite_qp.json")

    # if status != 0:
    #     ocp_solver.print_statistics()
    #     raise Exception(f"Acados returned status {status}.")

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

    # for i in range(N+1):
    #     H_i = ocp_solver.get_hessian_block(i)
    #     print(f"\n=== Hessian block at stage {i} ===\n{H_i}")

if __name__ == "__main__":
    main()
