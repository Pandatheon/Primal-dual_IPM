from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np

N = 5

def formulate_OCP(regularize_method='NO_REGULARIZE'):
    model = AcadosModel()
    model.name = "case_positive"

    # set up model
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    model.x = x
    model.u = u
    nx = x.shape[0]
    nu = u.shape[0]

    # dynamics
    A = np.array([[1.0, 0.0],
                  [0.0, 2.0]])
    B = np.array([[1.0],
                  [2.0]])
    b = np.array([0.0, 0.0])

    f_expl = ca.mtimes(A, x) + ca.mtimes(B, u) + b
    model.disc_dyn_expr = f_expl

    ocp = AcadosOcp()
    ocp.model = model

    # set up cost
    Q = np.diag(np.array([1.0, 1.0]))
    R = np.diag(np.array([1.0]))
    S = np.zeros((1, 2))
    q = np.zeros((2,))
    r = np.zeros((1,))
    # terminal cost
    Q_e = Q
    q_e = np.zeros((2,))
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = ca.diagcat(Q, R).full()
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros((nx + nu,))
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.W_e = Q_e
    ocp.cost.yref_e = np.zeros((nx,))

    x0 = np.array([3.0, 3.0])
    ocp.constraints.x0 = x0
    uh = 2.0
    # inequality constraintsL: h(x,u) = Cx + Du <= d
    ocp.constraints.lbu = np.array([-uh])
    ocp.constraints.ubu = np.array([uh])
    ocp.constraints.idxbu = np.array([0])

    # set options
    ocp.solver_options.tf = 1.0*N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.store_iterates = True
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    return ocp

def main():
    ocp = formulate_OCP()
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    status = ocp_solver.solve()

    if status != 0:
        raise Exception(f"Acados returned status {status}.")

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
