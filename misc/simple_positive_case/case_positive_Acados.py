from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
from case_positive_mats import QP_mats

mats = QP_mats()

N = mats.N

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
    A = mats.A
    B = mats.B
    b = mats.b

    f_expl = ca.mtimes(A, x) + ca.mtimes(B, u) + b
    model.disc_dyn_expr = f_expl

    ocp = AcadosOcp()
    ocp.model = model

    # get matrices
    Q = mats.Q
    R = mats.R
    S = mats.S
    q = mats.q
    r = mats.r
    Q_e = mats.Q_e
    q_e = mats.q_e
    # set up cost
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

    uh = mats.d[1]
    lh = -mats.d[0]
    # inequality constraintsL: h(x,u) = Cx + Du <= d
    C = mats.C[1:2, :]
    D = mats.D[1:2, :]
    ocp.constraints.C = C
    ocp.constraints.D = D
    ocp.constraints.ug = np.array([uh])
    ocp.constraints.lg = np.array([lh])

    # set options
    ocp.solver_options.tf = 1.0*N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.store_iterates = True
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    # ocp.solver_options.regularize_method = 'NO_REGULARIZE'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    return ocp

def main():
    ocp = formulate_OCP()
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    status = ocp_solver.solve()
    ocp_solver.dump_last_qp_to_json("acados_ocp_qp.json")

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
