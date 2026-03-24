'''

General template for getting Hessian from iterate of acados solver and checking its positive definiteness

RTI enables extract complete Hessian matrix from each iteration, while SQP only extracts from the final solution

the positiveness of Hessian is checked by looking at the eigenvalues of each block on the diagonal (corresponding to each stage)

'''
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from pendulum_model import export_pendulum_ode_model
import numpy as np
from utils import plot_pendulum, check_positive_definite

PLOT = True
REGULARIZATION = 'NO_REGULARIZE'  # 'NO_REGULARIZE', 'PROJECT', 'CONVEXIFY', 'MIRROR'

def formulate_ocp(N=20, Tf=1.0, Fmax=80):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    # set cost
    Q = np.diag([1e1, 1e1, 1e1, 1e1])
    R = np.diag([1e0])

    # # nonlinear
    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.zeros((nx+nu,))
    ocp.cost.W = ca.diagcat(Q, R).full()

    # terminal cost
    Q_e = np.diag([1e1, 1e1, 1e1, 1e1])
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.zeros((nx,))
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_e

    # set constraints
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.remove_x0_elimination()

    # set options
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.store_iterates = True
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.qp_solver_ric_alg  = 0
    ocp.solver_options.reg_epsilon = 1e-4
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.regularize_method = REGULARIZATION  # 'NO_REGULARIZATION', 'PROJECT', 'CONVEXIFY', 'MIRROR'
    ocp.solver_options.exact_hess_dyn = False
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'

    return ocp

def main():
    N = 20
    dt = 0.1
    Tf = N*dt
    Fmax = 80
    ocp = formulate_ocp(N=N, Tf=Tf, Fmax=Fmax)

    ocp_solver = AcadosOcpSolver(ocp,verbose=False)

    # call SQP_RTI solver in the loop:
    status = ocp_solver.solve()
    ocp_solver.print_statistics()
    print('solver status = ', status)
    cost = ocp_solver.get_cost()
    print("cost function value of solution = ", cost)
    ocp_solver.dump_last_qp_to_json('qp_json/pendulum_Hessian.json')

    check_positive_definite(ocp_solver)

    if PLOT:
    # plot
        simX = np.zeros((N+1, ocp.dims.nx))
        simU = np.zeros((N, ocp.dims.nu))
        for i in range(N):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
        simX[N,:] = ocp_solver.get(N, "x")
        plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=False)

if __name__ == "__main__":
    main()