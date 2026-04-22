import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import importlib
import copy

from utilis.utilis import write_csv_acados

case_modules = [
    'M_pos_R_pos_P_inde',
    # 'M_pos_R_inde_P_pos',
    # "M_inde_R_inde_P_pos_1",
    # "M_inde_R_inde_P_pos_2",
    # "M_inde_R_inde_P_pos_3",
    # "M_inde_R_inde_P_pos_4",
    # "M_inde_R_pos_P_inde_1",
    # "M_inde_R_pos_P_inde_2",
    # "M_inde_R_pos_P_inde_3",
    # 'M_inde_R_pos_P_inde_4',
    # "M_inde_R_inde_P_inde_2",
    # "M_inde_R_inde_P_inde_3",
    # 'M_inde_R_inde_P_inde_4',
]

def formulate_OCP(mats):
    model = AcadosModel()
    model.name = "case_indefinite"

    # set up model
    nx = mats.A.shape[0]
    nu = mats.B.shape[1]
    N = mats.N

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    model.x = x
    model.u = u

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
    q = mats.q.reshape(-1,1)
    r = mats.r.reshape(-1,1)
    Q_e = mats.Q_e
    q_e = mats.q_e.reshape(-1,1)
    # set up cost
    x_res = ocp.model.x
    u_res = ocp.model.u
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = 0.5 * (x_res.T @ Q @ x_res + u_res.T @ R @ u_res + 2 * x_res.T @ S.T @ u_res ) + q.T @ x_res + r.T @ u_res
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = 0.5 * (x_res.T @ Q_e @ x_res) + q_e.T @ x_res

    x0 = 3.0 * np.ones((nx,))
    ocp.constraints.x0 = x0

    uh = mats.d[nu+1]
    lh = -mats.d[nu+0]
    # inequality constraintsL: h(x,u) = Cx + Du <= d
    ocp.constraints.lbu = np.array([lh, lh])
    ocp.constraints.ubu = np.array([uh, uh])
    ocp.constraints.idxbu = np.array([0, 1])  # first two inputs are bounded

    # set options
    ocp.solver_options.tf = 1.0*N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.regularize_method = 'PROJECT'  # 'NO_REGULARIZE', 'PROJECT', 'MIRROR', 'CONVEXIFY'
    ocp.solver_options.reg_epsilon = 1e-4
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.eval_residual_at_max_iter = True
    return ocp

def main():
    for module_name in case_modules:
        print(f"\n>>> Running Case: {module_name}")

        mod = importlib.import_module(module_name)
        QP_mats = getattr(mod, "QP_mats") 
        # --------------------

        mats = QP_mats()
        
        if not hasattr(mats, 'name'):
            mats.name = module_name
        try:
            ocp = formulate_OCP(mats)
            ocp_solver = AcadosOcpSolver(ocp, verbose=False)
            status = ocp_solver.solve()
            if status != 0:
                print(f"  ACADOS solver failed with status {status}")
                continue
            cost_acados = ocp_solver.get_cost()
            print(f"=== Predicted cost: {cost_acados} ===")
            # Write results to CSV
            write_csv_acados(mats.name, 'ACADOS_PROJECT', cost_acados)
            # print('=== Optimal solution x at each step: ===')
            # for k in range(mats.N):
            #     x_k = ocp_solver.get(k, 'x')
            #     print(f" Step {k}: x = {x_k}")
            # print('=== Optimal solution u at each step: ===')
            # for k in range(mats.N):
            #     u_k = ocp_solver.get(k, 'u')
            #     print(f" Step {k}: u = {u_k}")
            ocp_solver.print_statistics()
            ocp = None
            ocp_solver = None
        except Exception as e:
            print(f"  Failed: {e}")
            continue
if __name__ == "__main__":
    main()
