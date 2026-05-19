# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager
import casadi as ca
import numpy as np

import json

class QP_From_JSON:
    def __init__(self, json_path):
        matrix_fields = ['Q', 'R', 'S', 'A', 'B', 'C', 'D']
        vector_fields = ['q', 'r', 'b', 'lbx', 'ubx', 'lbu', 'ubu', 'lg', 'ug']

        with open(json_path, 'r') as f:
            data = json.load(f)

        def extract_sequence(prefix):
            seq = []
            i = 0
            while f"{prefix}_{i:02d}" in data:
                if prefix in matrix_fields:
                    seq.append(np.array(data[f"{prefix}_{i:02d}"]))
                elif prefix in vector_fields:
                    seq.append(np.array(data[f"{prefix}_{i:02d}"]).flatten())
                i += 1
            return seq if seq else None

        self.Q_list = extract_sequence('Q')
        self.R_list = extract_sequence('R')
        self.S_list = extract_sequence('S')
        self.q_list = extract_sequence('q')
        self.r_list = extract_sequence('r')
        self.A_list = extract_sequence('A')
        self.B_list = extract_sequence('B')
        self.b_list = extract_sequence('b')
        self.lbx_list = extract_sequence('lbx')
        self.ubx_list = extract_sequence('ubx')
        self.lbu_list = extract_sequence('lbu')
        self.ubu_list = extract_sequence('ubu')
        self.C_list = extract_sequence('C')
        self.D_list = extract_sequence('D')
        self.lg_list = extract_sequence('lg')
        self.ug_list = extract_sequence('ug')

        self.N = len(self.A_list)
        self.nx = self.A_list[0].shape[0]
        self.nu = self.B_list[0].shape[1]

        if self.lbu_list is not None:
            self.nbu_0 = self.lbu_list[0].shape[0]
            self.nbu = self.lbu_list[1].shape[0] if len(self.lbu_list) > 1 else 0
        else:
            self.nbu_0 = 0
            self.nbu = 0

        if self.lbx_list is not None:
            self.nbx_0 = self.lbx_list[0].shape[0]
            self.nbx = self.lbx_list[1].shape[0] if len(self.lbx_list) > 1 else 0
        else:
            self.nbx_0 = 0
            self.nbx = 0

        if self.lg_list is not None:
            self.nh_0 = self.lg_list[0].shape[0]
            self.nh = self.lg_list[1].shape[0] if len(self.lg_list) > 1 else 0
        else:
            self.nh_0 = 0
            self.nh = 0

def formulate_parametric_OCP_QP(QP_data):
    ocp = AcadosOcp()
    N = QP_data.N
    nx = QP_data.nx
    nu = QP_data.nu
    nbx = np.max([QP_data.nbx_0, QP_data.nbx])
    nbu = np.max([QP_data.nbu_0, QP_data.nbu])

    # params
    A_param = AcadosParam("A", np.zeros((nx, nx)))
    B_param = AcadosParam("B", np.zeros((nx, nu)))
    b_param = AcadosParam("b", np.zeros((nx, 1)))
    Q_param = AcadosParam("Q", np.zeros((nx, nx)))
    R_param = AcadosParam("R", np.zeros((nu, nu)))
    S_param = AcadosParam("S", np.zeros((nu, nx)))
    q_param = AcadosParam("q", np.zeros((nx, 1)))
    r_param = AcadosParam("r", np.zeros((nu, 1)))
    lbx_param = AcadosParam("lbx", np.zeros((nbx, 1)))
    ubx_param = AcadosParam("ubx", np.zeros((nbx, 1)))
    lbu_param = AcadosParam("lbu", np.zeros((nbu, 1)))
    ubu_param = AcadosParam("ubu", np.zeros((nbu, 1)))

    param_manager = AcadosParamManager([A_param, B_param, b_param,
                                        Q_param, R_param, S_param, q_param, r_param,
                                        lbx_param, ubx_param, lbu_param, ubu_param])
    param_manager.N_horizon = N
    param_manager.p_dyna_num = QP_data.nx*QP_data.nx + QP_data.nu*QP_data.nx + QP_data.nx
    param_manager.p_cost_num = QP_data.nx*QP_data.nx + QP_data.nu*QP_data.nu +  QP_data.nx*QP_data.nu + QP_data.nx + QP_data.nu

    ocp.model.name = 'parametric_OCP_QP'

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
    QP_data = QP_From_JSON(json_path)
    ocp, param_manager = formulate_parametric_OCP_QP(QP_data)

    for i in range(QP_data.N):
        param_manager.set_value(name="A", value=QP_data.A_list[i], stage=i)
        param_manager.set_value(name="B", value=QP_data.B_list[i], stage=i)
        param_manager.set_value(name="b", value=QP_data.b_list[i], stage=i)
        param_manager.set_value(name="Q", value=QP_data.Q_list[i], stage=i)
        param_manager.set_value(name="R", value=QP_data.R_list[i], stage=i)
        param_manager.set_value(name="S", value=QP_data.S_list[i], stage=i)
        param_manager.set_value(name="q", value=QP_data.q_list[i], stage=i)
        param_manager.set_value(name="r", value=QP_data.r_list[i], stage=i)
        if i == 0:
            param_manager.set_value(name="lbx", value=QP_data.lbx_list[i], stage=i)
            param_manager.set_value(name="ubx", value=QP_data.ubx_list[i], stage=i)
        param_manager.set_value(name="lbu", value=QP_data.lbu_list[i], stage=i)
        param_manager.set_value(name="ubu", value=QP_data.ubu_list[i], stage=i)

    param_manager.set_value(name="Q", value=QP_data.Q_list[-1], stage=QP_data.N)
    param_manager.set_value(name="q", value=QP_data.q_list[-1], stage=QP_data.N)

    ocp_solver = AcadosOcpSolver(ocp, verbose=False)
    for i in range(ocp.solver_options.N_horizon):
        param_value = param_manager.get_p_stagewise_values(i)
        param_dyna_cost_values = param_value[:param_manager.p_dyna_num + param_manager.p_cost_num]
        param_cons_values = param_value[param_manager.p_dyna_num + param_manager.p_cost_num:]

        # dynamic and cost
        ocp_solver.set(i, 'p', param_dyna_cost_values)

        # constraints
        end_bx = 2*QP_data.nx
        lbu = param_cons_values[end_bx:end_bx+QP_data.nu].flatten()
        ubu = param_cons_values[end_bx+QP_data.nu:end_bx+2*QP_data.nu].flatten()

        ocp_solver.constraints_set(i, 'lbu', lbu)
        ocp_solver.constraints_set(i, 'ubu', ubu)

        if i == 0:
            lbx = param_cons_values[:QP_data.nx].flatten()
            ubx = param_cons_values[QP_data.nx:2*QP_data.nx].flatten()
            ocp_solver.constraints_set(i, 'lbx', lbx)
            ocp_solver.constraints_set(i, 'ubx', ubx)

    param_value_N = param_manager.get_p_stagewise_values(QP_data.N)
    param_dyna_cost_values_N = param_value_N[:param_manager.p_dyna_num + param_manager.p_cost_num]
    ocp_solver.set(QP_data.N, 'p', param_dyna_cost_values_N)

    status = ocp_solver.solve()
    if status != 0:
        print(f"ocp_solver returned status {status}.")
    ocp_solver.print_statistics()
    cost = ocp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    x = [ocp_solver.get(k, "x") for k in range(QP_data.N+1)]
    u = [ocp_solver.get(k, "u") for k in range(QP_data.N)]
    lam = [ocp_solver.get(k, "lam") for k in range(QP_data.N)]
    for k in range(QP_data.N):
        print(f"lam[{k}] = {lam[k]}")

if __name__ == "__main__":
    main()