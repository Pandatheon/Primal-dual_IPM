import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosOcpQpSolver, AcadosOcpQpOptions


json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Pos_QP_json/mock_pendulum_qp_data_000.json'

def main():

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    opts = AcadosOcpQpOptions()
    opts.mu0 = 1e1
    opts.hpipm_mode = 'CUSTOMIZED'
    opts.ric_alg = 0
    opts.tol_stat = 1e-6
    opts.tol_eq = 1e-8
    opts.tol_ineq = 1e-8
    opts.tol_comp = 1e-8
    opts.iter_max = 30

    qp_solver = AcadosOcpQpSolver(qp, opts)
    stats = qp_solver.solve()
    iter = qp_solver.get_stats("iter")
    print(f"HPIPM solved in {iter} iterations with status {stats}")

    x = [qp_solver.get(k, "x") for k in range(qp.N+1)]
    u = [qp_solver.get(k, "u") for k in range(qp.N)]
    lam = [qp_solver.get(k, "lam") for k in range(qp.N)]
    # print("Optimal state trajectory:")
    # for k in range(qp.N+1):
    #     print(f"x[{k}] = {x[k]}")
    # print("Optimal control trajectory:")
    # for k in range(qp.N):
    #     print(f"u[{k}] = {u[k]}")
    print("optimal Lagrange multipliers:")
    for k in range(qp.N):
        print(f"lam[{k}] = {lam[k]}")

if __name__ == "__main__":
    main()