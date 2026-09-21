import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosOcpQpSolver, AcadosOcpQpOptions


json_path = '/home/jingtao/Mehrotra_IPM/tests/QP_json/temp/overhead_crane_FU_PD_RE_PD_status_2_iter_19.json'
def main():


    opts = AcadosOcpQpOptions()
    opts.mu0 = 1e1
    opts.hpipm_mode = 'CUSTOMIZED'
    opts.ric_alg = 0
    opts.tol_stat = 1e-6
    opts.tol_eq = 1e-8
    opts.tol_ineq = 1e-8
    opts.tol_comp = 1e-6
    opts.iter_max = 30

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    qp_solver = AcadosOcpQpSolver(qp, opts)
    stats = qp_solver.solve()
    iter = qp_solver.get_stats("iter")
    print(f"HPIPM solved in {iter} iterations with status {stats}")
    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    qp_solver.print_statistics()

    # x = [qp_solver.get(k, "x") for k in range(qp.N+1)]
    # u = [qp_solver.get(k, "u") for k in range(qp.N)]
    # lam = [qp_solver.get(k, "lam") for k in range(qp.N)]
    # pi = [qp_solver.get(k, "pi") for k in range(qp.N)]
    # print("Optimal state trajectory:")
    # for k in range(qp.N+1):
    #     print(f"x[{k}] = {x[k]}")
    # print("Optimal control trajectory:")
    # for k in range(qp.N):
    #     print(f"u[{k}] = {u[k]}")
    # print("optimal Lagrange multipliers:")
    # for k in range(qp.N):
    #     print(f"lam[{k}] = {lam[k]}")
    # for k in range(qp.N):
    #     print(f"Stage {k}: pi = {pi[k]}")

if __name__ == "__main__":
    main()