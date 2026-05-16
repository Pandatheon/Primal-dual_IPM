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
    # cost = qp_solver.get_cost()
    # print(f"Optimal cost: {cost}")
    # print("lam values at each stage:")
    # for k in range(qp.N):
    #     lam = qp_solver.get(k, 'lam')
    #     print(f"Stage {k}: lam = {lam}")

if __name__ == "__main__":
    main()