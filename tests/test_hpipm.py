import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosOcpQpSolver


json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Furuta_pos_M.json'

def main():

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    qp_solver = AcadosOcpQpSolver(qp)
    stats = qp_solver.solve()
    iter = qp_solver.get_stats("iter")
    print(f"HPIPM solved in {iter} iterations with status {stats}")
    # cost = qp_solver.get_cost()
    # print(f"Optimal cost: {cost}")
    for k in range(qp.N):
        lam = qp_solver.get(k, 'lam')
        print(f"Stage {k}: lam = {lam}")

if __name__ == "__main__":
    main()