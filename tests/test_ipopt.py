import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosCasadiOcpQpSolver


json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Furuta_pos_M.json'

def main():

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    opts = {'ipopt': {'print_level': 0, 'max_iter': 1000, 'sb': 'yes'}}
    qp_solver = AcadosCasadiOcpQpSolver(qp, solver='ipopt', solver_opts=opts)
    stats = qp_solver.solve()
    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    for k in range(qp.N):
        lam = qp_solver.get(k, 'lam')
        print(f"Stage {k}: lam = {lam}")

if __name__ == "__main__":
    main()