import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from QP_template.QP_base import BaseQP
from solvers.RiccatiQPSolver import QPSolver
from solvers.Regularization_config import Regularizations_config, no_reg, Hessian_reg, condensed_Hessian_reg, stage_reg, reg_list

reg_config = Regularizations_config()
json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Furuta_pos_M.json'
def main():

    qp = BaseQP()
    qp.populate_from_json(json_path)
    print(f"QP loaded from {json_path} with N={qp.N}, nx={qp.nx}, nu={qp.nu}, nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}")

    method = no_reg[0]
    qp_solver = QPSolver(qp, reg=method, type='Riccati')
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    qp_solver.solve(x0=x0, max_iter=30)
    qp_solver.print_statistics()
    print('Method:',method)
    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")

if __name__ == "__main__":
    main()