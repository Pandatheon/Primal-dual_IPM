import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from QP_template.QP_base import BaseQP
from solvers.RiccatiQPSolver import QPSolver
from solvers.Regularization_config import Regularizations_config, reg_list

reg_config = Regularizations_config()
json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Furuta_inde_M.json'

def main():

    qp = BaseQP()
    qp.populate_from_json(json_path)
    print(f"QP loaded from {json_path} with N={qp.N}, nx={qp.nx}, nu={qp.nu}, nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}")

    method = reg_list[1]
    qp_solver = QPSolver(qp, reg=method)
    x0 = np.zeros((qp.nx,))
    qp_solver.solve(x0=x0, max_iter=200)
    print('Method:',method)
    qp_solver.print_statistics()

if __name__ == "__main__":
    main()