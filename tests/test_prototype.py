import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from QP_template.QP_base import BaseQP
from solvers.RiccatiQpSolver import QpSolver
from solvers.Regularization_config import Regularizations_config, no_reg, Block_Hessian_reg, Block_reduced_Hessian_reg, reg_list

reg_config = Regularizations_config()
json_path = '/home/jingtao/Mehrotra_IPM/cases/Tracking_inde_QP_json/acrobot_FU_INDE_RE_INDE_status_4_iter_17.json'
def main():

    qp = BaseQP()
    qp.populate_from_json(json_path)
    qp_name = os.path.basename(json_path)
    print(f"QP loaded from {qp_name} with N={qp.N}, nx={qp.nx[1]}, nu={qp.nu}, nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}")

    # method = 'NO_REGULARIZE'
    # qp_solver = QpSolver(qp, reg=method, type='Riccati', mehrotra=True, globalization=False)
    # qp_solver.solve(max_iter=30)

    method = 'PM_STAGE_MIRROR'
    qp_solver = QpSolver(qp, reg=method, type='Riccati', mehrotra=False, globalization=True)
    qp_solver.solve(max_iter=100)

    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    qp_solver.print_statistics()

if __name__ == "__main__":
    main()