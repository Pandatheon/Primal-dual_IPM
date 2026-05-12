'''
Main file for runnign benchmark
'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import importlib
import numpy as np
import copy

from solvers.RiccatiQPSolver import QPSolver
from QP_template.QP_base import BaseQP
from solvers.Regularization_config import reg_list


case_package = 'cases.Designed_case'

case_modules = [
    'M_pos_R_pos_P_inde',
    'M_pos_R_inde_P_pos',
    "M_inde_R_inde_P_pos_1",
    "M_inde_R_inde_P_pos_2",
    "M_inde_R_inde_P_pos_3",
    "M_inde_R_inde_P_pos_4",
    "M_inde_R_pos_P_inde_1",
    "M_inde_R_pos_P_inde_2",
    "M_inde_R_pos_P_inde_3",
    'M_inde_R_pos_P_inde_4',
    "M_inde_R_inde_P_inde_2",
    "M_inde_R_inde_P_inde_3",
    'M_inde_R_inde_P_inde_4',
]


def build_qp(mats):
    qp = BaseQP()
    qp.Q = [mats.Q for _ in range(mats.N)]
    qp.R = [mats.R for _ in range(mats.N)]
    qp.S = [mats.S for _ in range(mats.N)]
    qp.q = [mats.q for _ in range(mats.N)]
    qp.r = [mats.r for _ in range(mats.N)]
    qp.A = [mats.A for _ in range(mats.N)]
    qp.B = [mats.B for _ in range(mats.N)]
    qp.b = [mats.b for _ in range(mats.N)]
    qp.C = [mats.C for _ in range(mats.N)]
    qp.D = [mats.D for _ in range(mats.N)]
    qp.d = [mats.d for _ in range(mats.N)]
    qp.Q_e = mats.Q_e
    qp.q_e = mats.q_e
    qp.N = mats.N
    qp.nx = mats.A.shape[0]
    qp.nu = mats.B.shape[1]
    qp.nh = mats.C.shape[0]
    return qp

def main():
    module_name = 'M_pos_R_pos_P_inde'
    print(f"\n>>> Running Case: {module_name}")

    module_path = f"{case_package}.{module_name}"
    mod = importlib.import_module(module_path)
    QP_mats = getattr(mod, "QP_mats")
    # --------------------

    mats = QP_mats()

    if not hasattr(mats, 'name'):
        mats.name = module_path

    qp = build_qp(mats)
    x0 = 3.0 * np.ones((qp.A[0].shape[0],))
    for method in ['NO_REGULAIRZE']:
        print(f"  Testing: {method}")
        name = f"{mats.name}_{x0[0]}"
        copyqp = copy.deepcopy(qp)
        qp_solver = QPSolver(copyqp)
        sol = qp_solver.solve(x0, reg=method, max_iter=300)

        cost_qp = qp_solver.get_cost()
        print(f"=== Predicted cost: {cost_qp} ===")
        qp_solver.print_statistics()

if __name__ == '__main__':
    main()