import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)

import importlib
import numpy as np
import copy

from utils import write_csv
from QP_formulation import OCP_QP_varying
from RiccatiQPSolver_regularization import QP_solver
from Regularizations import Hessian_based_regularization, Reduced_Hessian_based_regularization_stage, Reduced_Hessian_based_regularization_global

case_modules = [
    "M_inde_R_inde_P_pos_1",
    "M_inde_R_inde_P_pos_2",
    "M_inde_R_inde_P_pos_3",
    "M_inde_R_pos_P_inde_1",
    "M_inde_R_pos_P_inde_2",
    "M_inde_R_pos_P_inde_3",
    'M_inde_R_inde_P_inde_1',
    "M_inde_R_inde_P_inde_2",
    "M_inde_R_inde_P_inde_3",
]

reg_method = Hessian_based_regularization + \
             Reduced_Hessian_based_regularization_stage + \
             Reduced_Hessian_based_regularization_global

for module_name in case_modules:
    print(f"\n>>> Running Case: {module_name}")

    mod = importlib.import_module(module_name)
    QP_mats = getattr(mod, "QP_mats") 
    # --------------------

    mats = QP_mats()
    
    if not hasattr(mats, 'name'):
        mats.name = module_name

    nx = mats.A.shape[0]
    x0 = 3.0 * np.ones((nx,))

    Q_list = [mats.Q for _ in range(mats.N)]
    R_list = [mats.R for _ in range(mats.N)]
    S_list = [mats.S for _ in range(mats.N)]
    q_list = [mats.q for _ in range(mats.N)]
    r_list = [mats.r for _ in range(mats.N)]
    A_list = [mats.A for _ in range(mats.N)]
    B_list = [mats.B for _ in range(mats.N)]
    b_list = [mats.b for _ in range(mats.N)]
    C_list = [mats.C for _ in range(mats.N)]
    D_list = [mats.D for _ in range(mats.N)]
    d_list = [mats.d for _ in range(mats.N)]
    Q_e = mats.Q_e
    q_e = mats.q_e

    qp = OCP_QP_varying(mats.N, 
                        Q_list, R_list, S_list, q_list, r_list,
                        A_list, B_list, b_list,
                        C_list, D_list, d_list,
                        Q_e, q_e)

    for method in reg_method:
        print(f"  Testing: {method}")
        try:
            copyqp = copy.deepcopy(qp)
            qp_solver = QP_solver(copyqp)
            sol = qp_solver.solve(x0, reg=method)

            write_csv(mats.name, qp_solver, method)
        except Exception as e:
            print(f"  Failed: {e}")
            write_csv(mats.name, qp_solver, method)
            continue
        
        cost_qp = qp_solver.get_cost()
        # print(f"=== Predicted cost: {cost_qp} ===")