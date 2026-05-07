import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from QP_template.QP_base import BaseQP
from solvers.RiccatiQPSolver import QPSolver
from solvers.Regularization_config import Regularizations_config, reg_list
from utils.utilis import read_matrix_list
import copy

json_path_list = [
    "/home/jingtao/Primal-dual_IPM/cases/Pendulum_indefinite_case/pendulum_caseF_qp_N=20.json",
    # "/home/jingtao/Primal-dual_IPM/cases/Pendulum_indefinite_case/pendulum_caseF_qp_N=100.json",
]
reg_config = Regularizations_config()
reg_method = [
    'PM_STAGE_MIRROR', 
    'PM_lookahead_CM'
    ]

for json_path in json_path_list:
    # get N from path name
    N = int(json_path.split("N=")[-1].split(".json")[0])
    print(f"\n>>> Running Pendulum Case with N={N}")
    uh = 80.0
    lh = -80.0
    qp = BaseQP()
    qp.N = int(json_path.split("N=")[-1].split(".json")[0])
    qp.A = read_matrix_list(json_path, "A")
    qp.B = read_matrix_list(json_path, "B")
    qp.nx = qp.A[0].shape[0]
    qp.nu = qp.B[0].shape[1]
    qp.b = read_matrix_list(json_path, "b")
    qp.Q = read_matrix_list(json_path, "Q")[:-1]
    qp.R = read_matrix_list(json_path, "R")
    qp.S = read_matrix_list(json_path, "S")
    qp.q = read_matrix_list(json_path, "q")[:-1]
    qp.r = read_matrix_list(json_path, "r")
    qp.C = [np.zeros((2, qp.nx)) for _ in range(qp.N)]
    qp.D = [np.array([[-1.0],[1.0]]) for _ in range(qp.N)]
    qp.d = [np.array([-lh, uh]) for _ in range(qp.N)]
    qp.nh = qp.C[0].shape[0]
    qp.Q_e = read_matrix_list(json_path, "Q")[-1]
    qp.q_e = read_matrix_list(json_path, "q")[-1]

    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    for method in reg_method:
        print(f"\n\n=== Testing regularization method: {method} ===")
        try:
            copyqp = copy.deepcopy(qp)
            qp_solver = QPSolver(copyqp, reg=method)
            sol = qp_solver.solve(x0, max_iter=100)
            qp_solver.print_statistics()
        except Exception as e:
            print(f"Method {method} failed with exception: {e}")
            continue
        cost_qp = qp_solver.get_cost()
        print(f"\n=== Predicted cost qp with {method} ===\n{cost_qp}")
        # qp_solver.print_statistics()