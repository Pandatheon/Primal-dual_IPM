import numpy as np
from QP_formulation import OCP_QP_varying
from RiccatiQPSolver_regularization import QP_solver
from Regularizations import Regularizations_config
from utils import read_matrix_list
import copy
from utils import write_csv

json_path_list = [
    "/home/jingtao/Primal-dual_IPM/cases/Pendulum_indefinite_case/pendulum_caseF_qp_N=100.json",
    "/home/jingtao/Primal-dual_IPM/cases/Pendulum_indefinite_case/pendulum_caseF_qp_N=50.json",
]
reg_config = Regularizations_config()
reg_method = reg_config.Hessian_based_regularization + \
             reg_config.stage_based_regularization

for json_path in json_path_list:
    # get N from path name
    N = int(json_path.split("N=")[-1].split(".json")[0])
    print(f"\n>>> Running Pendulum Case with N={N}")
    uh = 80.0
    lh = -80.0
    A = read_matrix_list(json_path, "A")
    B = read_matrix_list(json_path, "B")
    nx = A[0].shape[0]
    nu = B[0].shape[1]
    b = [np.zeros((nx,)) for _ in range(N)]
    Q = read_matrix_list(json_path, "Q")
    R = read_matrix_list(json_path, "R")
    S = read_matrix_list(json_path, "S")
    q = [np.zeros((nx,)) for _ in range(N)]
    r = [np.zeros((nu,)) for _ in range(N)]
    C = [np.zeros((2, nx)) for _ in range(N)]
    D = [np.array([[-1.0],[1.0]]) for _ in range(N)]
    d = [np.array([-lh, uh]) for _ in range(N)]
    Q_e = Q[-1]
    q_e = np.zeros((nx,))

    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    qp = OCP_QP_varying(N, Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e)

    for method in reg_method:
        print(f"\n\n=== Testing regularization method: {method} ===")
        try:
            copyqp = copy.deepcopy(qp)
            qp_solver = QP_solver(copyqp)
            sol = qp_solver.solve(x0, reg=method)
            write_csv(f"Pendulum_caseF_N={N}", qp_solver, method)
        except Exception as e:
            print(f"Method {method} failed with exception: {e}")
            write_csv(f"Pendulum_caseF_N={N}", qp_solver, method)
            continue
        cost_qp = qp_solver.get_cost()
        print(f"\n=== Predicted cost qp with {method} ===\n{cost_qp}")
        # qp_solver.print_statistics()