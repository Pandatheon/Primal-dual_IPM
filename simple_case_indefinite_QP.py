import numpy as np
from QP_formulation import OCP_QP_varying
from RiccatiQPSolver_regularization import QP_solver
from case_indefinite_mats import QP_mats
import copy
mats = QP_mats()
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
reg_method = [
    'NO_REGULARIZE',
    'PROJECT', 
    'MIRROR',
    'R_STAGE_LM', 
    'P_STAGE_LM',
    'R_GLOBAL_LM',
    'P_GLOBAL_LM',
]
for method in reg_method:
    print(f"\n\n=== Testing regularization method: {method} ===")
    try:
        copyqp = copy.deepcopy(qp)
        qp_solver = QP_solver(copyqp)
        sol = qp_solver.solve(x0, reg=method)
    except Exception as e:
        print(f"Method {method} failed with exception: {e}")
        continue
    cost_qp = qp_solver.get_cost()
    print(f"\n=== Predicted cost qp with {method} ===\n{cost_qp}")
    print("\n=== Predicted next state sequence (x) ===")
    print(sol.x)
    print("\n=== Predicted control sequence (u) ===")
    print(sol.u)
    qp_solver.print_statistics()