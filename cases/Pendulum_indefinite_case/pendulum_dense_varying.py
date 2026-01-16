import numpy as np
from QP_formulation import Dense_QP_varying
from IPQPSolver_varying import QP_solver
from utils import read_matrix_list

json_path = "pendulum_caseF_qp_2_100.json"

N = 20
uh = 80.0
lh = -80.0
A = read_matrix_list(json_path, "A")
B = read_matrix_list(json_path, "B")
nx = A[0].shape[0]
nu = B[0].shape[1]
b = [np.zeros((nx,)) for _ in range(N)]
# Q = read_matrix_list(json_path, "Q")
# R = read_matrix_list(json_path, "R")
# S = read_matrix_list(json_path, "S")
Q = [np.array([[10.0, 0.0, 0.0, 0.0],
               [0.0, 20.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]) for _ in range(N)]
R = [np.array([[0.1]]) for _ in range(N)]
S = [np.zeros((nu, nx)) for _ in range(N)]
q = [np.zeros((nx,)) for _ in range(N)]
r = [np.zeros((nu,)) for _ in range(N)]
C = [np.zeros((2, nx)) for _ in range(N)]
D = [np.array([[-1.0],[1.0]]) for _ in range(N)]
d = [np.array([-lh, uh]) for _ in range(N)]
Q_e = Q[-1]
q_e = np.zeros((nx,))

x0 = np.array([0.0, np.pi, 0.0, 0.0])

qp = Dense_QP_varying(N, Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e)
qp_solver = QP_solver(qp)
sol = qp_solver.solve(x0)
cost_qp = qp_solver.get_cost()

# # Print results
print(f"\n=== Predicted cost qp ===\n{cost_qp}")
print("\n=== Predicted next state sequence (x) ===")
print(sol.x)
print("\n=== Predicted control sequence (u) ===")
print(sol.u)
qp_solver.print_statistics()