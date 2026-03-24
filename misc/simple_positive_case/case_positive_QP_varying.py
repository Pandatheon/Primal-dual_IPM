import numpy as np
from QP_formulation import OCP_QP_varying
from RiccatiQPSolver_varying import QP_solver
from utils import read_matrix_list

json_path = "acados_ocp_qp.json"

N = 5
uh = 2.0
lh = -2.0
A = read_matrix_list(json_path, "A")
B = read_matrix_list(json_path, "B")
nx = A[0].shape[0]
nu = B[0].shape[1]
b = [np.array([0.0, 0.0]) for _ in range(N)]
Q = read_matrix_list(json_path, "Q")[0:N]
R = read_matrix_list(json_path, "R")
S = read_matrix_list(json_path, "S")
q = [np.zeros((nx,)) for _ in range(N)]
r = [np.zeros((nu,)) for _ in range(N)]
C = [np.zeros((2, nx)) for _ in range(N)]
D = [np.array([[-1.0],[1.0]]) for _ in range(N)]
d = [np.array([-lh, uh]) for _ in range(N)]
Q_e = Q[-1]
q_e = np.zeros((nx,))

x0 = np.array([3.0, 3.0])

qp = OCP_QP_varying(N, Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e)
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