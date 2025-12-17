import numpy as np
from QP_formulation import OCP_QP, OCP_QP_varying
from RiccatiQPSolver_varying import QP_solver

N = 5
#dynamics
uh = 2.0
A = [np.array([[1.0, 0.0],[0.0, 2.0]]) for _ in range(N)]
B = [np.array([[1.0],[2.0]]) for _ in range(N)]
nx = A[0].shape[0]
nu = B[0].shape[1]
b = [np.array([0.0, 0.0]) for _ in range(N)]
Q = [np.diag(np.array([1.0, 1.0])) for _ in range(N)]
R = [np.diag(np.array([1.0])) for _ in range(N)]
S = [np.zeros((nu, nx)) for _ in range(N)]
q = [np.zeros((nx,)) for _ in range(N)]
r = [np.zeros((nu,)) for _ in range(N)]
C = [np.zeros((2, nx)) for _ in range(N)]
D = [np.array([[-1.0],[1.0]]) for _ in range(N)]
d = [np.array([uh, uh]) for _ in range(N)]
Q_e = np.diag(np.array([1.0, 1.0]))
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