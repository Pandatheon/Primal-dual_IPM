import numpy as np
from QP_formulation import Dense_QP
from IPQPSolver import QP_solver

N = 5
#dynamics
A = np.array([[1.0, 0.0],
              [0.0, 2.0]])
B = np.array([[1.0],
              [2.0]])
b = np.array([0.0, 0.0])
nx = A.shape[0]
nu = B.shape[1]
#cost
Q = np.diag(np.array([1.0, 1.0]))
R = np.diag(np.array([1.0]))
S = np.zeros((nu, nx))
q = np.zeros((nx,))
r = np.zeros((nu,))
#terminal cost
Q_e = Q
q_e = np.zeros((nx,))

x0 = np.array([3.0, 3.0])
uh = 2.0
# inequality constraintsL: h(x,u) = Cx + Du <= d
C = np.zeros((2, nx))
D = np.array([[-1.0],[1.0]])
d = np.array([uh, uh])

qp = Dense_QP(N, Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e)
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