import numpy as np
from qp_with_riccati import QP, QP_solver, iterate

N = 5
#dynamics
A = np.array([[1.0, 0.0],
              [0.0, 2.0]])
B = np.array([[1.0, 0.0],
              [0.0, 1.0]])
b = np.array([0.0, 0.0])
nx = A.shape[0]
nu = B.shape[1]
#cost
Q = np.diag(np.array([2.0, 2.0]))
R = np.diag(np.array([1.0]))
S = np.zeros((nu, nx))
q = np.zeros((nx,))
r = np.zeros((nu,))
#terminal cost
Q_e = Q
q_e = np.zeros((nx,))

qp = QP(N, Q, R, S, q, r, A, B, b, Q_e, q_e)

x0 = np.array([10.0, 10.0])
x_guess = np.vstack([x0 for _ in range(N+1)])
u_guess = np.zeros((N, nu))
v_guess = np.zeros_like(u_guess)
sol_init = iterate(x=x_guess, u=u_guess, v=v_guess)

qp_solver = QP_solver(qp, regulatorR=2)
sol_next = qp_solver.solve(sol_init)
cost = qp_solver.get_cost()

print(f"\n=== Predicted cost ===\n{cost}")
print("\n=== Predicted next state sequence (x) ===")
print(sol_next.x)
print("\n=== Predicted control sequence (u) ===")
print(sol_next.u)