import numpy as np
from OCP_QP import QP
from RiccatiQPSolver_unconstrained import QP_solver
from QpAcadosOcp import QpAcadosOcp
from acados_template import AcadosOcpSolver

N = 5
#dynamics
A = np.array([[1.0, 0.0],
              [0.0, 2.0]])
B = np.array([[1.0],
              [1.0]])
b = np.array([1.0, 1.0])
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

x0 = np.array([3.0, 3.0])
C = np.zeros((2, nx))
D = np.zeros((2, nu))
d = np.array([0.0, 0.0])

### Solve with Riccati QP solver
qp = QP(N, Q, R, S, q, r, A, B, b, C, D, d, Q_e, q_e)
qp_solver = QP_solver(qp)
sol = qp_solver.solve(x0)
cost_qp = qp_solver.get_cost()
lambda_qp = qp_solver.recover_lambda()

### Solve with Acados OCP solver
qp_acados = QpAcadosOcp(qp, x0)
ocp_solver = AcadosOcpSolver(qp_acados.ocp, verbose=False)
ocp_solver.solve()
cost_acados = ocp_solver.get_cost()
lambda_acados = ocp_solver.get_flat("pi")

## Print results
print(f"\n=== Predicted cost acados ===\n{cost_acados}")
print("\n=== Predicted next state sequence (x) ===")
x_sol = np.zeros((ocp_solver.N + 1, qp_acados.ocp.model.x.size()[0]))
for i in range(ocp_solver.N + 1):
    x_sol[i, :] = ocp_solver.get(i, "x")
print(x_sol)
print("\n=== Predicted control sequence (u) ===")
u_sol = np.zeros((ocp_solver.N, qp_acados.ocp.model.u.size()[0]))
for i in range(ocp_solver.N):
    u_sol[i, :] = ocp_solver.get(i, "u")
print(u_sol)
print("\n=== Predicted multipliers for dynamics constraints (lambda) ===")
print(lambda_acados)

print(f"\n=== Predicted cost qp ===\n{cost_qp}")
print("\n=== Predicted next state sequence (x) ===")
print(sol.x)
print("\n=== Predicted control sequence (u) ===")
print(sol.u)
print("\n=== Predicted multipliers for dynamics constraints (lambda) ===")
print(lambda_qp)