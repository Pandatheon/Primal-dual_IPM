import numpy as np
from QP_formulation import OCP_QP
from RiccatiQPSolver_constrained import QP_solver
from case_positive_mats import QP_mat

mats = QP_mat()

x0 = np.array([3.0, 3.0])

qp = OCP_QP(mats.N, 
            mats.Q, mats.R, mats.S, mats.q, mats.r,
            mats.A, mats.B, mats.b, 
            mats.C, mats.D, mats.d, 
            mats.Q_e, mats.q_e)
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