import numpy as np
from QP_formulation import OCP_QP_varying
from RiccatiQPSolver_varying import QP_solver
from utils import read_matrix_list

json_path = "pendulum_caseF_qp_2_10.json"

N = 15
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
# Q = [np.array([[10.0, 0.0, 0.0, 0.0],
#                [0.0, 20.0, 0.0, 0.0],
#                [0.0, 0.0, 1.0, 0.0],
#                [0.0, 0.0, 0.0, 1.0]]) for _ in range(N)]
# R = [np.array([[0.1]]) for _ in range(N)]
# S = [np.zeros((nu, nx)) for _ in range(N)]
q = [np.zeros((nx,)) for _ in range(N)]
r = [np.zeros((nu,)) for _ in range(N)]
C = [np.zeros((2, nx)) for _ in range(N)]
D = [np.array([[-1.0],[1.0]]) for _ in range(N)]
d = [np.array([-lh, uh]) for _ in range(N)]
Q_e = Q[-1]
q_e = np.zeros((nx,))

if __name__ == "__main__":
    print("N:", N)
    for i in range(N):
        print(f"\n--- Stage {i} ---")
        print(f"A[{i}]:\n", A[i])
        print(f"B[{i}]:\n", B[i])
        print(f"b[{i}]:\n", b[i])
        print(f"Q[{i}]:\n", Q[i])
        print(f"R[{i}]:\n", R[i])
        print(f"S[{i}]:\n", S[i])
        H = np.block([[Q[i], S[i].T],
                    [S[i], R[i]]])
        eigvals = np.linalg.eigvals(H)
        print("Eigenvalues of H:\n", eigvals)
    print("Q_e:\n", Q_e)
    print("q_e:\n", q_e)

    # get R+B'PB for N horizon
    P = Q_e
    reduced_eigenvals_list = []
    for i in reversed(range(N)):
        reduced_hessian = R[i] + B[i].T @ P @ B[i]
        reduced_eigenvals = np.linalg.eigvals(reduced_hessian)
        reduced_eigenvals_list.append(reduced_eigenvals)
        K = -np.linalg.solve(reduced_hessian, (S[i] + B[i].T @ P @ A[i]))
        P = Q[i] + A[i].T @ P @ A[i] - (S[i].T + A[i].T @ P @ B[i]) @ (-K)
    print("Eigenvalues of reduced Hessians R + B'PB over horizon:")
    for i, eigs in enumerate(reduced_eigenvals_list):
        print(f" Step {N - i}: {eigs}")