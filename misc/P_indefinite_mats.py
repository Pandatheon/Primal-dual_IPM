import numpy as np

class QP_mats:
    def __init__(self):
        self.N = 5
        #dynamics
        self.A = np.array([[1.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0],
                           [0.0, 0.0, 2.0],])
        self.B = np.array([[1.0, 2.0],
                           [1.0, 1.0],
                           [0.0, 1.0],])
        self.b = np.array([0.0, 0.0, 0.0])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        # cost
        self.Q = np.diag(np.array([2.0, 2.0, 1.0]))
        self.R = np.diag(np.array([2.0, 2.0]))
        self.S = np.zeros((self.nu, self.nx))
        self.q = np.zeros((self.nx,))
        self.r = np.zeros((self.nu,))
        #terminal cost
        self.Q_e = self.Q
        self.q_e = np.zeros((self.nx,))

        self.uh = 2.0
        self.lh = -2.0
        # inequality constraintsL: h(x,u) = Cx + Du <= d
        self.C = np.zeros((4, self.nx))
        self.D = np.array([[-1.0, 0.0],[0.0, -1.0], [1.0, 0.0],[0.0, 1.0]])
        self.d = np.array([-self.lh, -self.lh, self.uh, self.uh])

if __name__ == "__main__":
    mats = QP_mats()
    print("N:", mats.N)
    print("A:\n", mats.A)
    print("B:\n", mats.B)
    print("b:\n", mats.b)
    print("Q:\n", mats.Q)
    print("R:\n", mats.R)
    print("S:\n", mats.S)
    print("q:\n", mats.q)
    print("r:\n", mats.r)
    print("C:\n", mats.C)
    print("D:\n", mats.D)
    print("d:\n", mats.d)
    print("Q_e:\n", mats.Q_e)
    print("q_e:\n", mats.q_e)
    H = np.block([[mats.Q, mats.S.T],
                  [mats.S, mats.R]])
    print("Hessian matrix H:\n", H)
    eigvals = np.linalg.eigvals(H)
    print("Eigenvalues of H:\n", eigvals)
    # get R+B'PB for N horizon
    P = mats.Q_e
    reduced_eigenvals_list = []
    BPB_eigenvals_list = []
    for _ in range(mats.N):
        reduced_hessian = mats.R + mats.B.T @ P @ mats.B
        reduced_eigenvals_list.append(np.linalg.eigvals(reduced_hessian))
        BPB_eigenvals_list.append(np.linalg.eigvals(mats.B.T @ P @ mats.B))
        K = -np.linalg.solve(reduced_hessian, (mats.S + mats.B.T @ P @ mats.A))
        P = mats.Q + mats.A.T @ P @ mats.A - (mats.S.T + mats.A.T @ P @ mats.B) @ (-K)
    print("Eigenvalues of reduced Hessians M (R + B'PB) over horizon:")
    for i, eigs in enumerate(reduced_eigenvals_list):
        print(f" Step {mats.N - i}: {eigs}")
    print("Corresponding B'PB eigenvalues:")
    for i, bpbs in enumerate(BPB_eigenvals_list):
        print(f" Step {mats.N - i}: {bpbs}")