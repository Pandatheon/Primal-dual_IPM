import numpy as np
import casadi as ca
class QP_mats:
    def __init__(self):
        self.N = 10
        #dynamics
        self.A = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.B = np.array([[1.0],
                           [2.0],
                           [2.0],
                           [1.0]])
        self.b = np.array([0.0, 0.0, 0.0, 0.0])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        #cost
        self.Q = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0,-1.0]])
        self.R = np.diag(np.array([1.0]))
        self.S = np.zeros((self.nu, self.nx))
        self.q = np.zeros((self.nx,))
        self.r = np.zeros((self.nu,))
        #terminal cost
        self.Q_e = self.Q
        self.q_e = np.zeros((self.nx,))

        self.uh = 2.0
        self.lh = -2.0
        # inequality constraintsL: h(x,u) = Cx + Du <= d
        self.C = np.zeros((2, self.nx))
        self.D = np.array([[-1.0],[1.0]])
        self.d = np.array([-self.lh, self.uh])

if __name__ == "__main__":
    from QP_formulation import OCP_QP
    from RiccatiQPSolver_regularization import QP_solver
    import numpy as np

    mats = QP_mats()
    qp = OCP_QP(mats.N,
                mats.Q, mats.R, mats.S, mats.q, mats.r,
                mats.A, mats.B, mats.b,
                mats.C, mats.D, mats.d,
                mats.Q_e, mats.q_e)
    H = np.block([[mats.Q, mats.S.T],
                 [mats.S, mats.R]])
    H_sym = H
    D_list = ca.ldl(H_sym)
    d_eigs = D_list[0].full()
    # clip negative eigenvalues to zero
    d_eigs[d_eigs < 0] = 0.0
    L = D_list[1].full().T + np.eye(H.shape[0])
    H_rec = L @ ca.diag(d_eigs) @ L.T
    print("Original H matrix:\n", H)
    print("Reconstructed H matrix with non-negative eigenvalues:\n", H_rec)
    Q = mats.Q

    # take the upper-left block and filp down to make it symmetric
    Q_sym = Q
    D_list = ca.ldl(Q_sym)
    d_eigs = D_list[0].full()
    # clip negative eigenvalues to zero
    d_eigs[d_eigs < 0] = 0.0
    L = D_list[1].full().T + np.eye(Q.shape[0])
    Q_rec = L @ ca.diag(d_eigs) @ L.T
    print("Original Q matrix:\n", Q)
    print("Reconstructed Q matrix with non-negative eigenvalues:\n", Q_rec)


    x0 = 3.0 * np.ones((mats.nx,))
    qp_solver = QP_solver(qp)
    eiglist = qp_solver.check_QP_definiteness(hessian='full')
    print("Eigenvalues of Q matrix:", eiglist)
    reduced_eiglist = qp_solver.check_QP_definiteness(x0, hessian='reduced')
    print("Eigenvalues of reduced QP cost matrices:", reduced_eiglist)