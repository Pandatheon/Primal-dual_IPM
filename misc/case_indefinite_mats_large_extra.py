import numpy as np

class QP_mats:
    def __init__(self):
        self.N = 10
        #dynamics
        self.A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0],])
        self.B = np.array([[1.0],
                           [2.0],
                           [2.0],
                           [1.0],
                           [1.0]])
        self.b = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        #cost
        self.Q = np.diag(np.array([1e1, 1e1, 1e1, 1e1, 1e1]))
        self.R = np.diag(np.array([-1.0]))
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

    x0 = np.array([3.0, 3.0])
    qp_solver = QP_solver(qp)
    eiglist = qp_solver.check_QP_definiteness(hessian='full')
    print("Eigenvalues of Q matrix:", eiglist)
    reduced_eiglist = qp_solver.check_QP_definiteness(x0, hessian='reduced')
    print("Eigenvalues of reduced QP cost matrices:", reduced_eiglist)