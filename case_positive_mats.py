import numpy as np

class QP_mat:
    def __init__(self):
        self.N = 5
        #dynamics
        self.A = np.array([[1.0, 0.0],
                           [0.0, 2.0]])
        self.B = np.array([[1.0],
                           [2.0]])
        self.b = np.array([0.0, 0.0])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        #cost
        self.Q = np.diag(np.array([1.0, 1.0]))
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
    mats = QP_mat()
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