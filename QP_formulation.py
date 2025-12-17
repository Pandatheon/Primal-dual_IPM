from dataclasses import dataclass
from typing import NamedTuple
from dataclasses import dataclass, astuple

import numpy as np

"""it can be 1D or 2D, symbol (ca.SX) or value (np.ndarray)."""

@dataclass
class iterate:
    """keep a convention that the value is always a 2D-array,
    each row of which coresponds to each discrete time.
    """
    x: np.ndarray
    u: np.ndarray
    pi: np.ndarray = None
    lam0: np.ndarray = None
    lam: np.ndarray = None
    s0: np.ndarray = None
    s: np.ndarray = None
    p: np.ndarray = None

    def __iter__(self):
        """easy unpacking"""
        return iter(astuple(self))

class QP_stage(NamedTuple):
    """QP stage cost parameters arranged as (Q, R, S, q, r, A, B, b)"""
    Q: np.ndarray
    R: np.ndarray
    S: np.ndarray
    q: np.ndarray
    r: np.ndarray
    A: np.ndarray
    B: np.ndarray
    b: np.ndarray
    C: np.ndarray
    D: np.ndarray
    d: np.ndarray

class QP_Terminal(NamedTuple):
    """QP terminal cost parameters arranged as (P, p)"""
    Q_e: np.ndarray
    q_e: np.ndarray

class OCP_QP():
    '''
    -------------------------------------------------------------
    stage-wise OCP QP formulation
    only supports time-invariant dynamics and cost for now
    -------------------------------------------------------------
    '''
    def __init__(self, N, Q:np.ndarray, R:np.ndarray, S:np.ndarray, q:np.ndarray, r:np.ndarray, 
                 A:np.ndarray, B:np.ndarray, b:np.ndarray,
                 C:np.ndarray, D:np.ndarray, d:np.ndarray,
                 Q_e:np.ndarray, q_e:np.ndarray):

        nx = Q.shape[0]
        nu = R.shape[0]
        self.Q = Q
        self.R = R
        self.S = S
        self.q = q
        self.r = r
        self.A = A
        self.B = B
        self.b = b
        self.C = C
        self.D = D
        self.d = d
        self.Q_e = Q_e
        self.q_e = q_e
        self.C0 = np.vstack((-np.eye(nx), np.eye(nx), self.C))
        self.D0 = np.vstack((np.zeros((2*nx, nu)), self.D))

        self.N = N
        self.nx = Q.shape[0]
        self.nu = R.shape[0]
        self.nh = C.shape[0]
        self.nh0 = self.C0.shape[0]

        self.stages = []
        self.stages.append(QP_stage(Q, R, S, q, r, A, B, b, self.C0, self.D0, d))
        for _ in range(N-1):
            self.stages.append(QP_stage(Q, R, S, q, r, A, B, b, C, D, d))
        self.terminal = QP_Terminal(Q_e, q_e)

    def __call__(self):
        return(
            self.Q, self.R, self.S, self.q, self.r,
            self.A, self.B, self.b,
            self.C, self.D, self.d,
            self.Q_e, self.q_e
        )

class OCP_QP_varying():
    '''
    -------------------------------------------------------------
    stage-wise OCP QP formulation
    supports time-variant dynamics and cost
    -------------------------------------------------------------
    '''
    def __init__(self, N, Q:list[np.ndarray], R:list[np.ndarray], S:list[np.ndarray], q:list[np.ndarray], r:list[np.ndarray], 
                 A:list[np.ndarray], B:list[np.ndarray], b:list[np.ndarray],
                 C:list[np.ndarray], D:list[np.ndarray], d:list[np.ndarray],
                 Q_e:np.ndarray, q_e:np.ndarray):

        C0 = np.vstack((-np.eye(self.nx), np.eye(self.nx), C[0]))
        D0 = np.vstack((np.zeros((2*self.nx, self.nu)), D[0]))
        self.N = N
        self.nx = Q[0].shape[0]
        self.nu = R[0].shape[0]
        self.nh = C[0].shape[0]
        self.nh0 = C0.shape[0]

        self.stages = []
        self.stages.append(QP_stage(Q[0], R[0], S[0], q[0], r[0], A[0], B[0], b[0], C0, D0, d[0]))
        for i in range(1, N):
            self.stages.append(QP_stage(Q[i], R[i], S[i], q[i], r[i], A[i], B[i], b[i], C[i], D[i], d[i]))
        self.terminal = QP_Terminal(Q_e, q_e)

class Dense_QP():
    '''
    -------------------------------------------------------------
    Dense OCP QP formulation
    only supports time-invariant dynamics and cost for now
    -------------------------------------------------------------
    '''
    def __init__(self, N, Q:np.ndarray, R:np.ndarray, S:np.ndarray, q:np.ndarray, r:np.ndarray, 
                 A:np.ndarray, B:np.ndarray, b:np.ndarray, 
                 C:np.ndarray, D:np.ndarray, d:np.ndarray, 
                 Q_e:np.ndarray, q_e:np.ndarray):

        nx = Q.shape[0]
        nu = R.shape[0]
        self.Q = Q
        self.R = R
        self.S = S
        self.q = q
        self.r = r
        self.A = A
        self.B = B
        self.b = b
        self.C = C
        self.D = D
        self.d = d
        self.Q_e = Q_e
        self.q_e = q_e
        self.C0 = np.vstack((-np.eye(nx), np.eye(nx), self.C))
        self.D0 = np.vstack((np.zeros((2*nx, nu)), self.D))
        self.d0 = np.hstack((np.zeros((2*nx, )), self.d))

        self.N = N
        self.nx = nx
        self.nu = nu
        self.nh = C.shape[0]
        self.nh0 = self.C0.shape[0]

        self.dimension_H = N*(self.nx+self.nu) + self.nx
        self.dimension_A = N*self.nx
        self.dimension_C = self.nh0 + (N-1)*self.nh
        self.dimension_T = self.dimension_C

        self.H_matrix = self._get_matrixs('H')
        self.A_matrix = self._get_matrixs('A')
        self.C_matrix = self._get_matrixs('C')

    def __call__(self):
        return(
            self.Q, self.R, self.S, self.q, self.r,
            self.A, self.B, self.b,
            self.C, self.D, self.d,
            self.Q_e, self.q_e
        )

    def _get_matrixs(self, matrix_str:str):
        nx = self.nx
        nu = self.nu
        if matrix_str == 'H':
            matrix = np.zeros((self.dimension_H, self.dimension_H))
            for i in range(self.N):
                matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu):i*(nx+nu)+nx] = self.Q
                matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.R
                matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.S.T
                matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu):i*(nx+nu)+nx] = self.S
            matrix[self.N*(nx+nu):self.N*(nx+nu)+nx, self.N*(nx+nu):self.N*(nx+nu)+nx] = self.Q_e

        elif matrix_str == 'A':
            matrix = np.zeros((self.dimension_A, self.dimension_H))
            for i in range(self.N):
                matrix[i*nx:i*nx+nx, i*(nx+nu):i*(nx+nu)+nx] = self.A
                matrix[i*nx:i*nx+nx, i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.B
                matrix[i*nx:i*nx+nx, (i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = -np.eye(nx)

        elif matrix_str == 'C':
            matrix = np.zeros((self.dimension_C, self.dimension_H))
            matrix[0:self.nh0, 0:nx] = self.C0
            matrix[0:self.nh0, nx:nx+nu] = self.D0
            for i in range(1, self.N):
                matrix[self.nh0+(i-1)*self.nh:self.nh0+i*self.nh, i*(nx+nu):i*(nx+nu)+nx] = self.C
                matrix[self.nh0+(i-1)*self.nh:self.nh0+i*self.nh, i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.D

        return matrix