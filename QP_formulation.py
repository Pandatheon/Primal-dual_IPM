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
    def __init__(self, N, 
                 Q_list:list[np.ndarray], R_list:list[np.ndarray], S_list:list[np.ndarray],
                 q_list:list[np.ndarray], r_list:list[np.ndarray], 
                 A_list:list[np.ndarray], B_list:list[np.ndarray], b_list:list[np.ndarray],
                 C_list:list[np.ndarray], D_list:list[np.ndarray], d_list:list[np.ndarray],
                 Q_e:np.ndarray, q_e:np.ndarray):

        self.N = N
        self.nx = Q_list[0].shape[0]
        self.nu = R_list[0].shape[0]
        self.nh = C_list[0].shape[0]

        C0 = np.vstack((-np.eye(self.nx), np.eye(self.nx), C_list[0]))
        D0 = np.vstack((np.zeros((2*self.nx, self.nu)), D_list[0]))
        self.nh0 = C0.shape[0]

        self.stages = []
        self.stages.append(QP_stage(Q_list[0], R_list[0], S_list[0], q_list[0], r_list[0], A_list[0], B_list[0], b_list[0], C0, D0, d_list[0]))
        for i in range(1, N):
            self.stages.append(QP_stage(Q_list[i], R_list[i], S_list[i], q_list[i], r_list[i], A_list[i], B_list[i], b_list[i], C_list[i], D_list[i], d_list[i]))
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

class Dense_QP_varying():
    '''
    -------------------------------------------------------------
    Dense OCP QP formulation
    supports time-variant dynamics and cost
    -------------------------------------------------------------
    '''
    def __init__(self, N, 
                 Q_list:list[np.ndarray], R_list:list[np.ndarray], S_list:list[np.ndarray],
                 q_list:list[np.ndarray], r_list:list[np.ndarray], 
                 A_list:list[np.ndarray], B_list:list[np.ndarray], b_list:list[np.ndarray],
                 C_list:list[np.ndarray], D_list:list[np.ndarray], d_list:list[np.ndarray],
                 Q_e:np.ndarray, q_e:np.ndarray):

        self.N = N
        self.nx = Q_list[0].shape[0]
        self.nu = R_list[0].shape[0]
        self.nh = C_list[0].shape[0]

        self.Q_list = Q_list
        self.R_list = R_list
        self.S_list = S_list
        self.q_list = q_list
        self.r_list = r_list
        self.A_list = A_list
        self.B_list = B_list
        self.b_list = b_list
        self.C_list = C_list
        self.D_list = D_list
        self.d_list = d_list
        self.Q_e = Q_e
        self.q_e = q_e
        self.C0 = np.vstack((-np.eye(self.nx), np.eye(self.nx), self.C_list[0]))
        self.D0 = np.vstack((np.zeros((2*self.nx, self.nu)), self.D_list[0]))
        self.d0 = np.hstack((np.zeros((2*self.nx, )), self.d_list[0]))
        self.nh0 = self.C0.shape[0]

        self.dimension_H = N*(self.nx+self.nu) + self.nx
        self.dimension_A = N*self.nx
        self.dimension_C = self.nh0 + (N-1)*self.nh
        self.dimension_T = self.dimension_C
        self.H_matrix = self._get_matrixs('H')
        self.A_matrix = self._get_matrixs('A')
        self.C_matrix = self._get_matrixs('C')

    def _get_matrixs(self, matrix_str:str):
            nx = self.nx
            nu = self.nu
            if matrix_str == 'H':
                matrix = np.zeros((self.dimension_H, self.dimension_H))
                for i in range(self.N):
                    matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu):i*(nx+nu)+nx] = self.Q_list[i]
                    matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.R_list[i]
                    matrix[i*(nx+nu):i*(nx+nu)+nx, i*(nx+nu)+nx:i*(nx+nu)+nx+nu] = self.S_list[i].T
                    matrix[i*(nx+nu)+nx:i*(nx+nu)+nx+nu, i*(nx+nu):i*(nx+nu)+nx] = self.S_list[i]
                matrix[self.N*(nx+nu):self.N*(nx+nu)+nx, self.N*(nx+nu):self.N*(nx+nu)+nx] = self.Q_e

            elif matrix_str == 'A':
                matrix = np.zeros((self.dimension_A, self.dimension_H))
                for i in range(self.N):
                    matrix[i*nx:i*nx+nx, i*(nx+nu):i*(nx+nu)+nx] = self.A_list[i]
                    matrix[i*nx:i*nx+nx, i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.B_list[i]
                    matrix[i*nx:i*nx+nx, (i+1)*(nx+nu):(i+1)*(nx+nu)+nx] = -np.eye(nx)

            elif matrix_str == 'C':
                matrix = np.zeros((self.dimension_C, self.dimension_H))
                matrix[0:self.nh0, 0:nx] = self.C0
                matrix[0:self.nh0, nx:nx+nu] = self.D0
                for i in range(1, self.N):
                    matrix[self.nh0+(i-1)*self.nh:self.nh0+i*self.nh, i*(nx+nu):i*(nx+nu)+nx] = self.C_list[i]
                    matrix[self.nh0+(i-1)*self.nh:self.nh0+i*self.nh, i*(nx+nu)+nx:(i+1)*(nx+nu)] = self.D_list[i]
            return matrix