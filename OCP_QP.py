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

class QP():
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
        self.stages = []
        for _ in range(N):
            self.stages.append(QP_stage(Q, R, S, q, r, A, B, b, C, D, d))
        self.terminal = QP_Terminal(Q_e, q_e)

        self.N = N
        self.nx = Q.shape[0]
        self.nu = R.shape[0]
        self.nh = C.shape[0]

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

    def __call__(self):
        return(
            self.Q, self.R, self.S, self.q, self.r,
            self.A, self.B, self.b,
            self.C, self.D, self.d,
            self.Q_e, self.q_e
        )
