from dataclasses import dataclass
from typing import NamedTuple, Tuple, TypeVar
from dataclasses import dataclass, astuple

import casadi as ca
import numpy as np

T = TypeVar('T', ca.SX, np.ndarray)
"""it can be 1D or 2D, symbol (ca.SX) or value (np.ndarray)."""

@dataclass
class iterate:
    """keep a convention that the value is always a 2D-array,
    each row of which coresponds to each discrete time.
    """
    x: np.ndarray
    u: np.ndarray
    p: np.ndarray=None

    def __post_init__(self):
        if self.p is None:
            N = self.x.shape[0] - 1
            # NOTE: the shape is consistent with how OCP treat p when it's None
            self.p = np.zeros((N+1, 1))

    def __iter__(self):
        """easy unpacking"""
        return iter(astuple(self))

class QP_stage(NamedTuple):
    """QP stage cost parameters arranged as (Q, R, S, q, r, A, B, b)"""
    Q: T
    R: T
    S: T
    q: T
    r: T
    A: T
    B: T
    b: T

class QP_Terminal(NamedTuple):
    """QP terminal cost parameters arranged as (P, p)"""
    Q_e: T
    q_e: T


class QP():
    def __init__(self, N, Q:T, R:T, S:T, q:T, r:T, A:T, B:T, b:T, Q_e:T, q_e:T):
        self.stages = []
        for _ in range(N):
            self.stages.append(QP_stage(Q, R, S, q, r, A, B, b))
        self.terminal = QP_Terminal(Q_e, q_e)

        self.nx = Q.shape[0]
        self.nu = R.shape[0]
        self.N = N