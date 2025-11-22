from typing import Tuple, TypeVar
from dataclasses import dataclass
from RiccatiQP import QP, QP_stage, iterate

import casadi as ca
import numpy as np

T = TypeVar('T', ca.SX, np.ndarray)
"""it can be 1D or 2D, symbol (ca.SX) or value (np.ndarray)."""

@dataclass
class Feedback:
    """QP terminal cost parameters arranged as (P, p)"""
    P: T
    p: T
    K: T
    k: T

    def compute_control(self, index: int, state: np.ndarray) -> np.ndarray:
        '''compute the control according to the feedback law u = k + Kx'''
        return self.K[index] @ state + self.k[index]

class QP_solver:

    def __init__(self, qp:QP):
        self.qp = qp
        self.feedback = None
        self.solution = None
        self.sol_init = iterate(
            x = np.zeros((qp.N + 1, qp.nx)),
            u =  np.zeros((qp.N, qp.nu))
        )

    def solve(self, x0:np.ndarray=None)-> iterate:
        if x0 is not None:
            self.sol_init.x = np.vstack([x0 for _ in range(self.qp.N+1)])
        self._backward_sweep()
        return self._forward_sweep(self.sol_init)

    def _forward_sweep(self, this_iter:iterate)-> 'iterate':
        x_bar = this_iter.x
        u_bar = this_iter.u
        param = this_iter.p

        for i in range(len(self.qp.stages)):
            u_bar[i] = self.feedback.compute_control(i, x_bar[i])
            A, B, b = self.qp.stages[i].A, self.qp.stages[i].B, self.qp.stages[i].b
            x_bar[i+1] = b + A@x_bar[i] + B@u_bar[i]
        self.solution = iterate(x_bar, u_bar, param)
        return self.solution

    def _backward_sweep(self):
        horizon = len(self.qp.stages)
        nu, nx = self.qp.stages[0].S.shape

        self.feedback = Feedback(
            P = np.empty((horizon+1, nx, nx), dtype=float),
            p = np.empty((horizon+1, nx), dtype=float), 
            K = np.empty((horizon, nu, nx), dtype=float), 
            k = np.empty((horizon, nu), dtype=float)
        )

        self.feedback.P[-1] = self.qp.terminal.Q_e
        self.feedback.p[-1] = self.qp.terminal.q_e
        for i in reversed(range(horizon)):
            self.feedback.P[i], self.feedback.p[i], self.feedback.K[i], self.feedback.k[i] = self.riccati_recursion(
               self.qp.stages[i], (self.feedback.P[i+1], self.feedback.p[i+1])
            )

    def riccati_recursion(self,
        qp_stage:QP_stage, qp_terminal:Feedback
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''The core step of Riccati recursion.'''
        Q, R, S, q, r, A, B, b = qp_stage
        P, p = qp_terminal

        K = -np.linalg.solve((R + B.T@P@B), (S + B.T@P@A))
        k = -np.linalg.solve((R + B.T@P@B), (r + B.T@(P@b + p)))
        new_P = Q + A.T@P@A - (S.T + A.T@P@B)@(-K)
        new_p = q + A.T@(P@b + p) - (S.T + A.T@P@B)@(-k)
        return new_P, new_p, K, k

    def get_cost(self):
        cost = 0.5 * (self.solution.x[0].T @ self.feedback.P[0] @ self.solution.x[0] 
                      + 2 * self.feedback.p[0].T @ self.solution.x[0])
        return cost