import numpy as np
from dataclasses import dataclass
from typing import Tuple
from QP_formulation import QP_stage
from Regularizations import Regularizations_config
from solvers.Regularization import stage_regularization


@dataclass
class Feedback:
    """QP terminal cost parameters arranged as (P, p)"""
    P: np.ndarray
    p: np.ndarray
    K: np.ndarray
    k: np.ndarray

    def compute_control(self, index: int, state: np.ndarray) -> np.ndarray:
        '''compute the control according to the feedback law u = k + Kx'''
        return self.K[index] @ state + self.k[index]


def riccati_recursion_with_stage_regularization(
    qp_stage:QP_stage, qp_terminal:Feedback, 
    nx,
    rg, rb, 
    reg_config,
    recorder, index:int=0,
    reg:str='NO_REGULARIZE'
)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''The core step of Riccati recursion.'''
    Q, R, S, q, r, A, B, b, C, D, d = qp_stage
    P, p = qp_terminal
    q = rg[:nx]
    r = rg[nx:]
    b = rb

    try:
        recorder.record_stage_condition_numbers(index, P, R + B.T @ P @ B)

        R_chol = np.linalg.cholesky(R + B.T@P@B)
        y1 = np.linalg.solve(R_chol, S + B.T@P@A)
        y2 = np.linalg.solve(R_chol, r + B.T@(P@b + p))
        K = -np.linalg.solve(R_chol.T, y1)
        k = -np.linalg.solve(R_chol.T, y2)
        new_P = Q + A.T@P@A - (S.T + A.T@P@B)@(-K)
        new_p = q + A.T@(P@b + p) - (S.T + A.T@P@B)@(-k)

        return new_P, new_p, K, k

    except np.linalg.LinAlgError:
        if reg == 'NO_REGULARIZE':
            raise np.linalg.LinAlgError("R + B'PB is not positive definite. Consider using regularization.")

        elif reg in reg_config.stage_based_regularization:
            try:
                reduced_hessian_reg, P_reg = stage_regularization(reg, reg_config, R, P, B, recorder, index)
            except np.linalg.LinAlgError:
                recorder.finalize()
                raise np.linalg.LinAlgError("Regularization failed to make the reduced Hessian positive definite.")

            R_chol = np.linalg.cholesky(reduced_hessian_reg)
            y1 = np.linalg.solve(R_chol, S + B.T@P_reg@A)
            y2 = np.linalg.solve(R_chol, r + B.T@(P_reg@b + p))
            K = -np.linalg.solve(R_chol.T, y1)
            k = -np.linalg.solve(R_chol.T, y2)
            new_P = Q + A.T@P_reg@A - (S.T + A.T@P_reg@B)@(-K)
            new_p = q + A.T@(P_reg@b + p) - (S.T + A.T@P_reg@B)@(-k)
            return new_P, new_p, K, k

        else:
            raise np.linalg.LinAlgError("Unkonwn error happened")