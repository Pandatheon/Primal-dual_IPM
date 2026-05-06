import numpy as np
import copy
from QP_template.QP_base import BaseQP

def stage_regularization(qp, reg_config, reduced_hessian, P_prev, stage) -> np.ndarray:
    R_k = qp.R[stage]
    B_k = qp.B[stage]
    method = reg_config.method.upper().split('_')[-1]
    if reg_config.method in ['M_STAGE_PROJECT', 'M_STAGE_MIRROR']:
        # EM M only
        eigvals_M, eigvecs_M = np.linalg.eig(reduced_hessian)
        eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method=method)
        P_reg = P_prev
        reduced_hessian_reg = eigvecs_M @ np.diag(eigvals_M_reg) @ eigvecs_M.T

    elif reg_config.method in ['PM_STAGE_PROJECT', 'PM_STAGE_MIRROR']:
        # EM P first
        eigvals_P, eigvecs_P = np.linalg.eig(P_prev)
        eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method=method)
        P_reg = eigvecs_P @ np.diag(eigvals_P_reg) @ eigvecs_P.T
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        eigvals_M, eigvecs_M = np.linalg.eig(reduced_hessian_P_reg)
        eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method=method)
        reduced_hessian_reg = eigvecs_M @ np.diag(eigvals_M_reg) @ eigvecs_M.T

    elif reg_config.method in ['RP_STAGE_PROJECT', 'RP_STAGE_MIRROR']:
        # EM R first
        eigvals_R, eigvecs_R = np.linalg.eig(R_k)
        eigvals_R_reg = do_regularization(eigvals_R, reg_config.delta, method=method)
        R_reg = eigvecs_R @ np.diag(eigvals_R_reg) @ eigvecs_R.T

        # EM P next
        eigvals_P, eigvecs_P = np.linalg.eig(P_prev)
        eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method=method)
        P_reg = eigvecs_P @ np.diag(eigvals_P_reg) @ eigvecs_P.T

        reduced_hessian_reg = R_reg + B_k.T@P_reg@B_k

    return reduced_hessian_reg, P_reg

def do_regularization(eigvals: np.ndarray, 
                      delta: float,
                      method: str)-> np.ndarray:
    if method == 'PROJECT':
        eigvals_reg = np.array([max(eigval, delta) for eigval in eigvals])
    elif method == 'MIRROR':
        eigvals_reg = np.array([max(abs(eigval), delta) for eigval in eigvals])
    return eigvals_reg

def regularize_QP(
        qp : BaseQP,
        reg_config)-> BaseQP:
    '''Regularize the modified QP to ensure positive definiteness of the reduced Hessian.'''

    regularized_QP = copy.deepcopy(qp)
    for k in range(qp.N):
        H = qp.Hessian_block(k)

        eigvals, eigvecs = np.linalg.eig(H)
        eigvals_reg = do_regularization(eigvals, reg_config.delta, reg_config.method)
        H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

        Q_reg = H_reg[:qp.nx, :qp.nx]
        R_reg = H_reg[qp.nx:, qp.nx:]
        S_reg = H_reg[qp.nx:, :qp.nx]
        regularized_QP.Q[k] = Q_reg
        regularized_QP.R[k] = R_reg
        regularized_QP.S[k] = S_reg

    Q_e = qp.Q_e
    eigvals_e, eigvecs_e = np.linalg.eig(Q_e)
    eigvals_e_reg = do_regularization(eigvals_e, reg_config.delta, reg_config.method)
    Q_e_reg = eigvecs_e @ np.diag(eigvals_e_reg) @ eigvecs_e.T
    regularized_QP.Q_e = Q_e_reg

    return regularized_QP