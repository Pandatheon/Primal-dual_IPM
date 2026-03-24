import numpy as np
import copy
from QP_template.OCP_QP import OCP_QP_varying, QP_stage, QP_Terminal

def stage_regularization(reg, reg_config,
                         R, P, B, 
                         recorder, index) -> np.ndarray:
    if reg in [reg_config.stage_based_regularization[0], reg_config.stage_based_regularization[3]]:
        # EM M only
        eigvals_M, eigvecs_M = np.linalg.eig(R+ B.T@P@B)

        if reg == reg_config.stage_based_regularization[0]:
            eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method='PROJECT')
        else:
            eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method='MIRROR')

        P_reg = P
        reduced_hessian_reg = eigvecs_M @ np.diag(eigvals_M_reg) @ eigvecs_M.T

        # record
        recorder.record_adjustment_M(index, eigvals_M_reg, eigvals_M, reduced_hessian_reg)
        recorder.increment_reg_count()

    elif reg in [reg_config.stage_based_regularization[1], reg_config.stage_based_regularization[4]]:

        # EM P first
        eigvals_P, eigvecs_P = np.linalg.eig(P)

        if reg == reg_config.stage_based_regularization[1]:
            eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method='PROJECT')
        else:
            eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method='MIRROR')

        P_reg = eigvecs_P @ np.diag(eigvals_P_reg) @ eigvecs_P.T

        # record
        recorder.record_adjustment_P(index, eigvals_P_reg, eigvals_P, P_reg)

        reduced_hessian_P_reg = R + B.T@P_reg@B

        #  EM modified M next
        eigvals_M, eigvecs_M = np.linalg.eig(reduced_hessian_P_reg)

        if reg == reg_config.stage_based_regularization[1]:
            eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method='PROJECT')
        else:
            eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method='MIRROR')

        reduced_hessian_reg = eigvecs_M @ np.diag(eigvals_M_reg) @ eigvecs_M.T

        # record
        eigvals_M_0 = np.linalg.eigvals(R + B.T@P@B)
        recorder.record_adjustment_M(index, eigvals_M_reg, eigvals_M_0, reduced_hessian_reg)
        recorder.increment_reg_count(2)

    elif reg in [reg_config.stage_based_regularization[2], reg_config.stage_based_regularization[5]]:

        # EM R first
        eigvals_R, eigvecs_R = np.linalg.eig(R)

        if reg == reg_config.stage_based_regularization[2]:
            eigvals_R_reg = do_regularization(eigvals_R, reg_config.delta, method='PROJECT')
        else:
            eigvals_R_reg = do_regularization(eigvals_R, reg_config.delta, method='MIRROR')

        R_reg = eigvecs_R @ np.diag(eigvals_R_reg) @ eigvecs_R.T

        # EM P next
        eigvals_P, eigvecs_P = np.linalg.eig(P)

        if reg == reg_config.stage_based_regularization[2]:
            eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method='PROJECT')
        else:
            eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method='MIRROR')

        P_reg = eigvecs_P @ np.diag(eigvals_P_reg) @ eigvecs_P.T
        
        # record
        recorder.record_adjustment_P(index, eigvals_P_reg, eigvals_P, P_reg)

        reduced_hessian_reg = R_reg + B.T@P_reg@B

        # record
        eigvals_M_reg = np.linalg.eigvals(reduced_hessian_reg)
        eigvals_M = np.linalg.eigvals(R + B.T@P@B)
        recorder.record_adjustment_M(index, eigvals_M_reg, eigvals_M, reduced_hessian_reg)
        recorder.increment_reg_count(2)

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
        qp : OCP_QP_varying,
        recorder,
        reg_config,
        method: str = 'PROJECT')-> OCP_QP_varying:
    '''Regularize the modified QP to ensure positive definiteness of the reduced Hessian.'''

    regularized_QP = copy.deepcopy(qp)
    for i in range(qp.N):
        stage = qp.stages[i]
        H = np.block([[stage.Q, stage.S.T],
                        [stage.S, stage.R]])

        eigvals, eigvecs = np.linalg.eig(H)
        eigvals_reg = do_regularization(eigvals, reg_config.delta, method)
        recorder.increment_reg_count()
        H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

        Q_reg = H_reg[:qp.nx, :qp.nx]
        R_reg = H_reg[qp.nx:, qp.nx:]
        S_reg = H_reg[qp.nx:, :qp.nx]
        regularized_QP.stages[i] = QP_stage(
            Q = Q_reg,
            R = R_reg,
            S = S_reg,
            q = stage.q,
            r = stage.r,
            A = stage.A,
            B = stage.B,
            b = stage.b,
            C = stage.C,
            D = stage.D,
            d = stage.d,
        )
    Q_e = qp.terminal.Q_e
    eigvals_e, eigvecs_e = np.linalg.eig(Q_e)
    eigvals_e_reg = do_regularization(eigvals_e, reg_config.delta, method)
    recorder.increment_reg_count()
    Q_e_reg = eigvecs_e @ np.diag(eigvals_e_reg) @ eigvecs_e.T
    regularized_QP.terminal = QP_Terminal(
        Q_e = Q_e_reg,
        q_e = qp.terminal.q_e,
    )
    return regularized_QP