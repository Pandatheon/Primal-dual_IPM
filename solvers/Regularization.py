import numpy as np
import copy
from QP_template.QP_base import BaseQP
from utils.modified_Cholesky import modified_cholesky_SE, modified_cholesky_SE99

def stage_regularization(qp, feedback, reduced_hessian, reg_config, stage) -> np.ndarray:
    R_k = qp.R[stage]
    B_k = qp.B[stage]
    P_prev = feedback.P[stage+1]

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

    elif reg_config.method == 'M_lookahead_CM':
        Nc = 2
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        def _recursion_lookahead(P, stage):
            actual_range = min(Nc, stage + 1)
            start_stage = stage - actual_range + 1
            end_stage = stage
            QP_pcondensed = qp.partial_condense_range(start_idx=start_stage, end_idx=end_stage)
            B_k = QP_pcondensed.B
            R_k = QP_pcondensed.R
            reduced_hessian_next = R_k + B_k.T @ P @ B_k
            try:
                np.linalg.cholesky(reduced_hessian_next)
                return False
            except np.linalg.LinAlgError:
                L, D, E, p = modified_cholesky_SE(reduced_hessian_next)
                return np.max(E)

        P_reg = P_prev

        L, D, E, p = modified_cholesky_SE(reduced_hessian)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)
        g1 = S_k + B_k.T @ P_reg @ A_k
        g1_permuted = g1[p]
        y1 = np.linalg.solve(R_chol, g1_permuted)
        K_permuted = -np.linalg.solve(R_chol.T, y1)
        K_curr = K_permuted[inv_p]
        P_curr = Q_k + A_k.T @ P_reg @ A_k - (S_k.T + A_k.T @ P_reg @ B_k) @ (-K_curr)

        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

        E_term = _recursion_lookahead(P_curr, stage - 1) if stage > 0 else False
        if E_term:
            reduced_hessian_reg = reduced_hessian_reg + E_term * np.eye(reduced_hessian.shape[0])

    elif reg_config.method == 'PM_lookahead_CM':
        Nc = 2
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        def _recursion_lookahead(P, stage):
            actual_range = min(Nc, stage + 1)
            start_stage = stage - actual_range + 1
            end_stage = stage
            QP_pcondensed = qp.partial_condense_range(start_idx=start_stage, end_idx=end_stage)
            B_k = QP_pcondensed.B
            R_k = QP_pcondensed.R
            reduced_hessian_next = R_k + B_k.T @ P @ B_k
            try:
                np.linalg.cholesky(reduced_hessian_next)
                return False
            except np.linalg.LinAlgError:
                L, D, E, p = modified_cholesky_SE(reduced_hessian_next)
                return np.max(E)

        L_P, D_P, E_P, p_P = modified_cholesky_SE(P_prev)
        inv_p_P = np.argsort(p_P)
        P_reg = (L_P @ np.diag(D_P) @ L_P.T)[np.ix_(inv_p_P, inv_p_P)]

        reduced_hessian = R_k + B_k.T @ P_reg @ B_k

        L, D, E, p = modified_cholesky_SE(reduced_hessian)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)
        g1 = S_k + B_k.T @ P_reg @ A_k
        g1_permuted = g1[p]
        y1 = np.linalg.solve(R_chol, g1_permuted)
        K_permuted = -np.linalg.solve(R_chol.T, y1)
        K_curr = K_permuted[inv_p]
        P_curr = Q_k + A_k.T @ P_reg @ A_k - (S_k.T + A_k.T @ P_reg @ B_k) @ (-K_curr)

        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

        E_term = _recursion_lookahead(P_curr, stage - 1) if stage > 0 else False
        if E_term:
            reduced_hessian_reg = reduced_hessian_reg + E_term * np.eye(reduced_hessian.shape[0])

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