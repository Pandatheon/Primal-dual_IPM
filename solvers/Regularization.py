import numpy as np
import copy
from QP_template.QP_base import BaseQP
from utils.modified_Cholesky import modified_cholesky_SE

def _safe_sym_eig(matrix: np.ndarray, tol: float = 1000.0):
    """Compute eig on a symmetrized, real-cast matrix to avoid complex drift."""
    matrix_real = np.real_if_close(matrix, tol=tol)
    matrix_sym = 0.5 * (matrix_real + matrix_real.conj().T)
    eigvals, eigvecs = np.linalg.eig(matrix_sym)
    eigvals = np.real_if_close(eigvals, tol=tol)
    eigvecs = np.real_if_close(eigvecs, tol=tol)
    return eigvals, eigvecs

def _to_real(matrix: np.ndarray, tol: float = 1000.0) -> np.ndarray:
    """Force a real array when imaginary parts are numerical noise."""
    matrix_real = np.real_if_close(matrix, tol=tol)
    if np.iscomplexobj(matrix_real):
        return np.real(matrix_real)
    return matrix_real

def stage_regularization(qp, feedback, reduced_hessian, reg_config, stage) -> np.ndarray:
    R_k = qp.R[stage]
    B_k = qp.B[stage]
    P_prev = feedback.P[stage+1]

    method = reg_config.method.upper().split('_')[-1]

    if reg_config.method == 'PM_STAGE_MIRROR':
        # EM P first
        eigvals_P, eigvecs_P = _safe_sym_eig(P_prev)
        eigvals_P_reg = do_regularization(eigvals_P, reg_config.delta, method=method)
        P_reg = eigvecs_P @ np.diag(eigvals_P_reg) @ eigvecs_P.conj().T
        P_reg = _to_real(P_reg)
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        eigvals_M, eigvecs_M = _safe_sym_eig(reduced_hessian_P_reg)
        eigvals_M_reg = do_regularization(eigvals_M, reg_config.delta, method=method)
        reduced_hessian_reg = eigvecs_M @ np.diag(eigvals_M_reg) @ eigvecs_M.conj().T
        reduced_hessian_reg = _to_real(reduced_hessian_reg)

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
            if Nc == 0:
                return False
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
        eigvals, eigvecs = _safe_sym_eig(H)
        eigvals_reg = do_regularization(eigvals, reg_config.delta, reg_config.method)
        H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.conj().T
        H_reg = _to_real(H_reg)

        Q_reg = _to_real(H_reg[:qp.nx, :qp.nx])
        R_reg = _to_real(H_reg[qp.nx:, qp.nx:])
        S_reg = _to_real(H_reg[qp.nx:, :qp.nx])
        regularized_QP.Q[k] = Q_reg
        regularized_QP.R[k] = R_reg
        regularized_QP.S[k] = S_reg

    Q_e = qp.Q_e
    eigvals_e, eigvecs_e = _safe_sym_eig(Q_e)
    eigvals_e_reg = do_regularization(eigvals_e, reg_config.delta, reg_config.method)
    Q_e_reg = eigvecs_e @ np.diag(eigvals_e_reg) @ eigvecs_e.conj().T
    Q_e_reg = _to_real(Q_e_reg)
    regularized_QP.Q_e = Q_e_reg

    return regularized_QP