import numpy as np
import copy
from QP_template.QP_base import BaseQP
from utils.modified_Cholesky import modified_cholesky_SE, modified_cholesky_GMW

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


    if reg_config.method == 'PM_MIRROR':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="MIRROR")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="MIRROR")

    elif reg_config.method == 'PM_PROJECT':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="PROJECT")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="PROJECT")

    elif reg_config.method == 'PM_GMW':
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        L_P, D_P, E_P, p_P = modified_cholesky_GMW(P_prev)
        inv_p_P = np.argsort(p_P)
        P_reg = (L_P @ np.diag(D_P) @ L_P.T)[np.ix_(inv_p_P, inv_p_P)]

        reduced_hessian_P_reg = R_k + B_k.T @ P_reg @ B_k

        L, D, E, p = modified_cholesky_GMW(reduced_hessian_P_reg)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)
        g1 = S_k + B_k.T @ P_reg @ A_k
        g1_permuted = g1[p]
        y1 = np.linalg.solve(R_chol, g1_permuted)
        K_permuted = -np.linalg.solve(R_chol.T, y1)
        K_curr = K_permuted[inv_p]
        P_curr = Q_k + A_k.T @ P_reg @ A_k - (S_k.T + A_k.T @ P_reg @ B_k) @ (-K_curr)

        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

    elif reg_config.method == 'M_PROJECT':
        P_reg = P_prev
        reduced_hessian_reg = do_regularization(reduced_hessian, reg_config.delta, method="PROJECT")

    elif reg_config.method == 'M_MIRROR':
        P_reg = P_prev
        reduced_hessian_reg = do_regularization(reduced_hessian, reg_config.delta, method="MIRROR")

    elif reg_config.method == 'M_GMW':
        P_reg = P_prev
        L, D, E, p = modified_cholesky_GMW(reduced_hessian)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)
        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

    #Trial methods below,
    elif reg_config.method == 'RP_STAGE_MIRROR':
        # EM R first
        R_reg = do_regularization(R_k, reg_config.delta, method="MIRROR")
        # EM P next
        P_reg = do_regularization(P_prev, reg_config.delta, method="MIRROR")

        reduced_hessian_reg = R_reg + B_k.T @ P_reg @ B_k

    elif reg_config.method == 'PM_STAGE_SE':
        Nc = 0
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        L_P, D_P, E_P, p_P = modified_cholesky_SE(P_prev)
        inv_p_P = np.argsort(p_P)
        P_reg = (L_P @ np.diag(D_P) @ L_P.T)[np.ix_(inv_p_P, inv_p_P)]

        reduced_hessian = R_k + B_k.T @ P_reg @ B_k

        L, D, E, p = modified_cholesky_SE(reduced_hessian)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)

        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

    elif reg_config.method == 'RP_STAGE_SE':
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        L_R, D_R, E_R, p = modified_cholesky_SE(R_k)
        R_reg = (L_R @ np.diag(D_R) @ L_R.T)[np.ix_(np.argsort(p), np.argsort(p))]
        L_P, D_P, E_P, p_P = modified_cholesky_SE(P_prev)
        P_reg = (L_P @ np.diag(D_P) @ L_P.T)[np.ix_(np.argsort(p_P), np.argsort(p_P))]

        reduced_hessian_reg = R_reg + B_k.T @ P_reg @ B_k

        R_chol = np.linalg.cholesky(reduced_hessian_reg)
        inv_p = np.argsort(p)

        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]

    elif reg_config.method == 'M_STAGE_SE':
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]
        flag = False

        def is_strictly_diagonally_dominant(A):
            diag = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag
            return np.all(diag > row_sums)

        def is_irreducibly_diagonally_dominant(A):
            from scipy.sparse.csgraph import connected_components
            diag = np.abs(np.diag(A))
            row_sums = np.sum(np.abs(A), axis=1) - diag
            if not np.all(diag >= row_sums):
                return False
            if not np.any(diag > row_sums):
                return False
            adj_matrix = (A != 0).astype(int)
            n_components = connected_components(adj_matrix, connection='strong', return_labels=False)
            
            return n_components == 1

        P_reg = P_prev
        L, D, E, p = modified_cholesky_SE(reduced_hessian)
        R_chol = L * np.sqrt(D)
        inv_p = np.argsort(p)
        reduced_hessian_reg = (R_chol @ R_chol.T)[np.ix_(inv_p, inv_p)]
        return reduced_hessian_reg, P_reg, flag

    elif reg_config.method == 'M_lookahead_CM':
        Nc = 1
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

    elif reg_config.method == 'M_STAGE_MIRROR':
        A_k = qp.A[stage]
        Q_k = qp.Q[stage]
        S_k = qp.S[stage]

        P_reg = P_prev

        reduced_hessian_reg = R_k + B_k.T @ P_reg @ B_k
        reduced_hessian_reg = do_regularization(reduced_hessian_reg, reg_config.delta, method="MIRROR")

    elif reg_config.method == 'PM_STAGE_PROJECT':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="PROJECT")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="PROJECT")

    elif reg_config.method == 'PM_STAGE_MIPROJECT':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="MIRROR")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="PROJECT")

    elif reg_config.method == 'PM_STAGE_PROMIRROR':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="PROJECT")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="MIRROR")

    elif reg_config.method == 'PM_STAGE_MINPROJECT':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="MINPROJECT")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="MINPROJECT")

    elif reg_config.method == 'PM_STAGE_TUNEDMIRROR':
        # EM P first
        P_reg = do_regularization(P_prev, reg_config.delta, method="TUNEDMIRROR")
        reduced_hessian_P_reg = R_k + B_k.T@P_reg@B_k

        #  EM modified M next
        reduced_hessian_reg = do_regularization(reduced_hessian_P_reg, reg_config.delta, method="TUNEDMIRROR")

    return reduced_hessian_reg, P_reg

def do_regularization(mat: np.ndarray,
                      delta: float,
                      method: str)-> np.ndarray:
    eigvals, eigvecs = _safe_sym_eig(mat)
    if method == 'PROJECT':
        eigvals_reg = np.maximum(eigvals, delta)
    elif method == 'MIRROR':
        eigvals_reg = np.maximum(np.abs(eigvals), delta)

    ## trial methods below
    elif method == 'MINPROJECT':
        min_pos_eigval = np.min(eigvals[eigvals > 0]) if np.any(eigvals > 0) else delta
        min_pos_eigval = max(min_pos_eigval, delta)
        eigvals_reg = np.maximum(eigvals, min_pos_eigval)
    elif method == 'TUNEDMIRROR':
        eigvals_scaled = np.where(eigvals < 0, 0.1 *eigvals, eigvals)
        eigvals_reg = np.maximum(np.abs(eigvals_scaled), delta)

    mat_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.conj().T
    mat_reg = _to_real(mat_reg)
    return mat_reg

def regularize_QP(
        qp : BaseQP,
        reg_config)-> BaseQP:
    '''Regularize the modified QP to ensure positive definiteness of the reduced Hessian.'''

    regularized_QP = copy.deepcopy(qp)
    for k in range(qp.N + 1):
        H = qp.Hessian_block(k) if k < qp.N else qp.Q_e

        if reg_config.method =='H_PROJECT' or reg_config.method == 'PRECOMPUTE':
            H_reg = do_regularization(H, reg_config.delta, 'PROJECT')
        elif reg_config.method == 'H_MIRROR':
            H_reg = do_regularization(H, reg_config.delta, 'MIRROR')
        elif reg_config.method == 'H_GMW':
            L, D, E, p = modified_cholesky_GMW(H)
            p_inv = np.argsort(p)
            H_reg = (L @ np.diag(D) @ L.T)[np.ix_(p_inv, p_inv)]
        else:
            raise ValueError(f"Unknown regularization method: {reg_config.method}, set reg_config.method to 'PROJECT_H', 'MIRROR_H' or 'CHOLESKY_H'")

        if k < qp.N:
            regularized_QP.Q[k] = H_reg[:qp.nx[k], :qp.nx[k]]
            regularized_QP.R[k] = H_reg[qp.nx[k]:, qp.nx[k]:]
            regularized_QP.S[k] = H_reg[qp.nx[k]:, :qp.nx[k]]
        else:
            regularized_QP.Q_e = H_reg

    return regularized_QP