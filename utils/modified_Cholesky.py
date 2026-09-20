import numpy as np

def pivot_symmetric(A_mod, j, k, L, p):
    """Swap rows/columns j and k in A_mod and keep L, p consistent."""
    if j == k:
        return
    A_mod[[j, k], :] = A_mod[[k, j], :]
    A_mod[:, [j, k]] = A_mod[:, [k, j]]
    if j > 0:
        L[[j, k], :j] = L[[k, j], :j]
    p[[j, k]] = p[[k, j]]

def eliminate_column(A_mod, j, L, D):
    """Apply one LDL^T elimination step for column j."""
    n = A_mod.shape[0]
    if j >= n - 1:
        return
    # Avoid 0.0 / 0.0 division.
    # If D[j] is exactly 0, the off-diagonal elements are also 0 based on Phase 2 logic.
    if D[j] == 0.0:
        L[j + 1:, j] = 0.0
    else:
        L[j + 1:, j] = A_mod[j + 1:, j] / D[j]
        A_mod[j + 1:, j + 1:] -= D[j] * np.outer(L[j + 1:, j], L[j + 1:, j])

def modified_cholesky_SE(A, mu=0.1):
    """
    Compute the modified Cholesky factorization (SE99) with TYPE-I:
    P(A + E)P^T = LDL^T

    Parameters:
    A (numpy.ndarray): Input n x n symmetric matrix
    mu (float): Parameter for phase 1 switching condition, usually in (0, 1]
    tau_bar (float): Tolerance for maximum diagonal element in phase 1
    tau (float): Tolerance parameter for phase 2 modification

    Returns:
    L (numpy.ndarray): Lower triangular matrix
    D (numpy.ndarray): 1D array representing the diagonal matrix
    E (numpy.ndarray): 1D array representing the modification diagonal matrix
    p (numpy.ndarray): 1D array representing the permutation vector
    """
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    E = np.zeros(n)
    p = np.arange(n)

    A_mod = A.astype(float).copy()
    diag_A = np.diag(A)
    gamma = np.max(diag_A)

    # Initialize default tolerances if not provided.

    tau_bar = np.finfo(float).eps ** (2 / 3)
    tau = np.finfo(float).eps ** (1 / 3)

    phase_one = True
    e_prev = 0.0

    for j in range(n):
        if phase_one:
            # Phase 1 switching condition using remaining diagonal entries.
            remaining_diag = np.diag(A_mod)[j:]
            max_diag = np.max(remaining_diag)
            min_diag = np.min(remaining_diag)

            if max_diag < tau_bar * gamma or min_diag < -mu * max_diag:
                phase_one = False
            else:
                # Phase 1 pivot: choose the largest diagonal entry.
                pivot_idx = j + np.argmax(remaining_diag)
                pivot_symmetric(A_mod, j, pivot_idx, L, p)

                A_jj = A_mod[j, j]
                if j < n - 1:
                    trailing_diag = np.diag(A_mod)[j + 1:]
                    zeta = trailing_diag - (A_mod[j + 1:, j] ** 2) / A_jj
                    if np.min(zeta) < -mu * gamma:
                        phase_one = False
                    else:
                        D[j] = A_jj
                        eliminate_column(A_mod, j, L, D)
                else:
                    D[j] = A_jj

        if not phase_one:
            # Phase 2 pivoting based on Gerschgorin lower bounds.
            if j < n - 1:
                g_bounds = np.zeros(n - j)
                for i in range(j, n):
                    row_abs_sum = np.sum(np.abs(A_mod[i, j:]))
                    off_diag_sum = row_abs_sum - np.abs(A_mod[i, i])
                    g_bounds[i - j] = A_mod[i, i] - off_diag_sum
                pivot_idx = j + np.argmax(g_bounds)
                pivot_symmetric(A_mod, j, pivot_idx, L, p)

            # Phase 2 diagonal modification target.
            if j == n - 1:
                D_target = max(tau * (-A_mod[j, j]) / (1.0 - tau), tau_bar * gamma)
            else:
                theta = np.sum(np.abs(A_mod[j + 1:, j]))
                D_target = max(tau_bar*gamma, theta)

            E[j] = max(0.0, D_target - A_mod[j, j], -2 * A_mod[j, j])
            e_prev = E[j]
            D[j] = A_mod[j, j] + E[j]
            eliminate_column(A_mod, j, L, D)

    return L, D, E, p


def modified_cholesky_SE99(A, mu=0.1, tau_bar=None):
    """
    Compute the modified Cholesky factorization (SE99):
    P(A + E)P^T = LDL^T

    Parameters:
    A (numpy.ndarray): Input n x n symmetric matrix
    mu (float): Parameter for phase 1 switching condition, usually in (0, 1]
    tau_bar (float): Tolerance for maximum diagonal element in phase 1
    tau (float): Tolerance parameter for phase 2 modification

    Returns:
    L (numpy.ndarray): Lower triangular matrix
    D (numpy.ndarray): 1D array representing the diagonal matrix
    E (numpy.ndarray): 1D array representing the modification diagonal matrix
    p (numpy.ndarray): 1D array representing the permutation vector
    """
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    E = np.zeros(n)
    p = np.arange(n)

    A_mod = A.astype(float).copy()
    diag_A = np.diag(A)
    gamma = np.max(diag_A)

    # Initialize default tolerances if not provided.
    if tau_bar is None:
        tau_bar = np.finfo(float).eps ** (2 / 3)
    tau = np.finfo(float).eps ** (1 / 3)

    phase_one = True
    e_prev = 0.0

    for j in range(n):
        if phase_one:
            # Phase 1 switching condition using remaining diagonal entries.
            remaining_diag = np.diag(A_mod)[j:]
            max_diag = np.max(remaining_diag)
            min_diag = np.min(remaining_diag)

            if max_diag < tau_bar * gamma or min_diag < -mu * max_diag:
                phase_one = False
            else:
                # Phase 1 pivot: choose the largest diagonal entry.
                pivot_idx = j + np.argmax(remaining_diag)
                pivot_symmetric(A_mod, j, pivot_idx, L, p)

                A_jj = A_mod[j, j]
                if j < n - 1:
                    trailing_diag = np.diag(A_mod)[j + 1:]
                    zeta = trailing_diag - (A_mod[j + 1:, j] ** 2) / A_jj
                    if np.min(zeta) < -mu * gamma:
                        phase_one = False
                    else:
                        D[j] = A_jj
                        eliminate_column(A_mod, j, L, D)
                else:
                    D[j] = A_jj

        if not phase_one:
            # Phase 2 pivoting based on Gerschgorin lower bounds.
            if j < n - 1:
                g_bounds = np.zeros(n - j)
                for i in range(j, n):
                    row_abs_sum = np.sum(np.abs(A_mod[i, j:]))
                    off_diag_sum = row_abs_sum - np.abs(A_mod[i, i])
                    g_bounds[i - j] = A_mod[i, i] - off_diag_sum
                pivot_idx = j + np.argmax(g_bounds)
                pivot_symmetric(A_mod, j, pivot_idx, L, p)

            # Phase 2 diagonal modification target.
            if j == n - 1:
                D_target = max(tau * (-A_mod[j, j]) / (1.0 - tau), tau_bar * gamma)
            else:
                theta = np.sum(np.abs(A_mod[j + 1:, j]))
                D_target = max(tau_bar*gamma, theta)

            E[j] = max(e_prev, D_target - A_mod[j, j])
            e_prev = E[j]
            D[j] = A_mod[j, j] + E[j]
            eliminate_column(A_mod, j, L, D)

    return L, D, E, p


def modified_cholesky_GMW_(A, mu=0.1):
    """
    Compute the modified Cholesky factorization:
    P(A + E)P^T = LDL^T

    Parameters:
    A (numpy.ndarray): Input n x n symmetric matrix

    Returns:
    L (numpy.ndarray): Lower triangular matrix
    D (numpy.ndarray): 1D array representing the diagonal matrix
    E (numpy.ndarray): 1D array representing the modification diagonal matrix
    p (numpy.ndarray): 1D array representing the permutation vector
    """
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    E = np.zeros(n)
    p = np.arange(n)

    A_mod = A.astype(float).copy()
    diag_A = np.diag(A)
    gamma = np.max(diag_A)

    # Initialize default tolerances if not provided.
    tau_bar = np.finfo(float).eps ** (2 / 3)
    eps = np.finfo(float).eps
    gamma_abs = np.max(np.abs(diag_A))                 # max|A_ii|

    if n > 1:
        xi = np.max(np.abs(A - np.diag(diag_A)))       # max|A_ij|, i≠j
        nu = np.sqrt(n**2 - 1)
    else:
        xi = 0.0
        nu = 1.0

    beta_sq = max(gamma_abs, xi / nu, eps)
    delta  = eps * max(gamma_abs + xi, 1.0)

    phase_one = False  # GMW does not have a separate phase 1, but we can still use the same loop structure for pivoting and modification.

    for j in range(n):
        if phase_one:
            # Phase 1 switching condition using remaining diagonal entries.
            remaining_diag = np.diag(A_mod)[j:]
            max_diag = np.max(remaining_diag)
            min_diag = np.min(remaining_diag)

            if max_diag < tau_bar * gamma or min_diag < -mu * max_diag:
                phase_one = False
            else:
                # Phase 1 pivot: choose the largest diagonal entry.
                pivot_idx = j + np.argmax(remaining_diag)
                pivot_symmetric(A_mod, j, pivot_idx, L, p)

                A_jj = A_mod[j, j]
                if j < n - 1:
                    trailing_diag = np.diag(A_mod)[j + 1:]
                    zeta = trailing_diag - (A_mod[j + 1:, j] ** 2) / A_jj
                    if np.min(zeta) < -mu * gamma:
                        phase_one = False
                    else:
                        D[j] = A_jj
                        eliminate_column(A_mod, j, L, D)
                else:
                    D[j] = A_jj

        if not phase_one:
            # GMW phase-2 modification
            if j < n - 1:
                theta_j = np.max(np.abs(A_mod[j+1:, j]))
            else:
                theta_j = 0.0

            c_jj = A_mod[j, j]
            d_j  = max(delta, abs(c_jj), (theta_j ** 2) / beta_sq)
            E[j] = d_j - c_jj
            D[j] = d_j
            eliminate_column(A_mod, j, L, D)

    return L, D, E, p

def modified_cholesky_GMW(A):
    """
    Compute pivoted GMW modified Cholesky factorization: P(A + E)P^T = LDL^T
    """
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    E = np.zeros(n)
    p = np.arange(n)
    A_mod = A.astype(float).copy()

    # Calculate global constants for GMW modification
    diag_A = np.diag(A)
    gamma_abs = np.max(np.abs(diag_A))

    if n > 1:
        xi = np.max(np.abs(A - np.diag(diag_A)))
        nu = np.sqrt(n**2 - 1)
    else:
        xi = 0.0
        nu = 1.0

    eps = np.finfo(float).eps
    beta_sq = max(gamma_abs, xi / nu, eps)
    delta = eps * max(gamma_abs + xi, 1.0)

    for j in range(n):
        # 1. Pivot: Find max diagonal in the remaining submatrix
        q = j + np.argmax(np.abs(np.diag(A_mod)[j:]))

        # 2. Perform symmetric permutation if needed
        if q != j:
            # Swap permutation vector
            p[j], p[q] = p[q], p[j]

            # Swap rows and columns in A_mod
            A_mod[[j, q], :] = A_mod[[q, j], :]
            A_mod[:, [j, q]] = A_mod[:, [q, j]]

            # Swap rows in L (only applied to previously computed columns)
            if j > 0:
                L[[j, q], :j] = L[[q, j], :j]

        # 3. GMW Modification
        if j < n - 1:
            theta_j = np.max(np.abs(A_mod[j+1:, j]))
        else:
            theta_j = 0.0

        c_jj = A_mod[j, j]
        D[j] = max(delta, abs(c_jj), (theta_j ** 2) / beta_sq)
        E[j] = D[j] - c_jj

        # 4. Eliminate column and update trailing submatrix
        if j < n - 1:
            L[j+1:, j] = A_mod[j+1:, j] / D[j]
            # Rank-1 update for the remaining submatrix
            A_mod[j+1:, j+1:] -= D[j] * np.outer(L[j+1:, j], L[j+1:, j])

    return L, D, E, p

# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    A = np.array([
        [ 1.0,  2.0,  3.0],
        [ 2.0, -1.0,  1.0],
        [ 3.0,  1.0,  0.0]
    ])

    # With pivoting
    L, D, E, p = modified_cholesky_SE(A)
    print("\nWith pivoting:")
    print("L:\n", L)
    print("D:\n", np.diag(D))
    print("E:\n", np.diag(E))
    print("Permutation vector p:\n", p)

    A_pos = L @ np.diag(D) @ L.T
    inv_p = np.argsort(p)
    A_pos = A_pos[np.ix_(inv_p, inv_p)]
    E_permuted = E[inv_p]
    A_reconstructed = A + np.diag(E_permuted)
    print("Difference between original A and reconstructed A (with pivoting):\n", A_pos - A_reconstructed)