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
    L[j + 1:, j] = A_mod[j + 1:, j] / D[j]
    A_mod[j + 1:, j + 1:] -= D[j] * np.outer(L[j + 1:, j], L[j + 1:, j])

def modified_cholesky_SE(A, mu=0.1):
    """
    Compute the modified Cholesky factorization (SE99) with MIRROR_H:
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