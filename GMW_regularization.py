import numpy as np

def gmw_modified_cholesky(A):
    """
    Performs the Gill-Murray-Wright (GMW) modified Cholesky factorization.
    
    Inputs:
        A: Symmetric matrix (numpy array, n x n)
        
    Outputs:
        L: Unit lower triangular matrix (numpy array, n x n)
        D: Diagonal elements of the D matrix (numpy array, n)
        E: Correction matrix (numpy array, n x n), such that A + E = L * D * L.T
    
    Reference:
        Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). 
        Algorithm 3.4 (p. 51).
    """
    n = A.shape[0]
    
    # 1. Initialization
    L = np.eye(n)
    D = np.zeros(n)        # Stores the diagonal elements of D
    C = A.copy().astype(float) # C stores the Schur complements during updates
    E = np.zeros((n, n))   # Diagonal correction matrix
    
    # 2. Calculate algorithm parameters: gamma, xi, beta
    # gamma: Maximum absolute value of the diagonal elements of A
    gamma = np.max(np.abs(np.diag(A)))
    
    # xi: Maximum absolute value of the off-diagonal elements of A
    off_diag_mask = ~np.eye(n, dtype=bool)
    if np.any(off_diag_mask):
        xi = np.max(np.abs(A[off_diag_mask]))
    else:
        xi = 0.0

    # Machine epsilon
    eps = np.finfo(float).eps
    
    # delta: Minimum threshold to ensure strict positive definiteness
    delta = eps * max(gamma + xi, 1.0)
    
    # beta: Parameter to ensure that the elements of the L matrix are bounded.
    # If elements of L are too large, the factorization becomes numerically unstable.
    if n > 1:
        beta = np.sqrt(max(gamma, xi / np.sqrt(n**2 - 1), eps))
    else:
        beta = np.sqrt(max(gamma, eps))

    # 3. Main Loop (Column-wise factorization)
    for j in range(n):
        # --- Step A: Calculate theta_j ---
        # theta_j is the maximum absolute value in the remaining part of current column j
        if j < n - 1:
            theta_j = np.max(np.abs(C[j+1:, j]))
        else:
            theta_j = 0.0
        
        # --- Step B: Calculate the modified diagonal element d_j ---
        # Logic: d_j must be large enough to satisfy three conditions:
        # 1. Maintain the scale of the original diagonal element (|C[j,j]|)
        # 2. Prevent elements in L from exploding ((theta_j / beta)^2)
        # 3. Ensure the matrix is strictly positive definite (delta)
        
        d_j = max(abs(C[j, j]), (theta_j / (beta))**2, delta) # herustic factor 2 for stability
        D[j] = d_j
        
        # Record the correction value in E
        E[j, j] = d_j - C[j, j]
        
        # --- Step C: Update L and the remaining submatrix (Schur Complement) ---
        for i in range(j + 1, n):
            # Calculate L element: l_ij = c_ij / d_j
            L[i, j] = C[i, j] / d_j
            
            # Update the remaining Schur complement in a vectorized manner
            # This propagates the elimination effect to the bottom-right submatrix
            C[i, j+1:n] = C[i, j+1:n] - L[i, j] * C[j+1:n, j]

    return L, D, E

# --- Test Case ---
if __name__ == "__main__":
    # Construct a non-positive definite matrix
    # Eigenvalues are approximately: 3.56, 0.44, -1.0
    A_non_pd = np.array([
        [2.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])

    print("Original Matrix A:")
    print(A_non_pd)

    # Execute GMW algorithm
    L, D, E = gmw_modified_cholesky(A_non_pd)

    print("\nCorrection Matrix E (Diagonal):")
    print(np.diag(np.diag(E))) # Display as diagonal for clarity
    
    print("\nModified Matrix (A + E):")
    A_mod = A_non_pd + E
    print(A_mod)
    
    print("\nReconstructed Matrix (L * D * L.T):")
    reconstructed = L @ np.diag(D) @ L.T
    print(reconstructed)
    
    # Verification of reconstruction error
    error = np.max(np.abs(reconstructed - A_mod))
    print(f"\nReconstruction Error: {error:.2e}")
    
    # Verification of Positive Definiteness
    eigvals = np.linalg.eigvals(A_mod)
    print(f"Eigenvalues of (A + E): {eigvals}")