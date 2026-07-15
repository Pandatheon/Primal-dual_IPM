"""
Analysis of an HPIPM/acados-style OCP-QP dump (N=25, nx=4, nu=1).

Does two things:
  (A) Structural diagnostics  -- no solve required, pure data:
        * stagewise Hessian inertia (is it non-convex?)
        * equality-Jacobian rank (does LICQ hold? -> dual degeneracy?)
        * reduced-Hessian inertia on the feasible control subspace (SOSC)
  (B) Condensed solve         -- the part the user asked to see:
        * eliminate dynamics by forward simulation  w = w0 + Gam @ u
        * solve the box-constrained QP in the 25 controls (terminal x fixed)
        * report the FULL objective (incl. the constant I dropped earlier)
          so the cost is comparable to the log's "Optimal cost".

Variable ordering used internally:  per stage k -> (u_k, x_k), then x_N at the end.
  w = [u0, x0, u1, x1, ..., u24, x24,  x25]   (length 129)
The reduced-Hessian inertia and the Jacobian rank are invariant to this choice.
"""

import json
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, Bounds, LinearConstraint

PATH = "/home/jingtao/Mehrotra_IPM/tests/QP_json/temp/acrobot_FU_INDE_RE_INDE_status_2_iter_20.json"
d = json.load(open(PATH))
def G(k):  # fetch a matrix by key as float ndarray
    return np.array(d[k], dtype=float)

N, nx, nu = 25, 4, 1
ns = nu + nx          # 5 vars per stage (u,x)
nv = N * ns + nx      # 129 total primal vars

def ou(k):            # offset of u_k
    return ns * k
def ox(k):            # offset of x_k (x_N lives at the tail)
    return 125 if k == N else ns * k + nu


# ---------------------------------------------------------------------------
# Build full Hessian H and gradient g
# stage cost: 0.5 [u;x]^T [[R, S],[S^T, Q]] [u;x] + [r;q]^T [u;x]
# ---------------------------------------------------------------------------
H = np.zeros((nv, nv))
g = np.zeros(nv)
for k in range(N):
    R = G(f"R_{k:02d}"); S = G(f"S_{k:02d}"); Q = G(f"Q_{k:02d}")
    idx = [ou(k)] + [ox(k) + i for i in range(nx)]
    H[np.ix_(idx, idx)] += np.block([[R, S], [S.T, Q]])
    g[ou(k)]            = G(f"r_{k:02d}")[0, 0]
    g[ox(k):ox(k)+nx]   = G(f"q_{k:02d}").ravel()
idxN = [ox(N) + i for i in range(nx)]
H[np.ix_(idxN, idxN)] += G("Q_25")
g[ox(N):ox(N)+nx]      = G("q_25").ravel()


# ---------------------------------------------------------------------------
# Forward simulation:  given controls u, return the full primal vector w
# (this *exactly* enforces dynamics x_{k+1}=A x_k + B u_k + b and the fixed x0)
# ---------------------------------------------------------------------------
x0 = G("lbx_00").ravel()                      # lbx_00 == ubx_00  -> x0 fixed
def w_of_u(u):
    w = np.zeros(nv)
    x = x0.copy()
    for k in range(N):
        w[ou(k)]          = u[k]
        w[ox(k):ox(k)+nx] = x
        x = G(f"A_{k:02d}") @ x + G(f"B_{k:02d}").ravel() * u[k] + G(f"b_{k:02d}").ravel()
    w[ox(N):ox(N)+nx] = x
    return w

def J_full(u):                                # TRUE objective, comparable to log
    w = w_of_u(u)
    return 0.5 * w @ H @ w + g @ w


# ===========================================================================
# (A) STRUCTURAL DIAGNOSTICS
# ===========================================================================
print("="*70)
print("(A) STRUCTURAL DIAGNOSTICS")
print("="*70)

# --- stagewise Hessian inertia ---
neg_stage = 0
for k in range(N):
    Hk = np.block([[G(f"R_{k:02d}"), G(f"S_{k:02d}")],
                   [G(f"S_{k:02d}").T, G(f"Q_{k:02d}")]])
    w = np.linalg.eigvalsh(Hk)
    if (w < -1e-9).any():
        neg_stage += 1
print(f"stages with negative curvature in [[R,S],[S^T,Q]] : {neg_stage}/{N}")

# --- equality Jacobian: dynamics (100) + x0 fixed (4) + xN fixed (4) ---
# terminal is an EQUALITY because lbx_25 == ubx_25
rows = []
for k in range(N):
    A = G(f"A_{k:02d}"); B = G(f"B_{k:02d}")
    for r in range(nx):
        row = np.zeros(nv)
        row[ox(k+1)+r] = 1.0
        for c in range(nx):
            row[ox(k)+c] = -A[r, c]
        row[ou(k)] = -B[r, 0]
        rows.append(row)
for r in range(nx):                            # x0 fixed
    row = np.zeros(nv); row[ox(0)+r] = 1.0; rows.append(row)
for r in range(nx):                            # xN fixed
    row = np.zeros(nv); row[ox(N)+r] = 1.0; rows.append(row)
C = np.array(rows)
sv = np.linalg.svd(C, compute_uv=False)
print(f"equality Jacobian C: shape={C.shape}, rank={np.linalg.matrix_rank(C)}, "
      f"sigma_min={sv[-1]:.3e}, cond={sv[0]/sv[-1]:.1f}")
print(f"  -> LICQ {'HOLDS' if sv[-1] > 1e-9 else 'FAILS'}  (box-constraint grads are "
      f"unit vectors, so bounds cannot add rank deficiency)")

# --- reduced Hessian on the feasible (control) subspace ---
Z  = null_space(C)            # (nv x 21)
Hr = Z.T @ H @ Z
wr = np.linalg.eigvalsh(Hr)
print(f"reduced Hessian (dim {Hr.shape[0]}): inertia +{(wr>1e-9).sum()}/-{(wr<-1e-9).sum()}/"
      f"0:{(np.abs(wr)<=1e-9).sum()}")
print(f"  most negative eig = {wr[0]:+.4e},  smallest positive = {wr[wr>1e-9].min():+.4e}")


# ===========================================================================
# (B) CONDENSED SOLVE
# ===========================================================================
print("="*70)
print("(B) CONDENSED BOX-QP SOLVE")
print("="*70)

# build w0 and the response matrix Gam so that w(u) = w0 + Gam @ u
w0  = w_of_u(np.zeros(N))
Gam = np.zeros((nv, N))
for j in range(N):
    e = np.zeros(N); e[j] = 1.0
    Gam[:, j] = w_of_u(e) - w0

# condensed objective:  0.5 u^T Hc u + gc^T u + const
Hc    = Gam.T @ H @ Gam
gc    = Gam.T @ (H @ w0 + g)
const = 0.5 * w0 @ H @ w0 + g @ w0           # <-- the term I dropped before
print(f"dropped constant  0.5*w0'H w0 + g'w0 = {const:+.6e}")

# bounds and terminal-equality (x_N must equal target = lbx_25 = ubx_25)
lbu = np.array([G(f"lbu_{k:02d}")[0, 0] for k in range(N)])
ubu = np.array([G(f"ubu_{k:02d}")[0, 0] for k in range(N)])
ExN = Gam[ox(N):ox(N)+nx, :]                  # d x_N / d u
xN0 = w0[ox(N):ox(N)+nx]
tgt = G("lbx_25").ravel()

res = minimize(
    fun=lambda u: 0.5 * u @ Hc @ u + gc @ u,
    x0=np.zeros(N),
    jac=lambda u: Hc @ u + gc,
    method="trust-constr",
    bounds=Bounds(lbu, ubu),
    constraints=[LinearConstraint(ExN, tgt - xN0, tgt - xN0)],  # equality band
    options={"maxiter": 3000, "gtol": 1e-12, "xtol": 1e-14, "verbose": 0},
)
u = res.x
w = w_of_u(u)

print(f"\nquadratic-only value 0.5u'Hc u+gc u = {res.fun:+.6e}   (NOT comparable to log)")
print(f"FULL objective J_full(u*)           = {J_full(u):+.6e}   (comparable to log)")
print(f"log 'Optimal cost'                  = -1.5436e-02")
print("\nfeasibility / stationarity of u*:")
print(f"  ||x_25 - target||      = {np.linalg.norm(w[ox(N):ox(N)+nx]-tgt):.3e}")
print(f"  bounds satisfied       = {np.all(u>=lbu-1e-9) and np.all(u<=ubu+1e-9)}")
print(f"  ||projected gradient|| = {res.optimality:.3e}")
print(f"  # controls at a bound  = {np.isclose(u,lbu,atol=1e-5).sum()+np.isclose(u,ubu,atol=1e-5).sum()}/{N}")

# NOTE: trust-constr on a non-convex QP converges to *a* KKT point, not
# necessarily a global/the-same-as-log minimizer. This solve is only a
# sanity probe; the load-bearing evidence is block (A), which needs no solve.