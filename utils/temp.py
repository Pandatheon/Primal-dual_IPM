"""
QP funnel figure -> vector PDF (funnel_qp.pdf) for a paper.
Only needs numpy + matplotlib.  Edit F, beta_tau, tau to reshape.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
 
# --- feasible polytope F : one vertex per row (any convex polygon) ---
F = np.array([[520,110],[700,128],[762,250],[690,384],[508,362],[452,232]], float)
 
beta_tau = 26.0   # inner  boundary  h = beta*tau   (dashed envelope)
tau      = 54.0   # outer  boundary  h = tau        (funnel radius)
 
def offset(P, d):
    """Push every facet of a convex polygon outward by d (h <= d level set)."""
    C = P.mean(0); n = len(P); pts, dirs = [], []
    for i in range(n):
        A, B = P[i], P[(i+1) % n]
        e = B - A; nrm = np.array([e[1], -e[0]]); nrm = nrm/np.linalg.norm(nrm)
        if nrm @ ((A+B)/2 - C) < 0: nrm = -nrm
        pts.append(A + d*nrm); dirs.append(e)
    out = []
    for i in range(n):
        p1, d1 = pts[(i-1) % n], dirs[(i-1) % n]
        p2, d2 = pts[i], dirs[i]
        M = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        t = np.linalg.solve(M, p2 - p1)[0]
        out.append(p1 + t*d1)
    return np.array(out)
 
inner, outer = offset(F, beta_tau), offset(F, tau)
 
# world coords: flip y so the figure is upright
def flip(P): return np.column_stack([P[:,0], -P[:,1]])
Ff, Inf, Ouf = flip(F), flip(inner), flip(outer)
 
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.add_patch(Polygon(Ouf, closed=True, fc="#ECEEF0", ec="none", zorder=1))
ax.add_patch(Polygon(Ff,  closed=True, fc="#C6CACE", ec="none", zorder=2))
ax.add_patch(Polygon(Ouf, closed=True, fill=False, ec="black", lw=1.4, zorder=3))
ax.add_patch(Polygon(Inf, closed=True, fill=False, ec="black", lw=1.1, ls=(0,(5,4)), zorder=3))
ax.add_patch(Polygon(Ff,  closed=True, fill=False, ec="black", lw=1.4, zorder=3))
 
cF = Ff.mean(0)
ax.text(cF[0], cF[1], r"$\mathcal{F}$", ha="center", va="center",
        fontsize=22, fontstyle="italic")
 
def facet_anchor(P, i):
    A, B = P[i], P[(i+1) % len(P)]; C = P.mean(0)
    m = (A+B)/2; e = B-A; nrm = np.array([e[1], -e[0]]); nrm = nrm/np.linalg.norm(nrm)
    if nrm @ (m-C) < 0: nrm = -nrm
    return m, nrm
 
m, nrm = facet_anchor(Ouf, 1); e = m + nrm*70
ax.plot([m[0], e[0]], [m[1], e[1]], color="black", lw=0.7)
ax.text(e[0]+6, e[1], r"$\theta\leq\theta_{\max}$", ha="left", va="center", fontsize=13)
# inner label
m, nrm = facet_anchor(Inf, 2); e = m + nrm*115
ax.plot([m[0], e[0]], [m[1], e[1]], color="black", lw=0.7)
ax.text(e[0]+6, e[1], r"$\theta\leq\theta_{\max}^{+}$", ha="left", va="center", fontsize=13)
# linear-constraint annotation
m, nrm = facet_anchor(Ff, 4)
ax.plot([m[0]-140, m[0]], [m[1], m[1]], color="gray", lw=0.6)
ax.text(m[0]-150, m[1], r"$Ax\leq b$", ha="right", va="center", fontsize=12, color="gray")
 
ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.16)
fig.tight_layout(pad=0.2)
fig.savefig("funnel_qp.pdf")          # vector PDF for the paper
fig.savefig("funnel_qp_mpl.png", dpi=200)
print("saved funnel_qp.pdf")
 
