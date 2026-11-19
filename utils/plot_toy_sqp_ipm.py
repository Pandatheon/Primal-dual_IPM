"""Two figures on a toy QP from the cold start w = 0.

    min  1/2 (r u0^2 + q1 x1^2 + r u1^2 + q2 x2^2)
    s.t. x_{k+1} = x_k + u_k,  k = 0, 1,   umin <= u0 <= umax

toy_bound_sqp_level, SqpQpSolver with the PROJECT regularization (MIRROR is commented out):
(a) original reduced QP objective (M indefinite),
(b) objective of the first SQP subproblem (M_reg), with the iterates of HPIPM on it,
(c) original reduced QP objective, with the SQP iterates.
toy_bound_ipm_level, QpSolver on the original QP without regularization:
(a) original reduced QP objective (M indefinite),
(b) Newton model of the first step with the first iteration,
(c) original reduced QP objective, with the IPM iterates.
The iterates are shown in (u0, u1), the dynamics gap of w = 0 is not visible there.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import HPacker, VPacker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from acados_template.acados_ocp_qp import AcadosOcpQp
from QP_template.QP_base import BaseQP
from solvers.RiccatiQpSolver import QpSolver
from solvers.SqpQpSolver import SqpQpSolver

XLIM, YLIM = (-1.6, 1.6), (-1.6, 1.0)
START = dict(marker="o", color="k", ms=3.5, ls="none")
EXACT = dict(marker="*", color="C2", ms=9, ls="none")
APPROX = dict(marker="X", color="C3", ms=6.5, ls="none")
ITERATE = dict(marker="o", color="C0", ms=2.2, ls="none")
STEP = dict(color="C0", lw=1.0)
SQP_ITERATE = dict(ITERATE, color="C1")  # SQP iterates in (c) of the SQP figure
SQP_STEP = dict(STEP, color="C1")
# contour lines coloured by objective value, deep blue = low -> cyan -> green -> yellow = high,
# the yellow end is a darker gold so that the outer lines stay visible on white and on the grey shading
RINGS = dict(cmap=LinearSegmentedColormap.from_list(
                 "rings", ["#2b2d8f", "#1665d8", "#159bd3", "#1fb5a8", "#7cc35e", "#e3c33b"]),
             linewidths=0.8, negative_linestyles="solid")


class TracedQpSolver(QpSolver):
    """QpSolver that records u = (u0, u1) after every iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterates = [np.zeros(2)]

    def update_solution(self, prev_sol):
        super().update_solution(prev_sol)
        self.iterates.append(np.concatenate(self.solution.u))


class TracedSqpSolver(SqpQpSolver):
    """SqpQpSolver with the REG regularization that stores its iterates, the mode only
    gives each regularization its own generated code."""
    # REG = "PROJECT"
    REG = "MIRROR"  # switch M_reg in main as well

    def __init__(self, qp, nlp_max_iter, qp_opts=None):
        super().__init__(qp, nlp_max_iter=nlp_max_iter, mode=self.REG.lower(), qp_opts=qp_opts)

    def formulate_parametric_OCP_QP(self, nlp_max_iter):
        ocp, param_manager = super().formulate_parametric_OCP_QP(nlp_max_iter)
        ocp.solver_options.regularize_method = self.REG
        ocp.solver_options.store_iterates = True
        return ocp, param_manager


def toy_ocp_qp(r, q1, q2, x0, umin, umax, u1_bound=None):
    one = lambda value: [[float(value)]]
    data = {
        "Q_0": one(0.0), "Q_1": one(q1), "Q_2": one(q2), "q_2": one(0.0),
        "idxb_0": [[0], [1]], "idxe_0": [[1]], "idxs_rev_0": [[-1, -1]],
        "lbu_0": one(umin), "ubu_0": one(umax), "lbu_mask_0": one(1.0), "ubu_mask_0": one(1.0),
        "lbx_0": one(x0), "ubx_0": one(x0), "lbx_mask_0": one(1.0), "ubx_mask_0": one(1.0),
    }
    for k in range(2):
        data |= {f"A_{k}": one(1.0), f"B_{k}": one(1.0), f"b_{k}": one(0.0),
                 f"R_{k}": one(r), f"S_{k}": one(0.0), f"q_{k}": one(0.0), f"r_{k}": one(0.0)}
    if u1_bound is not None:
        data |= {"idxb_1": [[0]], "idxs_rev_1": [[-1]],
                 "lbu_1": one(-u1_bound), "ubu_1": one(u1_bound), "lbu_mask_1": one(1.0), "ubu_mask_1": one(1.0)}
    return AcadosOcpQp.from_json(json_data=data)


def reduced_qp(stage_weights, T, t):
    """Condense 1/2 w' diag(stage_weights) w with w = T v + t into 1/2 v' M v + c' v."""
    H = np.diag(stage_weights)
    return T.T @ H @ T, T.T @ H @ t


def quad(K, g, v0, v1):
    """1/2 v' K v + g' v, elementwise so that it also works on grids."""
    return 0.5 * (K[0, 0] * v0**2 + 2 * K[0, 1] * v0 * v1 + K[1, 1] * v1**2) + g[0] * v0 + g[1] * v1


def best_u1(u0, M, c):
    """Minimizer over u1 for fixed u0 (the objective is convex in u1)."""
    return -(M[1, 0] * u0 + c[1]) / M[1, 1]


def subproblem_iterates(ocp_qp, qp_iter):
    """Iterates of HPIPM on the first SQP subproblem. HPIPM keeps only its current point, so a
    run stopped after k iterations gives iterate k. The iteration limit is fixed when acados
    generates the code, so every k has its own solver, which ends on nlp_max_iter (status 2)."""
    iterates = [np.zeros(2)]
    for k in range(1, qp_iter + 1):
        qp_opts = TracedSqpSolver.default_qp_opts()
        qp_opts.iter_max = k
        sub = TracedSqpSolver(ocp_qp, nlp_max_iter=1, qp_opts=qp_opts)
        sub.solve()
        iterates.append(sub.get_iterates().as_array("u")[-1, :, 0])
    return np.array(iterates)


def centre_legend_rows(leg, sizes):
    """Restack the entries of a one-row legend into rows of the given sizes, each centred. A single
    legend keeps the space that constrained layout reserves for it, two stacked legends overlap.
    Uses the private box of the legend entries (matplotlib 3.10)."""
    entries = leg._legend_handle_box.get_children()
    starts = np.cumsum([0] + sizes)
    rows = [HPacker(pad=0, sep=leg._legend_handle_box.sep, align="baseline", children=entries[a:b])
            for a, b in zip(starts[:-1], starts[1:])]
    box = VPacker(pad=0, sep=leg.labelspacing * leg._fontsize, align="center", children=rows)
    leg._legend_box._children[1] = leg._legend_handle_box = box
    leg._legend_box.set_figure(leg.get_figure(root=False))


def plot_row(name, panels, U0, U1, umin, umax, v_0, v_exact, v_approx=None, sqp_panel=None):
    """One figure from panels of (title, field, levels, path), v_approx is marked in (b),
    the path of panel sqp_panel is drawn in the SQP colour."""
    fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.3), sharey=True, constrained_layout=True)
    for i, (ax, (title, field, levels, path)) in enumerate(zip(axes, panels)):
        step, iterate = (SQP_STEP, SQP_ITERATE) if i == sqp_panel else (STEP, ITERATE)
        ax.contour(U0, U1, field, levels=levels, **RINGS)
        for xa, xb in ((XLIM[0], umin), (umax, XLIM[1])):
            ax.axvspan(xa, xb, color="0.88", zorder=0)
        ax.axvline(umin, color="k", lw=0.8)
        ax.axvline(umax, color="k", lw=0.8)
        if path is not None:
            for v_a, v_b in zip(path[:-1], path[1:]):
                if np.linalg.norm(v_b - v_a) > 1e-2:  # the last iterates sit on the solution
                    ax.annotate("", xy=v_b, xytext=v_a,
                                arrowprops=dict(arrowstyle="-|>", mutation_scale=6, shrinkA=0, shrinkB=0, **step))
            ax.plot(*path.T, **iterate, zorder=4)
        ax.plot(*v_0, **START, zorder=5)
        ax.plot(*v_exact, **EXACT, zorder=5)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$u_0$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel(r"$u_1$")

    handles = [
        Line2D([], [], **START, label="cold start"),
        Line2D([], [], marker=ITERATE["marker"], ms=ITERATE["ms"], **STEP, label="IPM iterates"),
        Line2D([], [], **EXACT, label="exact solution"),
    ]
    if sqp_panel is not None:
        handles.insert(2, Line2D([], [], marker=SQP_ITERATE["marker"], ms=SQP_ITERATE["ms"], **SQP_STEP,
                                 label="SQP iterates"))
    if v_approx is not None:
        axes[1].plot(*v_approx, **APPROX, zorder=5)
        handles.append(Line2D([], [], **APPROX, label="approximate direction"))
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
                     handletextpad=0.4, columnspacing=1.2, borderpad=0.1)
    if len(handles) > 3:  # more than three entries go on two rows, each centred on its own
        centre_legend_rows(leg, [3, len(handles) - 3])

    pgf = ROOT / f"{name}.pgf"
    fig.savefig(pgf, bbox_inches="tight", pad_inches=0.02)
    pgf.write_text("\\providecommand{\\mathdefault}[1]{#1}%\n" + pgf.read_text())
    fig.savefig(pgf.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)  # preview


def plot_toy_sqp_ipm(M, c, M_reg, c_reg, umin, umax, v_0, v_exact, v_reg, K_ipm, g_ipm, dv_ipm,
                     sub_path, sqp_path, ipm_path):
    # figures
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })
    U0, U1 = np.meshgrid(np.linspace(*XLIM, 400), np.linspace(*YLIM, 400))
    D0, D1 = U0 - v_0[0], U1 - v_0[1]  # local coordinates d = v - v_0
    rings = lambda f_min: f_min + (0.2 * np.arange(1, 14)) ** 2

    original = (quad(M, c, U0, U1), np.linspace(-6, 6, 25))
    regularized = (quad(M_reg, c_reg, U0, U1), rings(quad(M_reg, c_reg, *v_reg)))
    newton = (quad(K_ipm, g_ipm, D0, D1), rings(quad(K_ipm, g_ipm, *dv_ipm)))

    plot_row("toy_bound_sqp_level", [
        ("(a) Original problem", *original, None),
        (f"(b) Subproblem after {dict(PROJECT='projection', MIRROR='mirror')[TracedSqpSolver.REG]}",
         *regularized, sub_path),
        ("(c) Problem with SQP iterates", *original, sqp_path),
    ], U0, U1, umin, umax, v_0, v_exact, v_approx=v_reg, sqp_panel=2)

    plot_row("toy_bound_ipm_level", [
        ("(a) Original problem", *original, None),
        ("(b) With curvature contributions", *newton, ipm_path[:2]),
        ("(c) Problem with IPM iterates", *original, ipm_path),
    ], U0, U1, umin, umax, v_0, v_exact)


def main():
    # problem data
    r, q1, q2 = 1.0, -3.0, 0.5  # stage weights, q1 < 0 makes the QP indefinite
    x0 = 1.0                    # initial state
    umin, umax = -1.0, 1.0      # bound on u0

    # eliminate the states: w = (u0, x1, u1, x2) = T v + t with v = (u0, u1)
    T = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t = np.array([0.0, x0, 0.0, x0])

    # the bound umin <= u0 <= umax as G v <= h
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    h = np.array([umax, -umin])

    M, c = reduced_qp([r, q1, r, q2], T, t)
    reg_eps = 1e-4  # acados default reg_epsilon
    # M_reg, c_reg = reduced_qp([r, max(q1, reg_eps), r, q2], T, t) # PROJECT
    M_reg, c_reg = reduced_qp([r, abs(q1), r, q2], T, t) # MIRROR, switch TracedSqpSolver.REG as well

    v_exact = np.array([umax, best_u1(umax, M, c)])
    v_reg = np.linalg.solve(M_reg, -c_reg)
    v_0 = np.zeros(2)

    ocp_qp = toy_ocp_qp(r, q1, q2, x0, umin, umax, u1_bound=1e5)
    sqp = TracedSqpSolver(ocp_qp, nlp_max_iter=50)
    sqp_status = sqp.solve()
    sqp_path = sqp.get_iterates().as_array("u")[:, :, 0]
    qp_iters = sqp.get_stats("qp_iter").astype(int)
    sub_path = subproblem_iterates(ocp_qp, qp_iters[1])

    qp = BaseQP().populate_from_ocp_qp(toy_ocp_qp(r, q1, q2, x0, umin, umax))
    ipm = TracedQpSolver(qp, reg="NO_REGULARIZE", type="Riccati", mehrotra=True, globalization=False)
    mu0 = ipm.rhs.duality_measure
    ipm.solve()
    ipm_path = np.array(ipm.iterates)

    # the Newton model of the first IPM step
    s_0 = h - G @ v_0
    K_ipm = M + G.T @ np.diag(mu0 / s_0**2) @ G
    dv_ipm = ipm_path[1] - v_0
    g_ipm = -K_ipm @ dv_ipm

    print("first subproblem iterates =\n", sub_path)
    print("SQP iterates =\n", sqp_path, " status =", sqp_status, " QP iterations =", qp_iters)
    print("IPM iterates =\n", ipm_path, " status =", ipm.status)
    return dict(M=M, c=c, M_reg=M_reg, c_reg=c_reg, umin=umin, umax=umax,
                v_0=v_0, v_exact=v_exact, v_reg=v_reg,
                K_ipm=K_ipm, g_ipm=g_ipm, dv_ipm=dv_ipm,
                sub_path=sub_path, sqp_path=sqp_path, ipm_path=ipm_path)


if __name__ == "__main__":
    plot_toy_sqp_ipm(**main())
