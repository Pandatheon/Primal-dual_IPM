import matplotlib.pyplot as plt
from acados_template import latexify_plot, AcadosOcpSolver
import numpy as np

def calculate_KKT_norms(ocp_solver: AcadosOcpSolver):
    res_all = ocp_solver.get_stats("res_all")
    res_norms = np.linalg.norm(res_all, axis=1)
    # TODO: dont return 0 if QP failuer and last row is all zeros.

    return res_norms

def check_positive_definite(ocp_solver : AcadosOcpSolver):

    tol = 5e-5
    eigs_full = ocp_solver.qp_diagnostics('FULL_HESSIAN')
    eigs_proj = ocp_solver.qp_diagnostics('PROJECTED_HESSIAN')

    if eigs_full['min_eigv_global'] < 0:
        print("Full Hessian is indefinite, min eigenvalue: ", eigs_full['min_eigv_global'])
    elif abs(eigs_full['min_eigv_global']) < tol:
        print("Full Hessian is positive semi-definite, min eigenvalue: ", eigs_full['min_eigv_global'])
    else:
        print("Full Hessian is positive definite, min eigenvalue: ", eigs_full['min_eigv_global'])
    
    if eigs_proj['min_eigv_global'] < 0:
        print("Projected Hessian is indefinite, min eigenvalue: ", eigs_proj['min_eigv_global'])
    elif abs(eigs_proj['min_eigv_global']) < tol:
        print("Projected Hessian is positive semi-definite, min eigenvalue: ", eigs_proj['min_eigv_global'])
    else:
        print("Projected Hessian is positive definite, min eigenvalue: ", eigs_proj['min_eigv_global'])


def plot_diff(diff, labels=None, min_lim=1e-15, max_lim=1, min_iter=0, max_iter=30):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for d in diff:
        ax.semilogy(d)
    if labels is not None:
        ax.legend(labels)
    plt.xlabel("Iteration")
    plt.ylabel("KKT norm")
    # plt.title("Difference between Iterates and Optimal Solution")
    plt.xlim((min_iter, max_iter))
    plt.ylim((min_lim, max_lim))
    plt.grid()
    plt.show()


def plot_pendulum(shooting_nodes, u_max, U, X_true, X_est=None, Y_measured=None, latexify=True, plt_show=True, X_true_label=None,
    time_label='$t$', x_labels=['$x$', r'$\theta$', '$v$', r'$\dot{\theta}$'],
    title = None
                  ):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)

    plt.subplot(nx+1, 1, 1)
    line, = plt.step(t, np.append([U[0]], U))
    if X_true_label is not None:
        line.set_label(X_true_label)
    else:
        line.set_color('r')
    if title is not None:
        plt.title(title)
    plt.ylabel('$u$')
    plt.xlabel(time_label)
    plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max, 1.2*u_max])
    plt.grid()


    for i in range(nx):
        plt.subplot(nx+1, 1, i+2)
        line, = plt.plot(t, X_true[:, i], label='true')
        if X_true_label is not None:
            line.set_label(X_true_label)

        if WITH_ESTIMATION:
            plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
            plt.plot(t, Y_measured[:, i], 'x', label='measured')

        plt.ylabel(x_labels[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show()
    return