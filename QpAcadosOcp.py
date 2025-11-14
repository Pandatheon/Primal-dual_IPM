from acados_template import AcadosOcp
from RiccatiQPSolver import QP
import numpy as np
import casadi as ca


class QpAcadosOcp:
    def __init__ (self, QP: QP, x0):
        """
        Creates an equivalent AcadosOcp formulation from QP.
        """
        ocp = AcadosOcp()
        ocp.model.name = 'qp_acados_ocp'

        # dimensions
        N = QP.N
        nx = QP.nx
        nu = QP.nu

        # set model
        ocp.model.x = ca.SX.sym('x', nx)
        ocp.model.u = ca.SX.sym('u', nu)

        # set model dynamics
        A = QP.stages[0].A
        B = QP.stages[0].B
        b = QP.stages[0].b

        x_next = ca.mtimes(A, ocp.model.x) + ca.mtimes(B, ocp.model.u) + b
        ocp.model.disc_dyn_expr = x_next

        # set cost
        Q = QP.stages[0].Q
        R = QP.stages[0].R
        S = QP.stages[0].S
        q = QP.stages[0].q
        r = QP.stages[0].r

        Q_e = QP.terminal.Q_e
        q_e = QP.terminal.q_e

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.W = ca.diagcat(Q, R).full()
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
        ocp.cost.yref = np.zeros((nx + nu,))
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.W_e = Q_e
        ocp.cost.yref_e = np.zeros((nx,))

        # constraints
        ocp.constraints.x0 = x0

        # set solver options
        ocp.solver_options.tf = N
        ocp.solver_options.N_horizon = N
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP'

        self.ocp = ocp