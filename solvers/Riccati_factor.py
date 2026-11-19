from QP_template.iterate_base import iterate_sol, Feedback
from dataclasses import dataclass
import numpy as np
import copy


@dataclass
class RiccatiFactor:
    P: list = None       # P_k, k = 0..N (Feedback.P, used by the forward sweep)
    P_used: list = None  # P_{k+1} as factorized at stage k (regularized if the stage was), k = 0..N-1
    L: list = None       # Cholesky factor of the (regularized) reduced Hessian of stage k
    y1: list = None      # L_k^{-1} (S_k + B_k' P_used_k A_k)
    K: list = None       # feedback gain -L_k^{-T} y1_k
    CtTinv: list = None  # [C_k D_k]' T_k^{-1}, maps the rhs into the modified QP (modify_rhs), k = 0..N
    Lam: list = None     # diag(lam_k), k = 0..N

def modify_QP(qp, prev_sol:iterate_sol, LM_delta:float=0.0):
    modified_QP = copy.deepcopy(qp)
    CtTinv, Lam = [None]*(qp.N+1), [None]*(qp.N+1)

    for k in range(qp.N+1):
        nx = qp.nx[k]
        C_full = np.hstack((qp.C[k], qp.D[k])) if k < qp.N else qp.C[k]
        T_mat_inv = np.linalg.inv(np.diag(prev_sol.s[k]))
        Lam[k] = np.diag(prev_sol.lam[k])
        CtTinv[k] = C_full.T @ T_mat_inv
        lhs_modification = CtTinv[k] @ Lam[k] @ C_full

        if k < qp.N:
            modified_QP.Q[k] = qp.Q[k] + lhs_modification[:nx, :nx] + LM_delta * np.eye(nx)
            modified_QP.R[k] = qp.R[k] + lhs_modification[nx:, nx:] + LM_delta * np.eye(qp.nu)
            modified_QP.S[k] = qp.S[k] + lhs_modification[nx:, :nx]
        else:
            # primal regularization delta_w*I must also be added to the terminal Hessian
            modified_QP.Q_e = qp.Q_e + lhs_modification[:nx, :nx] + LM_delta * np.eye(nx)

    return modified_QP, CtTinv, Lam

def modify_rhs(qp, rhs, fact:RiccatiFactor):
    rhs_kkt = copy.deepcopy(rhs)
    for k in range(qp.N+1):
        nx = qp.nx[k]
        rhs_modification = fact.CtTinv[k] @ (fact.Lam[k] @ rhs_kkt.rd[k] - rhs_kkt.rm[k])
        rhs_kkt.rg[k][:nx] = rhs_kkt.rg[k][:nx] + rhs_modification[:nx]
        if k < qp.N:
            rhs_kkt.rg[k][nx:] = rhs_kkt.rg[k][nx:] + rhs_modification[nx:]
    return rhs_kkt

def backward_pass(qp, fact:RiccatiFactor, rhs_kkt)-> Feedback:
    horizon = qp.N
    feedback = Feedback(
        P = fact.P,
        p = [None]*(horizon+1),
        K = fact.K,
        k = [None]*(horizon),
    )
    feedback.p[horizon] = rhs_kkt.rg[-1][:qp.nx[horizon]]

    for k in reversed(range(horizon)):
        nx = qp.nx[k]
        A_k = qp.A[k]
        B_k = qp.B[k]
        q_k = rhs_kkt.rg[k][:nx]
        r_k = rhs_kkt.rg[k][nx:]
        b_k = rhs_kkt.rb[k]
        P_used = fact.P_used[k]
        p_prev = feedback.p[k+1]

        g2 = r_k + B_k.T@(P_used@b_k + p_prev)
        y2 = np.linalg.solve(fact.L[k], g2)
        feedback.k[k] = -np.linalg.solve(fact.L[k].T, y2)
        feedback.p[k] = q_k + A_k.T@(P_used@b_k + p_prev) - fact.y1[k].T@y2

    return feedback
