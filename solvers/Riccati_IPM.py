from typing import Tuple
from dataclasses import dataclass
from QP_template.OCP_QP import OCP_QP_varying, QP_stage, QP_Terminal, iterate
from solvers.Riccati_recursion import Feedback, riccati_recursion_with_stage_regularization
from Regularizations import Regularizations_config
from solvers.Regularization import regularize_QP
from solvers.regularization_recorder import Recorder
import numpy as np
import copy


def update_rhs(qp, centering_param:float, rhs_dict:dict, statistics_dict:dict, ineq_num:int,
               prev_sol:iterate, delta_aff_dict:dict = None, record:bool=False, step_type:str='predictor')-> np.ndarray:
    nx = qp.nx
    nu = qp.nu
    N = qp.N
    stage0 = qp.stages[0]

    res_stat = np.zeros((N+1, nx+nu))
    res_eq = np.zeros((N, nx))
    res_ineq0 = np.zeros((1, stage0.C.shape[0]))
    res_ineq = np.zeros((N-1, qp.stages[1].C.shape[0]))
    res_comp0 = np.zeros((1, stage0.C.shape[0]))
    res_comp = np.zeros((N-1, qp.stages[1].C.shape[0]))
    duality_measure = 0.0

    for i in range(len(qp.stages)):
        # stationarity residual
        if i == 0:
            res_stat[i][nx:] = qp.stages[0].R @ prev_sol.u[0] + qp.stages[0].r + qp.stages[0].B.T @ prev_sol.pi[0] + qp.stages[0].D.T @ prev_sol.lam0
        else:
            res_stat[i][:nx] = qp.stages[i].Q @ prev_sol.x[i] + qp.stages[i].S.T @ prev_sol.u[i] + qp.stages[i].q +  qp.stages[i].A.T @ prev_sol.pi[i] + qp.stages[i].C.T @ prev_sol.lam[i-1] - prev_sol.pi[i-1]
            res_stat[i][nx:] = qp.stages[i].R @ prev_sol.u[i] + qp.stages[i].S @ prev_sol.x[i] + qp.stages[i].r + qp.stages[i].B.T @ prev_sol.pi[i] + qp.stages[i].D.T @ prev_sol.lam[i-1]

        #equality residual
        res_eq[i] = qp.stages[i].A @ prev_sol.x[i] + qp.stages[i].B @ prev_sol.u[i] - prev_sol.x[i+1] - qp.stages[i].b

        #inequality residual
        if i == 0:
            res_ineq0 =  qp.stages[0].C @ prev_sol.x[0] + qp.stages[0].D @ prev_sol.u[0] + prev_sol.s0 - qp.stages[0].d
        else:
            res_ineq[i-1] = (qp.stages[i].C @ prev_sol.x[i] + qp.stages[i].D @ prev_sol.u[i] + prev_sol.s[i-1]) - qp.stages[i].d

        #complementarity residual
        if i == 0:
            if step_type == 'predictor':
                res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten()
                duality_measure += (prev_sol.lam0 @ prev_sol.s0)
            elif step_type == 'centering':
                res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten() - centering_param * rhs_dict['duality_measure'] * np.ones(len(prev_sol.lam0))
            elif step_type == 'predictor-corrector':
                res_comp0 = (prev_sol.lam0 * prev_sol.s0).flatten() + (delta_aff_dict['delta_lam0'] * delta_aff_dict['delta_s0']).flatten() - centering_param * rhs_dict['duality_measure'] * np.ones(len(prev_sol.lam0))
        else:
            if step_type == 'predictor':
                res_comp[i-1] = (np.diag(prev_sol.lam[i-1]) @ prev_sol.s[i-1]).flatten()
                duality_measure += (prev_sol.lam[i-1] @ prev_sol.s[i-1])
            elif step_type == 'centering':
                res_comp[i-1] = (prev_sol.lam[i-1] * prev_sol.s[i-1]).flatten() - centering_param * rhs_dict['duality_measure'] * np.ones(len(prev_sol.lam[i-1]))
            elif step_type == 'predictor-corrector':
                res_comp[i-1] = (prev_sol.lam[i-1] * prev_sol.s[i-1]).flatten() + (delta_aff_dict['delta_lam'][i-1] * delta_aff_dict['delta_s'][i-1]) - centering_param * rhs_dict['duality_measure'] * np.ones(len(prev_sol.lam[i-1]))

    # terminal residual
    res_stat[N][:nx] = qp.terminal.Q_e @ prev_sol.x[N] + qp.terminal.q_e - prev_sol.pi[N-1]

    rhs_dict['rg'] = res_stat
    rhs_dict['rb'] = res_eq
    rhs_dict['rd0'] = res_ineq0
    rhs_dict['rd'] = res_ineq
    rhs_dict['rm0'] = res_comp0
    rhs_dict['rm'] = res_comp
    if step_type == 'predictor' and not record:
        rhs_dict['duality_measure'] = max(duality_measure / ineq_num, 1e-16)
        statistics_dict['duality_measure_list'].append(rhs_dict['duality_measure'])
    return rhs_dict

def foward_sweep(qp, rhs_dict, 
                 prev_sol:iterate, feedback:Feedback)-> dict:
    delta_x = copy.deepcopy(prev_sol.x)
    delta_u = copy.deepcopy(prev_sol.u)
    delta_pi = copy.deepcopy(prev_sol.pi)
    delta_lam0 = copy.deepcopy(prev_sol.lam0)
    delta_lam = copy.deepcopy(prev_sol.lam)
    delta_s0 = copy.deepcopy(prev_sol.s0)
    delta_s = copy.deepcopy(prev_sol.s)

    # recover delta_x and delta_u, delta_pi first with modified KKT system
    # then recover delta_s and delta_lam with original KKT system
    delta_x[0] = 0
    for i in range(len(qp.stages)):
        # recover x and u
        rhs_b = rhs_dict['rb'][i]
        delta_u[i] = feedback.compute_control(i, delta_x[i])
        delta_x[i+1] = qp.stages[i].A@delta_x[i] + qp.stages[i].B@delta_u[i] + rhs_b

        # recover s
        if i == 0:
            rhs_d =  rhs_dict['rd0']
            delta_s0 = -(qp.stages[0].C @delta_x[0] + qp.stages[0].D @delta_u[0]) - rhs_d
        else:
            rhs_d = rhs_dict['rd'][i-1]
            delta_s[i-1] = -(qp.stages[i].C @delta_x[i] + qp.stages[i].D @delta_u[i]) - rhs_d
        # recover lam
        if i == 0:
            rhs_m = rhs_dict['rm0']
            rhs_m_modified = -rhs_m - (prev_sol.lam0 * delta_s0)
            delta_lam0 = (rhs_m_modified/prev_sol.s0).flatten()
        else:
            rhs_m = rhs_dict['rm'][i-1]
            rhs_m_modified = -rhs_m - (prev_sol.lam[i-1] * delta_s[i-1])
            delta_lam[i-1] = (rhs_m_modified/prev_sol.s[i-1]).flatten()
        # recover pi
        delta_pi[i] =  feedback.P[i+1] @ delta_x[i+1] + feedback.p[i+1]

    delta_dict = {
        'delta_x': delta_x,
        'delta_u': delta_u,
        'delta_pi': delta_pi,
        'delta_lam0': delta_lam0,
        'delta_lam': delta_lam,
        'delta_s0': delta_s0,
        'delta_s': delta_s,
    }
    return delta_dict

def backward_sweep(qp, rhs_dict, recorder, reg_config,
                   prev_sol:iterate, reg:str ='NO_REGULARIZE')-> Feedback:
    horizon = len(qp.stages)
    nu, nx = qp.stages[0].S.shape

    if reg == 'PROJECT' or reg == 'MIRROR':
        regularized_QP = regularize_QP(qp, recorder=recorder,
                                        reg_config= reg_config, method=reg)
    else:
        regularized_QP = copy.deepcopy(qp)
    modified_QP_stages = modify_QP(regularized_QP, rhs_dict, prev_sol)

    recorder.reset_stage_arrays()

    feedback = Feedback(
        P = np.zeros((horizon+1, nx, nx), dtype=float),
        p = np.zeros((horizon+1, nx), dtype=float),
        K = np.zeros((horizon, nu, nx), dtype=float),
        k = np.zeros((horizon, nu), dtype=float)
    )
    feedback.P[-1] = modified_QP_stages[-1].Q_e
    feedback.p[-1] = rhs_dict['rg_'][-1][:nx]
    for i in reversed(range(horizon)):
        feedback.P[i], feedback.p[i], feedback.K[i], feedback.k[i] = riccati_recursion_with_stage_regularization(
        qp_stage=modified_QP_stages[i],
        qp_terminal=(feedback.P[i+1], feedback.p[i+1]),
        nx = nx,
        rg=rhs_dict['rg_'][i], rb=rhs_dict['rb'][i],
        reg_config=reg_config, recorder=recorder,
        reg=reg, index=i
        )
    return feedback

def modify_QP(qp:OCP_QP_varying, rhs_dict, prev_sol:iterate):
    modified_QP_stages = []

    for i in range(len(qp.stages)):
        if i == 0:
            C_full = np.concatenate((qp.stages[0].C, qp.stages[0].D), axis=1)
            T_mat = np.diag(prev_sol.s0)
            Lam_mat = np.diag(prev_sol.lam0)
            rd = rhs_dict['rd0']
            rm = rhs_dict['rm0']
        else:
            C_full = np.concatenate((qp.stages[i].C, qp.stages[i].D), axis=1)
            T_mat = np.diag(prev_sol.s[i-1])
            Lam_mat = np.diag(prev_sol.lam[i-1])
            rd = rhs_dict['rd'][i-1]
            rm = rhs_dict['rm'][i-1]

        T_mat_inv = np.linalg.inv(T_mat)
        lhs_modification = C_full.T @ T_mat_inv @ Lam_mat @ C_full
        rhs_modification = C_full.T @ T_mat_inv @ (Lam_mat @ rd - rm)

        Q_tilt = lhs_modification[:qp.nx, :qp.nx]
        R_tilt = lhs_modification[qp.nx:, qp.nx:]
        S_tilt = lhs_modification[qp.nx:, :qp.nx]
        modified_QP_stages.append(QP_stage(
            Q = qp.stages[i].Q + Q_tilt,
                # in order to make R_stage and project the same when only R is negative definite
                # R_tilt has to be removed before riccati recursion to avoid possible successful cholesky decomposition
                # which is due to the curvature provided by constraints (which often happens!!!)
                # and add R_tilt back during recursion regularization after R is modified
            R = qp.stages[i].R + R_tilt,
            S = qp.stages[i].S + S_tilt,
            q = qp.stages[i].q, r = qp.stages[i].r,
            A = qp.stages[i].A, B = qp.stages[i].B, b = qp.stages[i].b,
            C = qp.stages[i].C, D = qp.stages[i].D, d = qp.stages[i].d,
        ))
        rhs_dict['rg_'][i][:qp.nx] = rhs_dict['rg'][i][:qp.nx] + rhs_modification[:qp.nx]
        rhs_dict['rg_'][i][qp.nx:] = rhs_dict['rg'][i][qp.nx:] + rhs_modification[qp.nx:]
    modified_QP_stages.append(qp.terminal)
    rhs_dict['rg_'][-1] = rhs_dict['rg'][-1]
    return modified_QP_stages

def fraction_to_boundary(prev_sol: iterate, delta_dict, statistics_dict, boundary:bool=True) -> iterate:
    s_old = np.hstack((prev_sol.s0, prev_sol.s.flatten()))
    lam_old = np.hstack((prev_sol.lam0, prev_sol.lam.flatten()))
    delta_s = np.hstack((delta_dict['delta_s0'], delta_dict['delta_s'].flatten()))
    delta_lam =  np.hstack((delta_dict['delta_lam0'], delta_dict['delta_lam'].flatten()))

    mask_lam = delta_lam < 0
    mask_s = delta_s < 0
    frac_pri = s_old[mask_s] / (-delta_s[mask_s])
    frac_dual = lam_old[mask_lam] / (-delta_lam[mask_lam])
    # take the minimum positive step size
    try:
        alpha_pri_max = np.min(frac_pri)
    except ValueError:
        alpha_pri_max = 1e-12
    try:
        alpha_dual_max = np.min(frac_dual)
    except ValueError:
        alpha_dual_max = 1e-12

    if boundary:
        # take 0.99 of the maximum step size
        alpha_pri = min(0.99 * alpha_pri_max, 1.0)
        alpha_dual = min(0.99 * alpha_dual_max, 1.0)
        alpha = min(alpha_pri, alpha_dual)

        statistics_dict['alpha_pri_list'].append(alpha_pri)
        statistics_dict['alpha_dual_list'].append(alpha_dual)
        statistics_dict['alpha_list'].append(alpha)
        return alpha_pri, alpha_dual, alpha
    else:
        return alpha_pri_max, alpha_dual_max, min(alpha_pri_max, alpha_dual_max)