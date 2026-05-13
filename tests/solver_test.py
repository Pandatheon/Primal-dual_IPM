import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosOcpQpSolver, AcadosOcpQpOptions
from QP_template.QP_base import BaseQP
from solvers.RiccatiQPSolver import QPSolver
from solvers.Regularization_config import Regularizations_config, no_reg, Hessian_reg, condensed_Hessian_reg, stage_reg, reg_list

json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Furuta_pos_M.json'

def main():

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    opts = AcadosOcpQpOptions()
    opts.mu0 = 1e1
    opts.hpipm_mode = 'CUSTOMIZED'
    qp_solver = AcadosOcpQpSolver(qp, opts)
    stats = qp_solver.solve()
    iter = qp_solver.get_stats("iter")
    print(f"HPIPM solved in {iter} iterations with status {stats}")
    x_acados = [qp_solver.get(k, 'x') for k in range(qp.N+1)]
    u_acados = [qp_solver.get(k, 'u') for k in range(qp.N)]
    lam_acados = [qp_solver.get(k, 'lam') for k in range(qp.N)]
    pi_acados = [qp_solver.get(k, 'pi') for k in range(qp.N)]

    qp = BaseQP()
    qp.populate_from_json(json_path)
    print(f"QP loaded from {json_path} with N={qp.N}, nx={qp.nx}, nu={qp.nu}, nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}")

    method = no_reg[0]
    qp_solver = QPSolver(qp, reg=method, type='Riccati')

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    qp_solver.solve(x0=x0, max_iter=100)
    # qp_solver.print_statistics()
    x_prototype = [qp_solver.get(k, 'x') for k in range(qp.N+1)]
    u_prototype = [qp_solver.get(k, 'u') for k in range(qp.N)]
    lam_prototype = [qp_solver.get(k, 'lam') for k in range(qp.N)]
    pi_prototype = [qp_solver.get(k, 'pi') for k in range(1, qp.N+1)]

    ux_print = [np.concatenate((u_prototype[k], x_prototype[k])) for k in range(qp.N)]
    print("ux values at each stage:")
    for k in range(qp.N):
        print(f"Stage {k}: ux = {ux_print[k]}")

    diff_x = [np.linalg.norm(x_acados[k] - x_prototype[k]) for k in range(qp.N+1)]
    diff_u = [np.linalg.norm(u_acados[k] - u_prototype[k]) for k in range(qp.N)]
    diff_lam = [np.linalg.norm(lam_acados[k] - lam_prototype[k]) for k in range(1, qp.N)]
    diff_pi = [np.linalg.norm(pi_acados[k] - pi_prototype[k]) for k in range(qp.N)]

    print("Difference in x between HPIPM and prototype:", np.linalg.norm(diff_x))
    print("Difference in u between HPIPM and prototype:", np.linalg.norm(diff_u))
    print("Difference in lam between HPIPM and prototype:", np.linalg.norm(diff_lam))
    print("Difference in pi between HPIPM and prototype:", np.linalg.norm(diff_pi))

if __name__ == "__main__":
    main()