import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from acados_template import AcadosOcpQp, AcadosCasadiOcpQpSolver


json_path = '/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json/acrobot_FU_PD_RE_PD_status_2_iter_25.json'

def main():

    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    opts = {'ipopt': {'print_level': 5,
                      'file_print_level': 8,
                      'output_file': 'ipopt_detailed_log.txt',
                      'mu_init':1e1,
                      'mu_min':1e-16,
                      'max_iter': 30,
                      'bound_mult_init_val': 1.0,
                      'nlp_scaling_method': 'none',
                      'least_square_init_duals': 'no',
                      'sb': 'yes',
                      'bound_mult_init_method': 'mu-based',
                      'accept_every_trial_step': "yes"
                      }}
    qp_solver = AcadosCasadiOcpQpSolver(qp, solver='ipopt', solver_opts=opts)
    stats = qp_solver.solve()
    iter = qp_solver.get_stats("iter")
    print(f"IPOPT iterations: {iter}")
    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")

    # print("Optimal state trajectory:")
    # for k in range(qp.N+1):
    #     x = qp_solver.get(k, 'x')
    #     print(f"x[{k}] = {x}")
    # for k in range(qp.N):
    #     lam = qp_solver.get(k, 'lam')
    #     print(f"Stage {k}: lam = {lam}")
    # for k in range(qp.N):
    #     pi = qp_solver.get(k, 'pi')
    #     print(f"Stage {k}: pi = {pi}")

if __name__ == "__main__":
    main()