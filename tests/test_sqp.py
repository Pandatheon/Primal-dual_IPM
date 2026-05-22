import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosParam, AcadosParamManager, AcadosOcpQp
import casadi as ca
import numpy as np
from solvers.SqpQpSolver import SqpQpSolver
import json

json_path = '/home/jingtao/Primal-dual_IPM/tests/QP_json/Pos_QP_json/Pendulum_FU_PD_RE_PD_status_2_iter_0.json'

def main():
    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    N = qp.N
    Sqp_solver = SqpQpSolver(qp)
    status = Sqp_solver.solve()
    if status != 0:
        print(f"ocp_solver returned status {status}.")
    Sqp_solver.print_statistics()
    cost = Sqp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    x = [Sqp_solver.get(k, "x") for k in range(N+1)]
    u = [Sqp_solver.get(k, "u") for k in range(N)]
    lam = [Sqp_solver.get(k, "lam") for k in range(N)]
    for k in range(N):
        print(f"lam[{k}] = {lam[k]}")

if __name__ == "__main__":
    main()