import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from solvers.FiaccoQpSolver import QpSolver
from solvers.Regularization_config import Regularizations_config, no_reg, Block_Hessian_reg, Block_reduced_Hessian_reg, reg_list

reg_config = Regularizations_config()
json_path = '/home/jingtao/Mehrotra_IPM/tests/QP_json/temp/overhead_crane_FU_PD_RE_PD_status_2_iter_5.json'

def iter_qp_json_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"QP folder not found: {root}")
    return sorted(root.glob("*.json"))


def main():

    qp = BaseQP()
    qp.populate_from_json(json_path)
    qp_name = os.path.basename(json_path)
    print(f"QP loaded from {qp_name} with N={qp.N}, nx={qp.nx[1]}, nu={qp.nu}, nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}")

    method = 'NO_REGULARIZE'
    qp_solver = QpSolver(qp, reg=method, type='Riccati', globalization=False)
    qp_solver.solve(max_iter=30)

    cost = qp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    qp_solver.print_statistics()

def run_benchmark(qp_path: list[Path]) -> None:

    qp = BaseQP()
    qp.populate_from_json(str(qp_path))
    print("-" * 50)
    print(
        f"QP loaded from {qp_path}"
    )

    print(f"Prototye Riccati:")
    qp_solver = QpSolver(qp, type='Riccati', reg='NO_REGULARIZE', globalization=False)
    qp_solver.solve(max_iter=30)
    iter = qp_solver.get_stats("iter")
    return iter

def run_benchmark_reg(qp_path: list[Path]) -> None:

    qp = BaseQP()
    qp.populate_from_json(str(qp_path))
    print("-" * 50)
    print(
        f"QP loaded from {qp_path}"
    )

    print(f"Prototye Riccati:")
    # reg ='PM_STAGE_MIRROR'
    reg = 'INERTIA_LM'
    qp_solver = QpSolver(qp, type='Riccati', reg=reg, globalization=True)
    qp_solver.solve(max_iter=100)


def main_loop() -> None:
    qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json')
    qp_json_paths = iter_qp_json_paths(qp_root)
    iter_total = 0
    for qp_path in qp_json_paths:
        iter_total += run_benchmark(qp_path)
    print(f"Total iterations for all QPs: {iter_total}")

def main_loop_with_reg() -> None:
    qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/Inde_QP_json')
    qp_json_paths = iter_qp_json_paths(qp_root)
    for qp_path in qp_json_paths:
        run_benchmark_reg(qp_path)

if __name__ == "__main__":
    # main()
    main_loop()
    # main_loop_with_reg()