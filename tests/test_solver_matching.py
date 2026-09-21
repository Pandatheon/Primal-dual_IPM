import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from solvers.RiccatiQpSolver import QpSolver
from solvers.Regularization_config import no_reg

from acados_template import AcadosOcpQp, AcadosOcpQpSolver, AcadosOcpQpOptions

def iter_qp_json_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"QP folder not found: {root}")
    return sorted(root.glob("*.json"))


def run_benchmark(qp_path: list[Path]) -> None:

    opts = AcadosOcpQpOptions()
    opts.mu0 = 1e1
    opts.hpipm_mode = 'CUSTOMIZED'
    opts.ric_alg = 0
    opts.tol_stat = 1e-6
    opts.tol_eq = 1e-8
    opts.tol_ineq = 1e-8
    opts.tol_comp = 1e-6
    opts.iter_max = 30

    qp = BaseQP()
    qp.populate_from_json(str(qp_path))
    print("-" * 50)
    print(
        f"QP loaded from {qp_path}"
    )

    print(f"Prototye Riccati:")
    qp_solver = QpSolver(qp, type='Riccati', reg='NO_REGULARIZE', mehrotra=True, globalization=False)
    qp_solver.solve(max_iter=30)
    iter_riccati = qp_solver.stat_dict['iter_count']

    qp_acados = AcadosOcpQp.from_json(json_file_path=str(qp_path))
    qp_solver_acados = AcadosOcpQpSolver(qp_acados, opts)
    stats = qp_solver_acados.solve()
    iter_hpipm = qp_solver_acados.get_stats("iter")

    print(f"HPIPM solved in {iter_hpipm} iterations with status {stats}")
    diff_iter_riccati = abs(iter_riccati - iter_hpipm)
    assert diff_iter_riccati <= 2, f"Iteration count difference {diff_iter_riccati} exceeds tolerance for Riccati on QP {qp_path.name}"
    print(f"QP {qp_path.name} passed, diff: {diff_iter_riccati} for Riccati")

def main() -> None:
    # qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json')
    qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/temp')
    qp_json_paths = iter_qp_json_paths(qp_root)
    for qp_path in qp_json_paths:
        run_benchmark(qp_path)

if __name__ == "__main__":
    main()