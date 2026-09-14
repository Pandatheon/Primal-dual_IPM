import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import csv
import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from solvers.RiccatiQpSolver import QpSolver
from solvers.Regularization_config import reg_list, Block_Hessian_reg, Block_reduced_Hessian_reg, no_reg, Refer_list

from solvers.SqpQpSolver import SqpQpSolver
from acados_template import AcadosCasadiOcpQpSolver, AcadosOcpQp

from utils.plot_stuff import plot_charts_pos, plot_decomp_flop_worstcase

CSV_COLUMNS = [
    "QP_name",
    "Regularization",
    "Iteration",
    "Cost",
    "Decomp_Flop",
    "Status",
]


def iter_qp_json_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"QP folder not found: {root}")
    return sorted(root.glob("*.json"))


def write_csv_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark(qp_json_paths: list[Path], methods: list[str], output_path: Path) -> None:
    if not qp_json_paths:
        print("No QP json files found; nothing to benchmark.")
        return

    if output_path.exists():
        output_path.unlink()
        print(f"Deleted existing {output_path} to start fresh.")

    methods = [
            'NO_REGULARIZE',
            # 'PROJECT_H',
            # 'M_STAGE_SE', 
            # 'M_STAGE_MIRROR',
            'PM_MIRROR',
            # 'RP_STAGE_MIRROR',
            # 'PM_STAGE_PROJECT',
            # 'PM_STAGE_MINPROJECT',
            # 'PM_STAGE_TUNEDMIRROR',
            # 'PM_STAGE_SE',
            'PM_GMW',
            'INERTIA_LM',
            # 'INERTIA_LM_RE',
            # 'PRECOMPUTE',
            # 'IPOPT',
            ]

    for qp_path in qp_json_paths:
        print(f"QP loaded from {qp_path}")
        for method in methods:
            print("-" * 50)
            print(f"Method: {method}")
            if method == 'SQP_PROJECT':
                qp = AcadosOcpQp.from_json(json_file_path=str(qp_path))
                qp_solver = SqpQpSolver(qp, reg='PROJECT')
            elif method == 'IPOPT':
                qp = AcadosOcpQp.from_json(json_file_path=str(qp_path))
                opts = {'ipopt': {'print_level': 0, 'max_iter': 100, 'sb': 'yes'}}
                qp_solver = AcadosCasadiOcpQpSolver(qp, solver='ipopt', solver_opts=opts)
            else:
                qp = BaseQP()
                qp.populate_from_json(str(qp_path))
                qp_solver = QpSolver(qp, reg=method, type='Riccati', mehrotra=False, globalization=True)

            cost = None
            iter = None
            decomp_flop = None
            try:
                if method != 'IPOPT':
                    qp_solver.solve(max_iter=100)
                else:
                    qp_solver.solve()
                cost = qp_solver.get_cost()
                if method == 'SQP_PROJECT':
                    iter = sum(qp_solver.get_stats('qp_iter'))
                else:
                    iter = qp_solver.get_stats("iter")
                if method not in Refer_list:
                    decomp_flop = qp_solver.stat_dict.get('decomp_flop', 'N/A')
            except Exception as exc:
                print(f"Method {method} failed on {qp_path.name}: {exc}")
            if cost is not None:
                print(f"Optimal cost: {cost}")
            if decomp_flop is None:
                decomp_flop = 'N/A'
            write_csv_row(
                output_path,
                {
                    "QP_name": qp_path.stem,
                    "Regularization": method,
                    "Iteration": iter,
                    "Cost": cost,
                    'Decomp_Flop': decomp_flop,
                    "Status": qp_solver.status,
                },
            )

def main() -> None:
    # qp_root = Path(__file__).resolve().parent.parent / "cases" / "Tracking_inde_QP_json"
    # qp_root = Path(__file__).resolve().parent.parent / "cases" / "artificial_qps"
    qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/temp')
    qp_json_paths = iter_qp_json_paths(qp_root)
    output_path = Path(__file__).resolve().parent / "temp_results" / "qp_results_riccati.csv"
    run_benchmark(qp_json_paths, reg_list, output_path)
    plot_charts_pos(str(output_path))
    plot_decomp_flop_worstcase(str(output_path))


if __name__ == "__main__":
    main()