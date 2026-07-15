import csv
import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from solvers.RiccatiQpSolver import QpSolver
from solvers.Regularization_config import reg_list, Block_Hessian_reg, condensed_Hessian_reg, Block_reduced_Hessian_reg, no_reg, Refer_list

from solvers.SqpQpSolver import SqpQpSolver
from acados_template import AcadosCasadiOcpQpSolver, AcadosOcpQp

CSV_COLUMNS = [
    "QP_name",
    "Regularization",
    "Iteration",
    "Cost",
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

    methods = ['Riccati'] + Refer_list

    for qp_path in qp_json_paths:
        print(f"QP loaded from {qp_path}")
        for method in methods:
            print("-" * 50)
            print(f"Method: {method}")
            if method == 'SQP_PROJECT':
                qp = AcadosOcpQp.from_json(json_file_path=str(qp_path))
                qp_solver = SqpQpSolver(qp, reg='NO_REGULARIZE', qp_max_iter=30)
            elif method == 'IPOPT':
                qp = AcadosOcpQp.from_json(json_file_path=str(qp_path))
                opts = {'ipopt': {'print_level': 0, 'max_iter': 300, 'sb': 'yes'}}
                qp_solver = AcadosCasadiOcpQpSolver(qp, solver='ipopt', solver_opts=opts)
            else:
                qp = BaseQP()
                qp.populate_from_json(str(qp_path))
                qp_solver = QpSolver(qp, reg='NO_REG', type=method, mehrotra=False, globalization=True)

            cost = None
            iter = None
            try:
                qp_solver.solve()
                cost = qp_solver.get_cost()
                if method == 'SQP_PROJECT':
                    iter = sum(qp_solver.get_stats('qp_iter'))
                else:
                    iter = qp_solver.get_stats("iter")
            except Exception as exc:
                print(f"Method {method} failed on {qp_path.name}: {exc}")
            if cost is not None:
                print(f"Optimal cost: {cost}")
            write_csv_row(
                output_path,
                {
                    "QP_name": qp_path.stem,
                    "Regularization": method,
                    "Iteration": iter,
                    "Cost": cost,
                    "Status": qp_solver.status,
                },
            )

def main() -> None:
    qp_root = Path(__file__).resolve().parent / "tests" / "QP_json" / "Pos_QP_json"
    qp_json_paths = iter_qp_json_paths(qp_root)
    output_path = Path(__file__).resolve().parent / "qp_results_pos.csv"
    run_benchmark(qp_json_paths, reg_list, output_path)


if __name__ == "__main__":
    main()