import csv
import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from solvers.RiccatiQPSolver import QPSolver
from solvers.Regularization_config import reg_list, Hessian_reg, condensed_Hessian_reg, stage_reg


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

    for qp_path in qp_json_paths:
        qp = BaseQP()
        qp.populate_from_json(str(qp_path))
        print(
            f"QP loaded from {qp_path} with N={qp.N}, nx={qp.nx}, nu={qp.nu}, "
            f"nh_0={qp.nh_0}, nh={qp.nh}, nh_e={qp.nh_e}, "
            f"nb_0={qp.nb_0}, nb={qp.nb}, nb_e={qp.nb_e}"
        )

        # Keep x0 length consistent with model dimensions.
        x0 = np.array([0.0, np.pi, 0.0, 0.0])

        for method in methods:
            print("-" * 50)
            print(f"Method: {method}")
            if method in condensed_Hessian_reg:
                qp_solver = QPSolver(qp, reg=method, type='Condensed')
            else:
                qp_solver = QPSolver(qp, reg=method, type='Riccati')
            cost = None

            try:
                qp_solver.solve(x0=x0, max_iter=300)
                cost = qp_solver.get_cost()
            except Exception as exc:
                print(f"Method {method} failed on {qp_path.name}: {exc}")
            if cost is not None:
                print(f"Optimal cost: {cost}")
            write_csv_row(
                output_path,
                {
                    "QP_name": qp_path.stem,
                    "Regularization": method,
                    "Iteration": qp_solver.statistics_dict.get("iter_count", 0),
                    "Cost": cost,
                    "Status": qp_solver.status,
                },
            )

def main() -> None:
    qp_root = Path(__file__).resolve().parent / "cases" / "QP_json"
    qp_json_paths = iter_qp_json_paths(qp_root)
    output_path = Path(__file__).resolve().parent / "qp_results.csv"
    run_benchmark(qp_json_paths, reg_list, output_path)


if __name__ == "__main__":
    main()