import csv
import numpy as np
from pathlib import Path

from QP_template.QP_base import BaseQP
from utils.plot_stuff import plot_stage_indefiniteness_source, plot_indefiniteness_coverage


CSV_COLUMNS = [
    "QP_name",
    "Horizon",
    "Inde Stage Rate",
    "Min Negative Eigenvalue",
]

CSV_STAGE_COLUMNS = [
    "QP_stage_name",
    "R_min_eig",
    "P_min_eig",
    "M_min_eig",
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


def write_csv_row_stage(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_STAGE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def QP_diagnostics(qp_json_paths: list[Path], output_path: Path) -> None:
    if not qp_json_paths:
        print("No QP json files found; nothing to benchmark.")
        return

    if output_path.exists():
        output_path.unlink()
        print(f"Deleted existing {output_path} to start fresh.")

    for qp_path in qp_json_paths:
        print(f"QP loaded from {qp_path}")
        qp = BaseQP()
        qp.populate_from_json(str(qp_path))

        P_list, _, M_list = qp.Riccati_recursion()
        N = len(P_list) - 1  # P_list has length N+1
        Inde_stage_count = 0
        min_neg_eig = 0.0
        for i in range(N):
            eigs_M = np.linalg.eigvals(M_list[i])
            if min(eigs_M) < 0:
                Inde_stage_count += 1
                min_neg_eig = min(min_neg_eig, min(eigs_M))

        write_csv_row(
            output_path,
            {
                "QP_name": qp_path.stem,
                "Horizon": N,
                "Inde Stage Rate": Inde_stage_count / N,
                "Min Negative Eigenvalue": min_neg_eig,
            },
        )


def QP_stage_diagnostics(qp_json_paths: list[Path], output_path: Path) -> None:
    if not qp_json_paths:
        print("No QP json files found; nothing to benchmark.")
        return

    if output_path.exists():
        output_path.unlink()
        print(f"Deleted existing {output_path} to start fresh.")

    for qp_path in qp_json_paths:
        print(f"QP loaded from {qp_path}")
        qp = BaseQP()
        qp.populate_from_json(str(qp_path))

        P_list, _, M_list = qp.Riccati_recursion()
        R_list = qp.R
        N = len(P_list) - 1
        Inde_stage_count = 0
        for i in range(N):
            eigs_M = np.linalg.eigvals(M_list[i])
            if min(eigs_M) < 0:
                Inde_stage_count += 1
                stage_name = f"{qp_path.stem}_stage_{i}"
                M_min_eig = min(eigs_M)
                R_min_eig = min(np.linalg.eigvals(R_list[i]))
                P_min_eig = min(np.linalg.eigvals(P_list[i+1]))

                write_csv_row_stage(
                    output_path,
                    {
                        "QP_stage_name": stage_name,
                        "R_min_eig": R_min_eig,
                        "P_min_eig": P_min_eig,
                        "M_min_eig": M_min_eig,
                    },
                )


def main() -> None:
    # qp_root = Path(__file__).resolve().parent / "cases" / "Tracking_inde_QP_json"
    qp_root = Path(__file__).resolve().parent / "cases" / "Time_optimal_inde_QP_json"
    # qp_root = Path(__file__).resolve().parent / "tests" / "QP_json" / "temp"
    qp_json_paths = iter_qp_json_paths(qp_root)
    output_path = Path(__file__).resolve().parent / "qp_diagonstics.csv"
    QP_diagnostics(qp_json_paths, output_path)
    # plot_indefiniteness_coverage('qp_diagonstics.csv', save_path='indefiniteness_coverage.pdf')

def main_stage_diagnostics() -> None:
    qp_root = Path(__file__).resolve().parent / "cases" / "Tracking_inde_QP_json"
    # qp_root = Path(__file__).resolve().parent / "cases" / "Time_optimal_inde_QP_json"
    qp_json_paths = iter_qp_json_paths(qp_root)
    output_path = Path(__file__).resolve().parent / "qp_stage_diagnostics.csv"
    QP_stage_diagnostics(qp_json_paths, output_path)
    # plot_stage_indefiniteness_source('qp_stage_diagnostics.csv', save_path='stage_indefiniteness.pdf')

if __name__ == "__main__":
    main()
    # main_stage_diagnostics()