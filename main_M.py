import os
from solvers.RiccatiQPSolver import QP_solver
from utilis.utilis import read_matrix_from_npz
import copy

folder_path = "/home/jingtao/Primal-dual_IPM/cases/Designed_case"

subdirs = os.listdir(folder_path)

cases = [
    "M_indefinite_case/M_inde_P_inde_mats.npz",
    "M_indefinite_case/M_inde_P_pos_mats.npz",
    "M_positive_case/M_pos_P_pos_mats.npz",
    "M_positive_case/M_pos_P_inde_mats.npz",
]

reg_method = [
    'NO_REGULARIZE',
    'PROJECT',
    'MIRROR',
    'M_STAGE_LM',
    'RP_STAGE_LM',
    'M_STAGE_GMW',
    'RP_STAGE_GMW',
    'M_GLOBAL_LM',
    'RP_GLOBAL_LM',
]

def main():
    for case in cases:
        npz_path = os.path.join(folder_path, case)
        print(f"\n\n=== Loading case from: {npz_path} ===")
        qp_, x0 = read_matrix_from_npz(npz_path)
        print(f"Initial state x0:\n{x0}")
        for method in reg_method:
            print(f"\n\n=== Testing regularization method: {method} ===")
            try:
                copyqp = copy.deepcopy(qp_)
                qp_solver = QP_solver(copyqp)
                sol = qp_solver.solve(x0, reg=method)
            except Exception as e:
                print(f"Method {method} failed with exception: {e}")
                continue
            cost_qp = qp_solver.get_cost()
            print(f"\n=== Predicted cost qp with {method} ===\n{cost_qp}")
            qp_solver.print_statistics()

if __name__ == "__main__":
    main()