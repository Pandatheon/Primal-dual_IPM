import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from acados_template import AcadosOcpQp, AcadosOcpQpSolver, AcadosOcpQpOptions
from solvers.SqpQpSolver import SqpQpSolver
from pathlib import Path

def iter_qp_json_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"QP folder not found: {root}")
    return sorted(root.glob("*.json"))

def main():
    qp = AcadosOcpQp.from_json(json_file_path=json_path)
    N = qp.N
    Sqp_solver = SqpQpSolver(qp, nlp_max_iter=2)
    status = Sqp_solver.solve()
    if status != 0:
        print(f"ocp_solver returned status {status}.")
    Sqp_solver.print_statistics()
    cost = Sqp_solver.get_cost()
    print(f"Optimal cost: {cost}")
    # Sqp_solver.dump_last_qp_to_json("/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json/furuta_cons_FU_PD_RE_PD_status_2_iter_29_SQP.json", overwrite=True)
    # x = [Sqp_solver.get(k, "x") for k in range(N+1)]
    # u = [Sqp_solver.get(k, "u") for k in range(N)]
    # lam = [Sqp_solver.get(k, "lam") for k in range(N)]
    # for k in range(N):
    #     print(f"lam[{k}] = {lam[k]}")

def main_loop(qp_json_paths: list[Path]):
    opts = AcadosOcpQpOptions()
    opts.mu0 = 1e1
    opts.hpipm_mode = 'CUSTOMIZED'
    opts.ric_alg = 0
    opts.tol_stat = 1e-6
    opts.tol_eq = 1e-8
    opts.tol_ineq = 1e-8
    opts.tol_comp = 1e-6
    opts.iter_max = 100

    for qp_path in qp_json_paths:
        qp_name = os.path.basename(qp_path)
        print(f"Runing QP {qp_name} with SqpQpSolver:")

        # SQPQP solver
        qp = AcadosOcpQp.from_json(json_file_path=qp_path)
        Sqp_solver = SqpQpSolver(qp, nlp_max_iter=2, mode='convex')
        status = Sqp_solver.solve()
        print(f"ocp_solver returned status {status}.")
        Sqp_solver.print_statistics()
        cost = Sqp_solver.get_cost()
        print(f"Optimal cost: {cost}")
        Sqp_solver = None

        # HPIPM solver
        qp_acados = AcadosOcpQp.from_json(json_file_path=str(qp_path))
        qp_solver_acados = AcadosOcpQpSolver(qp_acados, opts)
        stats = qp_solver_acados.solve()
        iter_hpipm = qp_solver_acados.get_stats("iter")

        print(f"HPIPM solved in {iter_hpipm} iterations with status {stats}")

if __name__ == "__main__":
    # # json_path = '/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json/acrobot_FU_PD_RE_PD_status_2_iter_47.json'
    # json_path = '/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json/free_flying_robot_FU_SEMI_RE_PD_status_2_iter_2.json'
    # main()

    qp_root = Path('/home/jingtao/Mehrotra_IPM/tests/QP_json/Pos_QP_json')
    qp_json_paths = iter_qp_json_paths(qp_root)
    main_loop(qp_json_paths)