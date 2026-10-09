"""Adapter to benchmark the Riccati IPM (solvers/RiccatiQpSolver.py) with ocp_qp_benchmark.

Solver entry of an ocp_qp_benchmark configuration JSON:

    {"solver": "RICCATI_NO_REGULARIZE",
     "qp_class": "/home/jingtao/Mehrotra_IPM/benchmark_adapter.py:RiccatiQp",
     "solver_class": "/home/jingtao/Mehrotra_IPM/benchmark_adapter.py:RiccatiSolver",
     "opts": {"reg": "NO_REGULARIZE", "mehrotra": false, "iter_max": 120}}
"""

import contextlib
import copy
import io
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from acados_template import AcadosOcpIterate, AcadosOcpQp
from ocp_qp_benchmark.core import ExternalQp, ExternalQpSolver

# QP_template, solvers and utils are imported from this folder; QP_base changes the working
# directory on import, which would break the paths of the benchmark, so it is restored
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_cwd = os.getcwd()
try:
    from QP_template.QP_base import BaseQP
    from QP_template.iterate_base import iterate_sol
    from solvers.RiccatiQpSolver import QpSolver
finally:
    os.chdir(_cwd)


class RiccatiQp(ExternalQp):
    """BaseQP built from the benchmark problem."""

    def __init__(self, base_qp: BaseQP):
        self.base_qp = base_qp

    @classmethod
    def from_acados_qp(cls, qp: AcadosOcpQp) -> "RiccatiQp":
        if qp.has_slacks():
            raise NotImplementedError("soft constraints (slacks) are not supported")
        nu_0, nx_0 = qp.dims.nu[0], qp.dims.nx[0]
        idxbx_0 = qp.idxb[0][qp.idxb[0] >= nu_0] - nu_0
        if not np.array_equal(idxbx_0, np.arange(nx_0)):
            raise NotImplementedError("the x0 elimination of BaseQP needs bounds on all initial states, in order")
        if qp.dims.nbxe[0] == 0:
            qp = copy.deepcopy(qp)
            fixed = (qp.lbx[0] == qp.ubx[0]) & (qp.lbx_mask[0] != 0) & (qp.ubx_mask[0] != 0)
            qp.set("idxe", 0, np.flatnonzero(fixed) + qp.dims.nbu[0])
            qp.make_consistent()
        return cls(BaseQP().populate_from_ocp_qp(qp))

    def to_iterate(self, sol: iterate_sol) -> AcadosOcpIterate:
        """Map a recovered QpSolver solution to the acados layout.

        sol.pi[k + 1] is the multiplier of the dynamics from stage k to k + 1 (acados pi[k]),
        sol.lam is already in the acados layout [lbu, lbx, lg, ubu, ubx, ug].
        """
        N = self.base_qp.N
        return AcadosOcpIterate(
            x=[x_k.copy() for x_k in sol.x],
            u=[u_k.copy() for u_k in sol.u],
            z=[np.zeros(0) for _ in range(N)],
            sl=[np.zeros(0) for _ in range(N + 1)],
            su=[np.zeros(0) for _ in range(N + 1)],
            pi=[pi_k.copy() for pi_k in sol.pi[1:]],
            lam=[lam_k.copy() for lam_k in sol.lam],
        )


class RiccatiSolver(ExternalQpSolver):
    """QpSolver with the options of the solver entry.

    Options:
        reg, type, mehrotra, globalization: arguments of QpSolver.
        iter_max: maximum number of iterations (default: 100).
        tol_stat, tol_eq, tol_ineq, tol_comp: convergence tolerances, as in acados.
        verbose: print the solver output (default: False).

    Status: 0 converged, 1 KKT factorization failed, 2 max iterations, 3 NaN/inf residual,
    4 minimal step, -1 exception raised by the solver.
    """

    _TOLERANCES = {"tol_stat": "res_stat", "tol_eq": "res_eq", "tol_ineq": "res_ineq", "tol_comp": "res_comp"}

    def __init__(self, qp: RiccatiQp, opts: dict):
        super().__init__(qp, opts)
        opts = dict(opts)
        self.iter_max = opts.pop("iter_max", 100)
        self.verbose = opts.pop("verbose", False)
        tolerances = {self._TOLERANCES[key]: opts.pop(key) for key in list(opts) if key in self._TOLERANCES}
        self._solver = QpSolver(qp.base_qp, **opts)
        self._solver.res_max.update(tolerances)
        self._time_tot = None
        self._raised = False

    def solve(self) -> int:
        output = sys.stdout if self.verbose else io.StringIO()
        start_time = perf_counter()
        try:
            with contextlib.redirect_stdout(output):
                self._solver.solve(max_iter=self.iter_max)
        except Exception as e:
            logging.warning(f"Riccati IPM raised {type(e).__name__}: {e}")
            self._solver.status = -1
            self._raised = True
        self._time_tot = perf_counter() - start_time
        return self._solver.status

    def get_stats(self, field: str):
        if field == "iter":
            return self._solver.get_stats("iter")
        elif field == "time_tot":
            return self._time_tot
        raise NotImplementedError(f"get_stats(): field '{field}' not implemented.")

    def print_statistics(self):
        self._solver.print_statistics()

    def get_iterate(self):
        # after an exception the solution is not recovered to the full layout
        if self._raised:
            return None
        return self.qp.to_iterate(self._solver.solution)
