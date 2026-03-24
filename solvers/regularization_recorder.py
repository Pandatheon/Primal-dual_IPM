import numpy as np


class Recorder:
    """Track regularization diagnostics and counters for the Riccati solver."""

    CAP = 1e50
    INIT_COND_REDUCTION = 1e15

    def __init__(self, horizon: int):
        self.condition_number_arr_M = np.zeros((horizon,))
        self.condition_number_arr_P = np.zeros((horizon,))
        self.adjustment_record_arr_M = np.zeros((horizon,))
        self.adjustment_record_arr_P = np.zeros((horizon,))

        self.max_cond_reduction_M = self.INIT_COND_REDUCTION
        self.max_cond_reduction_P = self.INIT_COND_REDUCTION
        self.max_adjustment_M = 0.0
        self.max_adjustment_P = 0.0
        self.reg_count = 0

    def _safe_cond(self, matrix: np.ndarray) -> float:
        cond = np.linalg.cond(matrix)
        return cond if cond < self.CAP else self.CAP

    def _safe_max(self, values: np.ndarray) -> float:
        max_value = np.max(values)
        return max_value if max_value < self.CAP else self.CAP

    def reset_stage_arrays(self) -> None:
        self.condition_number_arr_M.fill(0.0)
        self.condition_number_arr_P.fill(0.0)
        self.adjustment_record_arr_M.fill(0.0)
        self.adjustment_record_arr_P.fill(0.0)

    def record_stage_condition_numbers(self, index: int, P: np.ndarray, reduced_hessian: np.ndarray) -> None:
        self.condition_number_arr_P[index] = self._safe_cond(P)
        self.condition_number_arr_M[index] = self._safe_cond(reduced_hessian)

    def record_adjustment_M(self, index: int, eigvals_reg: np.ndarray, eigvals_base: np.ndarray, reduced_hessian_reg: np.ndarray) -> None:
        adjustment_eig_M = self._safe_max(eigvals_reg - eigvals_base)
        self.adjustment_record_arr_M[index] = adjustment_eig_M
        self.max_adjustment_M = max(self.max_adjustment_M, adjustment_eig_M)

        cond_reg = self._safe_cond(reduced_hessian_reg)
        adjustment_cond_M = cond_reg - self.condition_number_arr_M[index] if cond_reg < self.CAP else self.CAP
        self.max_cond_reduction_M = min(self.max_cond_reduction_M, adjustment_cond_M)

    def record_adjustment_P(self, index: int, eigvals_reg: np.ndarray, eigvals_base: np.ndarray, P_reg: np.ndarray) -> None:
        adjustment_eig_P = self._safe_max(eigvals_reg - eigvals_base)
        self.adjustment_record_arr_P[index] = adjustment_eig_P
        self.max_adjustment_P = max(self.max_adjustment_P, adjustment_eig_P)

        cond_reg = self._safe_cond(P_reg)
        adjustment_cond_P = cond_reg - self.condition_number_arr_P[index] if cond_reg < self.CAP else self.CAP
        self.max_cond_reduction_P = min(self.max_cond_reduction_P, adjustment_cond_P)

    def increment_reg_count(self, amount: int = 1) -> None:
        self.reg_count += amount

    def finalize(self) -> None:
        if self.max_cond_reduction_M == self.INIT_COND_REDUCTION:
            self.max_cond_reduction_M = 0.0
        if self.max_cond_reduction_P == self.INIT_COND_REDUCTION:
            self.max_cond_reduction_P = 0.0
