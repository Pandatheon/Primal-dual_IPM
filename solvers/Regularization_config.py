from dataclasses import dataclass


no_reg = ['NO_REGULARIZE']

Hessian_reg = ['PROJECT_H', 'MIRROR_H', 'CHOLESKY_H']

condensed_Hessian_reg = ['PROJECT_M', 'MIRROR_M', 'CHOLESKY_M']

stage_reg = ['RP_STAGE_MIRROR', 'PM_STAGE_MIRROR', 'M_lookahead_CM', 'PM_lookahead_CM']

reg_list = no_reg + Hessian_reg + condensed_Hessian_reg + stage_reg

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization