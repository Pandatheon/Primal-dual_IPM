from dataclasses import dataclass


no_reg = ['NO_REGULARIZE']

Hessian_reg = ['PROJECT_H', 'MIRROR_H']

condensed_Hessian_reg = ['PROJECT_M', 'MIRROR_M', 'CHOLESKY_M']

stage_reg = ['RP_STAGE_MIRROR', 'PM_STAGE_MIRROR', 'M_lookahead_CM', 'PM_STAGE_CM']

reg_list = no_reg + Hessian_reg + condensed_Hessian_reg + stage_reg

refer_list = ['IPOPT', 'SQP_PROJECT']

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization