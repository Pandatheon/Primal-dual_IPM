from dataclasses import dataclass


Hessian_reg = ['NO_REGULARIZE', 'PROJECT', 'MIRROR']

condensed_Hessian_reg = ['M_PROJECT', 'M_MIRROR', 'M_CM']

stage_reg = ['PM_STAGE_MIRROR', 'M_lookahead_CM', 'PM_lookahead_CM']

reg_list = Hessian_reg + condensed_Hessian_reg + stage_reg

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization