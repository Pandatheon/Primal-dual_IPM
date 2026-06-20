from dataclasses import dataclass


no_reg = ['NO_REGULARIZE']

Block_Hessian_reg = ['PROJECT_H', 'MIRROR_H']

Block_reduced_Hessian_reg = ['PM_STAGE_MIRROR', 'PM_STAGE_SE', 'PM_STAGE_GMW']

Iteration_based_reg = ['INERTIAL_LM', 'PRECOMPUTE']

reg_list = no_reg + Block_Hessian_reg + Block_reduced_Hessian_reg

Refer_list = ['IPOPT']

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization