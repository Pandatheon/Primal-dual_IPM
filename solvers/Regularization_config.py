from dataclasses import dataclass


no_reg = ['NO_REGULARIZE']

Block_Hessian_reg = ['H_PROJECT', 'H_MIRROR', 'H_GMW']

Block_reduced_Hessian_reg = [
    'M_PROJECT', 'M_MIRROR', 'M_GMW', 
    'PM_PROJECT', 'PM_MIRROR', 'PM_GMW'
    ]

Iteration_based_reg = ['INERTIA_LM', 'PRECOMPUTE']

reg_list = no_reg + Block_Hessian_reg + Block_reduced_Hessian_reg

Refer_list = ['IPOPT']

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization