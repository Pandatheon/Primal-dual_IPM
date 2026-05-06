from dataclasses import dataclass


Hessian_based_regularization = ['NO_REGULARIZE', 'PROJECT', 'MIRROR']

Reduced_Hessian_based_regularization = ['M_STAGE_PROJECT', 'PM_STAGE_PROJECT','RP_STAGE_PROJECT',
                                    'M_STAGE_MIRROR', 'PM_STAGE_MIRROR','RP_STAGE_MIRROR']

reg_list = Hessian_based_regularization + Reduced_Hessian_based_regularization

@dataclass
class Regularizations_config:
    method: str = ' '
    delta: float = 1e-4  # Small positive value used in regularization