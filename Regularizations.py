
class Regularizations_config:
    def __init__(self):
        self.Hessian_based_regularization = ['NO_REGULARIZE', 'PROJECT', 'MIRROR']
        self.Reduced_Hessian_based_regularization = ['M_STAGE_PROJECT', 'PM_STAGE_PROJECT','RP_STAGE_PROJECT',
                                           'M_STAGE_MIRROR', 'PM_STAGE_MIRROR','RP_STAGE_MIRROR']
        # self.global_based_regularization = ['R_GLOBAL_EMP', 'PR_GLOBAL_EMP']
        self.delta = 1e-4  # Small positive value used in regularization adjustments