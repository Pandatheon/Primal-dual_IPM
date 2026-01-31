
class Regularizations_config:
    def __init__(self):
        self.Hessian_based_regularization = ['NO_REGULARIZE', 'PROJECT', 'MIRROR']
        self.stage_based_regularization = ['R_STAGE_EMP', 'PR_STAGE_EMP',
                                                    'R_STAGE_EMM', 'PR_STAGE_EMM',
                                                    'RP_STAGE_PROJECT', 'RP_STAGE_MIRROR']
        self.global_based_regularization = ['R_GLOBAL_EMP', 'PR_GLOBAL_EMP']
        self.delta = 1e-4  # Small positive value used in regularization adjustments