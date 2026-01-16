import json
from QP_formulation import OCP_QP_varying
import copy
import numpy as np

def read_matrix_list(json_path, mat_str):
    """
    Reads a list of stage-varying matrices from a dictionary where keys are of the form
    'mat_str' and returns a list of matrices.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    matrix_list = []
    for key in data.keys():
        if key.startswith(mat_str):
            matrix_list.append(np.array(data[key]))
    f.close()
    return matrix_list

def read_matrix_from_npz(npz_path):
    """
    Reads a matrix from a .npz file given the matrix name.
    """
    data = np.load(npz_path)
    N = int(data['N'])
    nx = int(data['nx'])
    Q_list = [copy.deepcopy(data['Q']) for _ in range(N)]
    R_list = [copy.deepcopy(data['R']) for _ in range(N)]
    S_list = [copy.deepcopy(data['S']) for _ in range(N)]
    q_list = [copy.deepcopy(data['q']) for _ in range(N)]
    r_list = [copy.deepcopy(data['r']) for _ in range(N)]
    A_list = [copy.deepcopy(data['A']) for _ in range(N)]
    B_list = [copy.deepcopy(data['B']) for _ in range(N)]
    b_list = [copy.deepcopy(data['b']) for _ in range(N)]
    C_list = [copy.deepcopy(data['C']) for _ in range(N)]
    D_list = [copy.deepcopy(data['D']) for _ in range(N)]
    d_list = [copy.deepcopy(data['d']) for _ in range(N)]
    Q_e = copy.deepcopy(data['Q_e'])
    q_e = copy.deepcopy(data['q_e'])
    x0 = 3.0 * np.ones((nx,))
    qp = OCP_QP_varying(N, Q_list, R_list, S_list, q_list, r_list, 
                        A_list, B_list, b_list, 
                        C_list, D_list, d_list, 
                        Q_e, q_e)
    return qp, x0