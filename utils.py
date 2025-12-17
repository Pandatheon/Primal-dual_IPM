import json
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