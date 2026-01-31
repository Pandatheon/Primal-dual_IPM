import json
from QP_formulation import OCP_QP_varying
import copy
import numpy as np
import pandas as pd
import os

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

def plot_metrics(condition_number_arr_P, condition_number_arr_M, 
                 adjustment_record_arr_R, adjustment_record_arr_P,
                 max_adjustment_R, max_adjustment_P,
                 max_condition_number_M, max_condition_number_P,
                 index, N, reg_type=''):
    import matplotlib.pyplot as plt
    plt.semilogy(range(index, N), condition_number_arr_P[index:N], label='Condition number of $P_{k+1}$')
    plt.semilogy(range(index, N), condition_number_arr_M[index:N], label='Condition number of $M_k$')
    plt.semilogy(range(index, N), adjustment_record_arr_R[index:N], label='Max Adjustment to the eigenvalues of $M_k$')
    if adjustment_record_arr_P is None:
        pass
    else:
        plt.semilogy(range(index, N), adjustment_record_arr_P[index:N], label='Max Adjustment to the eigenvalues of $P_{k+1}$')
    print('P', condition_number_arr_P)
    print('M before regularization', condition_number_arr_M)
    print('Max Adjustment to the eigenvalues of M', adjustment_record_arr_R)
    print('Max Adjustment to the eigenvalues of P', adjustment_record_arr_P)
    print('Overall Max Adjustment to the eigenvalues R', max_adjustment_R)
    print('Overall Max Adjustment to the eigenvalues P', max_adjustment_P)
    print('Max Condition Number M', max_condition_number_M)
    print('Max Condition Number P', max_condition_number_P)
    plt.title('Riccati Recursion with ' + reg_type)
    plt.xlabel('Horizon N')
    plt.ylabel('Condition Number in Log Scale')
    plt.xticks(np.arange(index, N, 1.0))
    plt.grid()
    plt.legend()
    plt.show()

def write_csv(name, solver, reg, filename='qp_results.csv'):
    """
    Writes QP problem and solver data to a CSV file using Pandas.
    
    Parameters:
    QP_name (str): The name of the QP problem.
    QP_solver (dict): A dictionary containing solver details.
                      Must include keys: 'regularization', 'iteration', 
                      'cost', 'condition number'.
    filename (str): The output filename, defaults to 'qp_results.csv'.
    """
    
    # Construct a dictionary for the new row
    # Note: Values are wrapped in lists [] to create a DataFrame
    data = {
        'QP_name': [name],
        'Regularization': [reg],
        'Iteration': [solver.statistics_dict['iter_count']],
        'Cost': [solver.get_cost()],
        'Max M Condition Number': [solver.max_condition_number_M],
        'Max P Condition Number': [solver.max_condition_number_P],
        'Max R Adjustment': [solver.max_adjustment_R],
        'Max P Adjustment': [solver.max_adjustment_P],
    }
    
    # Create a temporary DataFrame for the single row
    new_df = pd.DataFrame(data)
    
    # Check if the file already exists
    file_exists = os.path.isfile(filename)
    
    # Write to CSV
    # mode='a': Append mode
    # header=not file_exists: Write header only if file is new
    # index=False: Do not write row numbers (0, 1, 2...)
    new_df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    print(f"Data appended for: {name}")