import json
import copy
import numpy as np
import pandas as pd
import os

def read_matrix_list(json_path, mat_str):
    """
    Reads a list of stage-varying matrices from a dictionary where keys are of the form
    'mat_str' and returns a list of matrices.
    """
    matrix_field = ['Q', 'R', 'S', 'A', 'B', 'C', 'D']
    vector_field = ['q', 'r', 'b', 'd']
    with open(json_path, "r") as f:
        data = json.load(f)
    matrix_list = []
    for key in data.keys():
        if key.startswith(mat_str):
            if mat_str in matrix_field:
                matrix_list.append(np.array(data[key]))
            elif mat_str in vector_field:
                matrix_list.append(np.array(data[key]).flatten())
    f.close()
    return matrix_list

def plot_metrics(condition_number_arr_P, condition_number_arr_M, 
                 adjustment_record_arr_M, adjustment_record_arr_P,
                 max_adjustment_M, max_adjustment_P,
                 max_condition_number_M, max_condition_number_P,
                 index, N, reg_type=''):
    import matplotlib.pyplot as plt
    plt.semilogy(range(index, N), condition_number_arr_P[index:N], label='Condition number of $P_{k+1}$')
    plt.semilogy(range(index, N), condition_number_arr_M[index:N], label='Condition number of $M_k$')
    plt.semilogy(range(index, N), adjustment_record_arr_M[index:N], label='Max Adjustment to the eigenvalues of $M_k$')
    if adjustment_record_arr_P is None:
        pass
    else:
        plt.semilogy(range(index, N), adjustment_record_arr_P[index:N], label='Max Adjustment to the eigenvalues of $P_{k+1}$')
    print('P', condition_number_arr_P)
    print('M before regularization', condition_number_arr_M)
    print('Max Adjustment to the eigenvalues of M', adjustment_record_arr_M)
    print('Max Adjustment to the eigenvalues of P', adjustment_record_arr_P)
    print('Overall Max Adjustment to the eigenvalues M', max_adjustment_M)
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

def write_csv(name, solver, recorder, reg, filename='qp_results.csv'):
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
        'Regularization Count': [int(recorder.reg_count/2)],
        'Cost': [solver.get_cost()],
        'Max M Condition Number Reduction': [recorder.max_cond_reduction_M],
        'Max P Condition Number Reduction': [recorder.max_cond_reduction_P],
        'Max M Adjustment': [recorder.max_adjustment_M],
        'Max P Adjustment': [recorder.max_adjustment_P],
        'Status': [solver.status],
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

def write_csv_acados(name, reg_name, cost, filename='qp_results.csv'):
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
        'Regularization': [reg_name],
        'Iteration': [0],
        'Cost': [cost],
        'Max M Condition Number Reduction': [None],
        'Max P Condition Number Reduction': [None],
        'Max M Adjustment': [None],
        'Max P Adjustment': [None],
        'Status': [0],
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

def convert_npz_to_json(npz_file, output_dir):
    """Convert .npz file to JSON format similar to acados_ocp_qp.json"""
    
    # Load the .npz file
    data = np.load(npz_file)
    
    # Extract data
    A = data['A']  # Shape: (nx, nx)
    B = data['B']  # Shape: (nx, nu)
    Q = data['Q']  # Shape: (nx, nx)
    S = data['S']  # Shape: (nu, nx)
    R = data['R']  # Shape: (nu, nu)
    Q_e = data['Q_e']  # Shape: (nx, nx)
    N = int(data['N'])  # Number of time steps
    
    # Create JSON structure
    json_data = {}
    
    # Add A matrices for each time step (0 to N-1)
    for i in range(N):
        json_data[f'A_{i}'] = A.tolist()
    
    # Add B matrices for each time step (0 to N-1)
    for i in range(N):
        json_data[f'B_{i}'] = B.tolist()
    
    # Add Q matrices for each time step (0 to N-1) and terminal Q_e at step N
    for i in range(N):
        json_data[f'Q_{i}'] = Q.tolist()
    json_data[f'Q_{N}'] = Q_e.tolist()
    
    # Add R matrices for each time step (0 to N-1)
    for i in range(N):
        json_data[f'R_{i}'] = R.tolist()
    
    # Add S matrices for each time step (0 to N-1)
    for i in range(N):
        json_data[f'S_{i}'] = S.tolist()
    
    # Determine output filename
    npz_filename = os.path.basename(npz_file)
    json_filename = npz_filename.replace('.npz', '.json')
    output_path = os.path.join(output_dir, json_filename)
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Converted {npz_filename} -> {json_filename}")
    return output_path