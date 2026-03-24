import numpy as np
import json
import os
from pathlib import Path

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

def main():
    # Define input and output directories
    input_dir = '/home/jingtao/Primal-dual_IPM/cases/M_indefinite_case'
    output_dir = '/home/jingtao/Primal-dual_IPM/cases/M_indefinite_case'
    
    # Get all .npz files
    npz_files = sorted(Path(input_dir).glob('*.npz'))
    
    print(f"Found {len(npz_files)} .npz files to convert")
    print()
    
    # Convert each file
    for npz_file in npz_files:
        convert_npz_to_json(str(npz_file), output_dir)
    
    print()
    print("Conversion complete!")

if __name__ == '__main__':
    main()
