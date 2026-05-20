from fileinput import filename
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np

OVERFLOW_THRESHOLD = 1e16

def set_ieee_style():
    # 1.Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 2. Font Sizes
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14

def plot_charts(filename='qp_results.csv'):
    # Load and clean data
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    df = df[~df['Regularization'].isin(['NO_REGULARIZE'])]

    # Convert status to numeric
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce')

    methods = df['Regularization'].unique()
    method_palette = sns.color_palette("tab10", n_colors=len(methods))
    method_colors = dict(zip(methods, method_palette))

    # Calculate total problems
    if 'x0_id' in df.columns:
        total_problems = df[['QP_name', 'x0_id']].drop_duplicates().shape[0]
    else:
        total_problems = df['QP_name'].nunique()

    success_data = df[df['Status'] == 0]
    max_x_limit = success_data['Iteration'].max() * 1.05 if not success_data.empty else 100

    plt.figure(figsize=(9, 7))

    # Plot step functions
    for method in methods:
        method_data = df[(df['Regularization'] == method) & (df['Status'] == 0)]
        solved_iters = np.sort(method_data['Iteration'].values)

        current_solved_count = len(solved_iters)
        if current_solved_count == 0:
            continue

        x_plot = np.concatenate(([0], solved_iters, [max_x_limit]))
        y_plot = np.concatenate(([0], np.arange(1, current_solved_count + 1), [current_solved_count]))

        plt.step(x_plot, y_plot, where='post', label=method,
                 color=method_colors[method], linewidth=2.5, alpha=0.85)

        plt.plot(solved_iters[-1], current_solved_count, marker='o',
                 color=method_colors[method], markersize=5)

    # Format plot
    plt.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(0, total_problems, f' Total Problems: {total_problems}',
             va='bottom', ha='left', color='gray')

    plt.xlabel("Iteration Count")
    plt.ylabel("Solved Problems (Count)")
    plt.xlim(0, max_x_limit)
    plt.ylim(0, total_problems * 1.05)

    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc="best", frameon=True)

    plt.tight_layout()
    name = filename.split('.')[0]
    plt.savefig(f"iteration_{name}.pdf", bbox_inches='tight')
    plt.show()

# Entry point
if __name__ == "__main__":
    set_ieee_style()
    plot_charts('qp_results_riccati.csv')
    plot_charts('qp_results_condensed.csv')
    plot_charts('qp_results_pos.csv')