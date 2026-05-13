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
    """
    Hybrid Analysis Chart:
    1. M/P Metrics: Side-by-Side (1x2) with Independent Y-axis.
    2. Iteration: Restored to ORIGINAL Baseline (Single large plot).
    """
    try:
        # --- 1. Load and Clean (Shared) ---
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()

        df = df[~df['Regularization'].isin(['NO_REGULARIZE'])]

        if df.empty:
            print("Error: DataFrame is empty.")
            return

        methods = df['Regularization'].unique()
        qp_names = df['QP_name'].unique()

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
        marker_cycle = itertools.cycle(markers)
        color_cycle = itertools.cycle(plt.cm.tab10.colors)

        style_map = {}
        for name in qp_names:
            style_map[name] = {'marker': next(marker_cycle), 'color': next(color_cycle)}

        print("Generating Iteration Performance Profile (Status == 0)...")

        if 'Iteration' in df.columns and 'Status' in df.columns:
            # 1. Prepare data
            iter_df = df.copy()
            metric = 'Iteration'

            if iter_df['Status'].dtype == object:
                iter_df['Status'] = pd.to_numeric(iter_df['Status'], errors='coerce')

            iter_methods = [m for m in methods if m in iter_df['Regularization'].unique()]
            total_problems = iter_df['QP_name'].nunique()

            method_palette = sns.color_palette("tab10", n_colors=len(iter_methods))
            method_colors = dict(zip(iter_methods, method_palette))

            success_data = iter_df[iter_df['Status'] == 0]

            if not success_data.empty:
                max_iter_val = success_data[metric].max()
                max_x_limit = max_iter_val * 1.05
            else:
                max_x_limit = 100

            plt.figure(figsize=(9, 7))

            for method in iter_methods:
                method_data = iter_df[iter_df['Regularization'] == method]

                valid_mask = (method_data['Status'] == 0)
                solved_iters = method_data[valid_mask][metric].values
                solved_iters.sort()

                x_plot = np.concatenate(([0], solved_iters, [max_x_limit]))
                current_solved_count = len(solved_iters)
                y_plot = np.concatenate(([0], np.arange(1, current_solved_count + 1), [current_solved_count]))
                plt.step(x_plot, y_plot, where='post', label=method,
                         color=method_colors[method], linewidth=2.5, alpha=0.85)

                if len(solved_iters) > 0:
                    plt.plot(solved_iters[-1], current_solved_count, marker='o',
                             color=method_colors[method], markersize=5)

            plt.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.text(0, total_problems, f' Total Problems: {total_problems}',
                     va='bottom', ha='left', color='gray')

            plt.xlabel("Iteration Count")
            plt.ylabel("Solved Problems (Count)")

            plt.xlim(0, max_x_limit)
            plt.ylim(0, total_problems * 1.05)

            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(loc="lower right", frameon=True)

            plt.tight_layout()
            plt.savefig("iteration_performance_profile.pdf", bbox_inches='tight')

        else:
            print("Skipping Iteration Chart: 'Iteration' or 'Status' column missing.")

        print("Displaying all windows...")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Entry point
if __name__ == "__main__":
    set_ieee_style()
    plot_charts()