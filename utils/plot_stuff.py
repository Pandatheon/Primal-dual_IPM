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

def plot_average_cost(filename='qp_results.csv'):
    # Load and clean data
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    # Filter targets and status
    targets = [m for m in df['Regularization'].unique() if m != 'NO_REGULARIZE']
    df_plot = df[(df['Status'] == 0) & (df['Regularization'].isin(targets))]

    # Calculate average cost
    average_cost = df_plot.groupby('Regularization')['Cost'].mean().reset_index()

    # Set figure size
    plt.figure(figsize=(8, 6))

    # Apply barplot styling
    ax = sns.barplot(
        data=average_cost, x='Regularization', y='Cost', hue='Regularization',
        order=targets, palette='deep', edgecolor='black', linewidth=1.2,
        width=0.5, dodge=False, legend=False
    )

    # Add numeric labels to negative bars
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            ax.annotate(f'{height:.0f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='top', 
                        xytext=(0, -5), 
                        textcoords='offset points',
                        fontsize=14)

    # Adjust y-axis range to emphasize differences
    ax.set_ylim(average_cost['Cost'].min() * 1.1, 0) 

    # Apply axis styling
    ax.set_ylabel('Average Cost (Successful)')
    ax.set_xlabel('')
    ax.set_xticklabels([t.replace('_', ' ') for t in targets])
    ax.tick_params(direction='in', length=4, width=1)
    ax.tick_params(axis='x', rotation=60)
    sns.despine()

    # Save and display
    plt.tight_layout()
    name = filename.split('.')[0]
    plt.savefig(f"cost_{name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_polar_iteration_cost(filename='qp_results_riccati.csv'):
    # Load and clean data
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    # Convert to numeric format
    df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

    # Filter target regularizations
    targets = [m for m in df['Regularization'].unique() if m != 'NO_REGULARIZE']
    df_plot = df[df['Regularization'].isin(targets)].copy()

    # Calculate Max Iteration (considering all status cases)
    max_iters = df_plot.groupby('Regularization')['Iteration'].max().reset_index()
    max_iters.rename(columns={'Iteration': 'Max_Iteration'}, inplace=True)

    # Calculate Average Cost (considering only successful cases where Status == 0)
    if 'Status' in df.columns:
        df_success = df_plot[df_plot['Status'] == 0].dropna(subset=['Cost'])
    else:
        df_success = df_plot.dropna(subset=['Cost'])
        
    avg_costs = df_success.groupby('Regularization')['Cost'].mean().reset_index()
    avg_costs.rename(columns={'Cost': 'Avg_Cost'}, inplace=True)

    # Merge iteration and cost stats
    stats = pd.merge(max_iters, avg_costs, on='Regularization', how='inner')

    # Normalize average cost to [0, pi/2]
    cost_min = stats['Avg_Cost'].min()
    cost_max = stats['Avg_Cost'].max()
    
    if cost_max == cost_min:
        stats['Angle'] = np.pi / 4 
    else:
        stats['Angle'] = (stats['Avg_Cost'] - cost_min) / (cost_max - cost_min) * (np.pi / 2)

    # Setup polar plot
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': 'polar'})
    
    # Restrict to first quadrant
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))

    # Plot each method as a line from origin
    for i, row in stats.iterrows():
        theta = row['Angle']
        r = row['Max_Iteration']
        label = row['Regularization']
        
        ax.plot([0, theta], [0, r], color=colors[i], linewidth=2, marker='o', 
                label=f"{label}")

    # Format axes and legend
    ax.set_xlabel("Average Cost (Angle)", labelpad=20)
    ax.set_ylabel("Max Iteration (Radius)", labelpad=30)
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    name = filename.split('.')[0]
    plt.savefig(f"polar_{name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# Entry point
if __name__ == "__main__":
    set_ieee_style()
    plot_charts('qp_results_riccati.csv')
    plot_charts('qp_results_pos.csv')
    plot_average_cost('qp_results_riccati.csv')
    # plot_polar_iteration_cost('qp_results_riccati.csv')