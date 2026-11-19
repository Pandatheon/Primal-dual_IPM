"""
Cumulative solved-problem profile against factorization cost.

x-axis: proxy units (the `Decomp_Flop` column), y-axis: number of problems
solved within that cost. Same layout as `plot_charts` in plot_stuff.py, with
cost in place of iteration count.

With `with_iterations=True` the figure has two panels sharing the y-axis:
iteration count on the left, proxy units on the right.

Usage:
    from utils.plot_flop_profile import plot_flop_profile
    plot_flop_profile('qp_results_riccati.csv')
    plot_flop_profile('qp_results_riccati.csv', with_iterations=True)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from utils.plot_stuff import set_ieee_style, method_color_map
except ImportError:  # running this file directly from utils/
    from plot_stuff import set_ieee_style, method_color_map


def _draw_profile(ax, solved, column, methods, method_colors, log_x):
    """Step lines of solved count vs `column` on `ax`; returns the x-limits used."""
    x_min = solved[column].min()
    x_max = solved[column].max() * (1.5 if log_x else 1.05)
    x_start = x_min / 1.5 if log_x else 0

    for method in methods:
        vals = np.sort(solved.loc[solved['Regularization'] == method, column].values)
        n = len(vals)
        if n == 0:
            continue
        x_plot = np.concatenate(([x_start], vals, [x_max]))
        y_plot = np.concatenate(([0], np.arange(1, n + 1), [n]))
        ax.step(x_plot, y_plot, where='post', label=method,
                color=method_colors[method], linewidth=2.5, alpha=0.85)
        ax.plot(vals[-1], n, marker='o', color=method_colors[method], markersize=5)

    if log_x:
        ax.set_xscale('log')
    ax.set_xlim(x_start, x_max)
    ax.grid(True, which='both', ls='--', alpha=0.2)
    return x_start, x_max


def plot_flop_profile(
    filename='qp_results_riccati.csv',
    exclude=('IPOPT', 'SQP_PROJECT'),
    log_x=True,
    with_iterations=False,
    log_iter=False,
    label_size=22,
    tick_size=20,
):
    """Step plot of solved problems (Status 0) vs proxy units, one line per method.

    A problem counts as solved at cost c if the method finished it with
    Status 0 and Decomp_Flop <= c. With `with_iterations=True`, a second panel
    on the left shows the same solved problems vs iteration count, with a
    shared y-axis. `log_x` sets the proxy-unit scale, `log_iter` the
    iteration scale. `label_size` / `tick_size` set the axis-label and
    tick-label font sizes. Returns {method: n_solved}.
    """
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce')
    df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
    df['Decomp_Flop'] = pd.to_numeric(
        df['Decomp_Flop'].replace('N/A', np.nan), errors='coerce'
    )

    # Colours from the unfiltered CSV, so each method matches the other figures.
    method_colors = method_color_map(df['Regularization'].unique())

    # Total count before dropping methods, so it matches the iteration profile.
    if 'x0_id' in df.columns:
        total_problems = df[['QP_name', 'x0_id']].drop_duplicates().shape[0]
    else:
        total_problems = df['QP_name'].nunique()

    exclude = set(exclude)
    methods = [m for m in df['Regularization'].unique() if m not in exclude]
    # One solved set for both panels, so each line ends at the same height.
    solved = df[(df['Status'] == 0) & df['Decomp_Flop'].notna()
                & df['Iteration'].notna() & df['Regularization'].isin(methods)]

    if solved.empty:
        print(f"No solved problems with Decomp_Flop in {filename}, skipping.")
        return {}

    n_solved = {m: int((solved['Regularization'] == m).sum()) for m in methods}

    if with_iterations:
        fig, (ax_it, ax_fl) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        panels = [(ax_it, 'Iteration', 'Iteration Count', log_iter),
                  (ax_fl, 'Decomp_Flop', 'Proxy units', log_x)]
    else:
        fig, ax_fl = plt.subplots(figsize=(9, 7))
        panels = [(ax_fl, 'Decomp_Flop', 'Proxy units', log_x)]

    for i, (ax, column, xlabel, log) in enumerate(panels):
        x_start, _ = _draw_profile(ax, solved, column, methods, method_colors, log)
        ax.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)
        if i == 0:
            ax.text(x_start, total_problems, f' Total Problems: {total_problems}',
                    va='bottom', ha='left', color='gray')
            ax.set_ylabel('Solved Problems (Count)', fontsize=label_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

    ax_fl.set_ylim(0, total_problems * 1.1)
    panels[0][0].legend(loc='lower right', frameon=True)  # iteration panel when shown

    plt.tight_layout()

    p = Path(filename)
    prefix = 'iter_flop_profile' if with_iterations else 'flop_profile'
    save_path = p.parent / f"{prefix}_{p.stem}.pdf"
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    return n_solved


if __name__ == "__main__":
    set_ieee_style()
    csv = 'qp_results_riccati.csv'
    if Path(csv).exists():
        print(plot_flop_profile(csv, with_iterations=True))
    else:
        print(f"{csv} not found, skipping.")
