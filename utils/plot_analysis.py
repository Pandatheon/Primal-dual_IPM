"""
Plot QP Riccati solver failures by regularization method and failure status.

- Merges the 3 result CSVs and de-duplicates on (QP_name, Regularization),
  since the files are overlapping exports of the same 175-problem benchmark.
- Excludes NO_REGULARIZE and Status 0 (success).
- Failure status: 1 = Unregularized, 2 = Max iterations reached, 3 = NaN encountered.

Usage:
    from plot_analysis import plot_qp_failures, plot_failures_by_category
    plot_qp_failures(["file1.csv", "file2.csv", "file3.csv"])
    plot_failures_by_category(["file1.csv", "file2.csv", "file3.csv"])
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

try:
    from utils.plot_stuff import set_ieee_style
except ImportError:  # running this file directly from utils/
    from plot_stuff import set_ieee_style


# Failure status code -> label and colour (status 0 = success, excluded).
# Same muted, print-friendly family used by plot_stuff.py.
STATUS_LABELS = {1: "Unregularized", 2: "Max iterations reached", 3: "NaN encountered"}
STATUS_COLORS = {1: "#4c72b0", 2: "#dd8452", 3: "#c44e52"}

# Sequential ramp for the heatmap, light -> dark in the same muted blue family.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "muted_blue", ["#f2f4f8", "#c7d2e3", "#8fa5c6", "#4c72b0", "#2f4a78"]
)


def _despine(ax, keep=('left', 'bottom')):
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_visible(s in keep)


def load_failures(csv_paths, exclude_methods=("NO_REGULARIZE",)):
    """Merge the CSV exports and return (crosstab, problems_per_method).

    `crosstab` is a method-by-status count table (failing statuses only),
    sorted so the method with the most failures comes last (top of a barh).
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        for col in ("QP_name", "Regularization"):
            df[col] = df[col].astype(str).str.strip()
        frames.append(df)

    # Files overlap on shared methods, so keep one row per (problem, method).
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["QP_name", "Regularization"])
    if exclude_methods:
        data = data[~data["Regularization"].isin(exclude_methods)]

    n_total = data.groupby("Regularization").size()

    statuses = sorted(STATUS_LABELS)
    fail = data[data["Status"] != 0]
    ct = (
        pd.crosstab(fail["Regularization"], fail["Status"])
        .reindex(columns=statuses)
        .fillna(0)
        .astype(int)
    )
    ct["tot"] = ct[statuses].sum(axis=1)
    return ct.sort_values("tot"), n_total


def load_category_failures(csv_paths, exclude_methods=("NO_REGULARIZE",)):
    """Merge the CSV exports and return (count, rate, problems_per_category).

    The problem category is the first token of QP_name (e.g. "furuta",
    "pendulum"). `count` holds failed-problem counts per (method, category);
    `rate` is the failure rate (%) within each category. Both are ordered with
    the worst method on top and the largest category on the left.
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        for col in ("QP_name", "Regularization"):
            df[col] = df[col].astype(str).str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["QP_name", "Regularization"])
    if exclude_methods:
        data = data[~data["Regularization"].isin(exclude_methods)]
    data = data.assign(category=data["QP_name"].str.split("_").str[0])

    n_cat = data.groupby("category")["QP_name"].nunique()
    fail = data[data["Status"] != 0]
    count = pd.crosstab(fail["Regularization"], fail["category"])

    methods = sorted(data["Regularization"].unique())
    cats = list(n_cat.sort_values(ascending=False).index)  # big -> small
    count = count.reindex(index=methods, columns=cats).fillna(0).astype(int)
    count = count.loc[count.sum(axis=1).sort_values(ascending=False).index]

    rate = count.divide([n_cat[c] for c in count.columns], axis=1) * 100
    return count, rate, n_cat


def plot_qp_failures(csv_paths, save_path="qp_failures_by_status.pdf",
                     exclude_methods=("NO_REGULARIZE",), n_problems=175):
    """Draw a stacked horizontal bar chart of failures per method and save it.

    Self-contained figure (no title -- use a caption).

    csv_paths: list of QP result CSV files.
    save_path: output figure path (returned on success).
    """
    ct, n_total = load_failures(csv_paths, exclude_methods)
    statuses = sorted(STATUS_LABELS)

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.62 * len(ct) + 2.0)))

    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for s in statuses:
        vals = ct[s].values
        ax.barh(y, vals, left=left, height=0.68, color=STATUS_COLORS[s],
                edgecolor='white', linewidth=0.8, zorder=3)
        # label only segments wide enough to hold the text
        for i, (v, base) in enumerate(zip(vals, left)):
            if v >= 3:
                ax.text(base + v / 2, i, str(v), ha='center', va='center',
                        fontsize=12, color='white', zorder=4)
        left = left + vals

    # Total count + failure rate at the end of each bar.
    for i, (method, row) in enumerate(ct.iterrows()):
        tot = row["tot"]
        share = tot / n_total[method]
        ax.annotate(f'{int(tot)} ({share:.0%})', (tot, i), ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points', fontsize=14)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace('_', ' ') for m in ct.index])
    ax.set_ylim(-0.6, len(ct) - 0.4)
    ax.set_xlim(0, ct["tot"].max() * 1.18)
    ax.set_xlabel(f'Number of failed problems (out of {n_problems})')
    ax.grid(True, axis='x', ls='-', lw=0.5, color='#ededed', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', direction='in', length=4, width=1)
    ax.tick_params(axis='y', length=0)
    _despine(ax, keep=('bottom',))

    ax.legend(handles=[Patch(facecolor=STATUS_COLORS[s], label=STATUS_LABELS[s])
                       for s in statuses],
              loc='lower right', frameon=True, handletextpad=0.5)

    plt.tight_layout()
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        fig.savefig(f'{root}{ext or ".pdf"}', bbox_inches='tight')
        print(f"saved {save_path}")
    plt.show()
    return fig, ax


def plot_failures_by_category(csv_paths, save_path="qp_failures_by_category.pdf",
                              exclude_methods=("NO_REGULARIZE",)):
    """Draw a method-by-category failure heatmap and save it.

    Cell text = number of failed problems (the direct answer); cell colour =
    failure rate within the category, which keeps categories of different size
    comparable. Self-contained figure (no title -- use a caption).

    csv_paths: list of QP result CSV files.
    save_path: output figure path (returned on success).
    """
    count, rate, n_cat = load_category_failures(csv_paths, exclude_methods)
    methods, cats = list(count.index), list(count.columns)

    fig, ax = plt.subplots(figsize=(1.0 * len(cats) + 2.5,
                                    0.55 * len(methods) + 1.8))
    im = ax.imshow(rate.values, cmap=SEQ_CMAP, vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([f'{c}\n(N={n_cat[c]})' for c in cats])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([m.replace('_', ' ') for m in methods])
    ax.tick_params(length=0)
    _despine(ax, keep=())

    # thin white gap between cells
    ax.set_xticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    # cell text = raw count; switch to white ink on dark cells
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            v = int(count.values[i, j])
            ax.text(j, i, str(v), ha='center', va='center', fontsize=12,
                    color='white' if rate.values[i, j] >= 50 else 'black')

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label('Failure rate within category (%)')
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)

    plt.tight_layout()
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        fig.savefig(f'{root}{ext or ".pdf"}', bbox_inches='tight')
        print(f"saved {save_path}")
    plt.show()
    return fig, ax


if __name__ == "__main__":
    import sys
    set_ieee_style()

    files = sys.argv[1:] or [
        "/home/jingtao/Mehrotra_IPM/temp/Full_Hessian_results/qp_results_riccati.csv",
        "/home/jingtao/Mehrotra_IPM/temp/Inertia_correction_results/qp_results_riccati.csv",
        "/home/jingtao/Mehrotra_IPM/temp/Riccati-based_results/qp_results_riccati.csv",
    ]
    plot_qp_failures(files)
    plot_failures_by_category(files)