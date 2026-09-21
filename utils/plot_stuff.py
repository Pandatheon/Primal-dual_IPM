from fileinput import filename
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy.stats import gmean

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

    # Convert status to numeric
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce')

    methods = df['Regularization'].unique()

    other_methods = [m for m in methods if m != 'NO_REGULARIZE']

    method_palette = sns.color_palette("tab10", n_colors=len(other_methods))
    method_colors = dict(zip(other_methods, method_palette))

    if 'NO_REGULARIZE' in methods:
        method_colors['NO_REGULARIZE'] = 'gray'

    # Calculate total problems
    if 'x0_id' in df.columns:
        total_problems = df[['QP_name', 'x0_id']].drop_duplicates().shape[0]
    else:
        total_problems = df['QP_name'].nunique()

    success_data = df[df['Status'] == 0]
    max_x_limit = success_data['Iteration'].max() * 1.05 if not success_data.empty else 100

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot step functions
    for method in methods:
        method_data = df[(df['Regularization'] == method) & (df['Status'] == 0)]
        solved_iters = np.sort(method_data['Iteration'].values)

        current_solved_count = len(solved_iters)
        if current_solved_count == 0:
            continue

        # Change start point from 0 to 1 to avoid log(0) calculation error
        x_plot = np.concatenate(([1], solved_iters, [max_x_limit]))
        y_plot = np.concatenate(([0], np.arange(1, current_solved_count + 1), [current_solved_count]))

        # Main plot
        ax.step(x_plot, y_plot, where='post', label=method,
                color=method_colors[method], linewidth=2.5, alpha=0.85)
        ax.plot(solved_iters[-1], current_solved_count, marker='o',
                color=method_colors[method], markersize=5)


    # Format main plot
    ax.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Change text x-coordinate from 0 to 10 to match xlim and avoid log(0) error
    ax.text(10, total_problems, f' Total Problems: {total_problems}',
            va='bottom', ha='left', color='gray')

    ax.set_xlabel("Iteration Count")
    ax.set_ylabel("Solved Problems (Count)")
    # Start main plot from 1 to reveal full data path
    ax.set_xlim(10, max_x_limit)
    ax.set_ylim(10, total_problems * 1.2)
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    # Safely handle file paths using pathlib
    p = Path(filename)
    save_path = p.parent / f"iteration_{p.stem}.pdf"

    # Save specific figure object before showing
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_charts_pos(filename='qp_results.csv'):
    # Load and clean data
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    # Convert status to numeric
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce')

    methods = df['Regularization'].unique()

    other_methods = [m for m in methods if m != 'NO_REGULARIZE']

    method_palette = sns.color_palette("tab10", n_colors=len(other_methods))
    method_colors = dict(zip(other_methods, method_palette))

    linestyles = ['-', '--', ':', '-.']
    method_linestyles = {m: linestyles[i % len(linestyles)] for i, m in enumerate(methods)}

    if 'NO_REGULARIZE' in methods:
        method_colors['NO_REGULARIZE'] = 'gray'

    # Calculate total problems
    if 'x0_id' in df.columns:
        total_problems = df[['QP_name', 'x0_id']].drop_duplicates().shape[0]
    else:
        total_problems = df['QP_name'].nunique()

    success_data = df[df['Status'] == 0]
    max_x_limit = success_data['Iteration'].max() * 1.05 if not success_data.empty else 100

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot step functions
    for method in methods:
        method_data = df[(df['Regularization'] == method) & (df['Status'] == 0)]
        solved_iters = np.sort(method_data['Iteration'].values)

        current_solved_count = len(solved_iters)
        if current_solved_count == 0:
            continue

        # Change start point from 0 to 1 to avoid log(0) calculation error
        x_plot = np.concatenate(([1], solved_iters, [max_x_limit]))
        y_plot = np.concatenate(([0], np.arange(1, current_solved_count + 1), [current_solved_count]))

        # Main plot
        ax.step(x_plot, y_plot, where='post', label=method,
                color=method_colors[method], linewidth=2.5, alpha=0.85,
                linestyle=method_linestyles[method])
        ax.plot(solved_iters[-1], current_solved_count, marker='o',
                color=method_colors[method], markersize=5)


    # Format main plot
    ax.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Change text x-coordinate from 0 to 10 to match xlim and avoid log(0) error
    ax.text(10, total_problems, f' Total Problems: {total_problems}',
            va='bottom', ha='left', color='gray')

    ax.set_xlabel("Iteration Count")
    ax.set_ylabel("Solved Problems (Count)")
    # Start main plot from 1 to reveal full data path
    ax.set_xlim(0, max_x_limit)
    ax.set_ylim(0, total_problems * 1.2)
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    # Safely handle file paths using pathlib
    p = Path(filename)
    save_path = p.parent / f"iteration_{p.stem}.pdf"

    # Save specific figure object before showing
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_decomp_flop_worstcase(
    filename='qp_results_riccati.csv',
    exclude=('NO_REGULARIZE', 'IPOPT', 'SQP_PROJECT'),
    common_only=False,
):

    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    df['Decomp_Flop'] = pd.to_numeric(
        df['Decomp_Flop'].replace('N/A', np.nan), errors='coerce'
    )

    exclude = set(exclude)
    targets = [m for m in df['Regularization'].unique() if m not in exclude]
    df = df[df['Regularization'].isin(targets)].copy()
    df['solved'] = df['solved'] = df['Status'].isin([0, 2]) & df['Decomp_Flop'].notna()

    P = df['QP_name'].nunique()
    n_solved_all = (df[df['solved']].groupby('Regularization')['QP_name']
                      .nunique().reindex(targets).fillna(0))
    n_failed_all = (df[df['Status'] != 0].groupby('Regularization')['QP_name']
                    .nunique().reindex(targets).fillna(0))

    work = df
    if common_only:
        solved_by = df[df['solved']].groupby('QP_name')['Regularization'].nunique()
        common = solved_by[solved_by == len(targets)].index
        work = df[df['QP_name'].isin(common)]
    solved = work[work['solved']]

    g = solved.groupby('Regularization')['Decomp_Flop']
    stats = pd.DataFrame({
        'n_solved': g.count(),
        'median':   g.median(),
        'mean':     g.mean(),
        'p95':      g.quantile(0.95),
        'p99':      g.quantile(0.99),
        'max':      g.max(),
    }).reindex(targets)
    stats['n_failed']   = n_failed_all
    stats['solve_rate'] = n_solved_all / P

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=solved, x='Regularization', y='Decomp_Flop',
        order=targets, hue='Regularization', palette='deep',
        showfliers=True, fliersize=3, linewidth=1.0,
        width=0.5, dodge=False, legend=False
    )
    ax.set_yscale('log')
    ax.set_ylabel('Units in log')
    ax.set_xlabel('')
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace('_', ' ') for t in targets], rotation=45, ha='right')
    ax.tick_params(direction='in', length=4, width=1)

    ymax = solved['Decomp_Flop'].max()
    for i, m in enumerate(targets):
        if stats.loc[m, 'n_solved'] and stats.loc[m, 'n_solved'] > 0:
            ax.annotate(f"max={stats.loc[m,'max']:.2g}\nfail={int(stats.loc[m,'n_failed'])}",
                        (i, stats.loc[m, 'max']), ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', fontsize=11)
    ax.set_ylim(top=ymax * 3)
    sns.despine()

    plt.tight_layout()
    p = Path(filename)
    save_path = p.parent / f"decomp_flop_worstcase_{p.stem}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return stats


def plot_DM_charts(filename='qp_results.csv'):
    # Load and clean data
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    df = df[~df['Regularization'].isin(['NO_REGULARIZE'])]
    df['Status'] = pd.to_numeric(df['Status'], errors='coerce')

    # Identify total number of problems
    if 'x0_id' in df.columns:
        total_problems = df[['QP_name', 'x0_id']].drop_duplicates().shape[0]
        df['Problem_ID'] = df['QP_name'] + '_' + df['x0_id'].astype(str)
    else:
        total_problems = df['QP_name'].nunique()
        df['Problem_ID'] = df['QP_name']

    # Filter successful solves to calculate minimum iterations
    success_df = df[df['Status'] == 0]

    # Create pivot table: rows=Problems, cols=Solvers, values=Iterations
    pivot_df = success_df.pivot_table(
        index='Problem_ID',
        columns='Regularization',
        values='Iteration',
        aggfunc='min'
    )

    # Calculate min iterations per problem across all solvers
    min_iters = pivot_df.min(axis=1)

    # Calculate performance ratios (tau)
    ratios = pivot_df.div(min_iters, axis=0)

    # Setup colors and figure
    methods = df['Regularization'].unique()
    method_palette = sns.color_palette("tab10", n_colors=len(methods))
    method_colors = dict(zip(methods, method_palette))

    fig, ax = plt.subplots(figsize=(9, 7))

    # Determine upper limit for x-axis (max tau)
    max_tau = ratios.max().max()

    # Plot performance profile step functions
    for method in methods:
        if method not in ratios.columns:
            continue

        # Get sorted finite ratios for current method
        method_ratios = ratios[method].dropna().values
        method_ratios = np.sort(method_ratios)

        current_solved_count = len(method_ratios)
        if current_solved_count == 0:
            continue

        # Setup x (tau) and y (fraction of problems) arrays
        x_plot = np.concatenate(([1], method_ratios, [max_tau * 1.05]))
        y_vals = np.arange(1, current_solved_count + 1) / total_problems
        y_plot = np.concatenate(([0], y_vals, [current_solved_count / total_problems]))

        # Main plot
        ax.step(x_plot, y_plot, where='post', label=method,
                color=method_colors[method], linewidth=2.5, alpha=0.85)
        ax.plot(method_ratios[-1], current_solved_count / total_problems, marker='o',
                color=method_colors[method], markersize=5)

    # Format plot
    ax.set_xlabel(r"Performance Ratio $\tau$")
    ax.set_ylabel(r"Fraction of Problems Solved $\rho_s(\tau)$")

    # Dolan-More profiles often use log2 scale for x-axis
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlim(1, max_tau * 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    # Save and show
    p = Path(filename)
    save_path = p.parent / f"dolan_more_{p.stem}.pdf"
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_indefiniteness_coverage(csv_path, save_path=None):
    """Indefiniteness coverage of a QP benchmark set as THREE independent,
    self-contained figures (no titles -- use captions):

        1. scatter : breadth (indefinite-stage rate) vs depth (|lambda_min|),
                     one soft colour/marker per OCP, quadrant annotations.
        2. breadth : single-colour histogram of the indefinite-stage rate.
        3. depth   : single-colour histogram of |lambda_min| (log x-axis).

    Returns {name: (fig, ax)}. `save_path` acts as a stem, e.g.
    "coverage.pdf" -> coverage_scatter.pdf / _breadth.pdf / _depth.pdf .
    """
    # soft 10-colour qualitative palette (muted, print-friendly)
    PALETTE = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
            '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
    HIST_FILL = '#9fb0c3'   # quiet slate-blue for the single-colour histograms
    HIST_EDGE = '#5b6b7d'

    def load_qp_diagnostics(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df['QP_name'] = df['QP_name'].str.strip()

        # Update regex to optionally match '_MIRROR' at the end of the string
        df['problem'] = df['QP_name'].str.replace(
            r'_FU_INDE_RE_INDE_status_\d+_iter_\d+(?:_MIRROR)?$', '', regex=True)

        df['breadth'] = df['Inde Stage Rate']
        df['depth'] = df['Min Negative Eigenvalue'].abs()
        return df

    def _save(fig, suffix):
        if save_path is None:
            return
        root, ext = os.path.splitext(save_path)
        ext = ext or '.pdf'
        fig.savefig(f'{root}_{suffix}{ext}', bbox_inches='tight')

    def _despine(ax, keep=('left', 'bottom')):
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_visible(s in keep)

    df = load_qp_diagnostics(csv_path)
    rate, sev = df['breadth'].values, df['depth'].values

    order = (df.groupby('problem')['depth'].median().sort_values().index.tolist())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(order)}
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    mk = {p: markers[i % len(markers)] for i, p in enumerate(order)}

    # ------------------------------------------------------------------
    # Figure 1: scatter (breadth vs depth)
    # ------------------------------------------------------------------
    fig_s, ax = plt.subplots(figsize=(9, 7))
    for p in order:
        m = df['problem'] == p
        ax.scatter(rate[m], sev[m], s=46, marker=mk[p],
                   facecolor=colors[p], edgecolor='white', linewidth=0.6,
                   alpha=0.92, label=p, zorder=3)
    ax.set_yscale('log')
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel('Indefinite-stage rate')
    ax.set_ylabel(r'$|\lambda_{\min}|$ of the reduced Hessian')
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.grid(True, which='major', ls='-', lw=0.5, color='#ededed', zorder=0)
    ax.grid(True, which='minor', ls=':', lw=0.3, color='#f4f4f4', zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)

    rmed, smed = np.median(rate), np.median(sev)
    ax.axvline(rmed, color='#b5b5b5', lw=0.8, alpha=0.8, zorder=1)
    ax.axhline(smed, color='#b5b5b5', lw=0.8, alpha=0.8, zorder=1)

    ax.legend(title='OCP problem', loc='upper center', ncol=5, frameon=False,
              fontsize=8.5, title_fontsize=9, bbox_to_anchor=(0.5, -0.16),
              columnspacing=1.2, handletextpad=0.3)
    _save(fig_s, 'scatter')

    # ------------------------------------------------------------------
    # Figure 2: breadth histogram (single colour, minimal)
    # ------------------------------------------------------------------
    fig_b, axb = plt.subplots(figsize=(9, 4))
    axb.hist(rate, bins=np.linspace(0, 1.05, 20),
             color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.6)
    axb.set_xlim(-0.03, 1.05)
    axb.set_xlabel('Indefinite-stage rate')
    axb.set_ylabel('number of QPs')
    axb.grid(True, axis='y', ls='-', lw=0.5, color='#ededed')
    axb.set_axisbelow(True)
    _despine(axb)
    _save(fig_b, 'breadth')

    # ------------------------------------------------------------------
    # Figure 3: depth histogram (single colour, log x)
    # ------------------------------------------------------------------
    fig_d, axd = plt.subplots(figsize=(9, 4))
    # Set a lower bound to prevent log10(0)
    min_val = max(sev.min(), 1e-8)
    max_val = sev.max()
    # Generate log bins with the lower bound
    logbins = np.logspace(np.log10(min_val), np.log10(max_val), 20)
    axd.hist(sev, bins=logbins,
             color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.6)
    axd.set_xscale('log')
    axd.set_xlabel(r'$|\lambda_{\min}|$ of the reduced Hessian')
    axd.set_ylabel('number of QPs')
    axd.grid(True, axis='y', ls='-', lw=0.5, color='#ededed')
    axd.set_axisbelow(True)
    _despine(axd)
    _save(fig_d, 'depth')

    plt.show()
    return {'scatter': (fig_s, ax),
            'breadth': (fig_b, axb),
            'depth': (fig_d, axd)}


def plot_stage_indefiniteness_source(csv_path, save_path=None):
    """Where stage-wise indefiniteness comes from, as TWO self-contained
    figures (no titles -- use captions):

        1. source    : how often the cost block R, the cost-to-go block P,
                       or both are indefinite across all indefinite stages.
        2. signature : per-OCP 100% stacked share of the same three classes.

    Returns (fig_source, fig_signature). `save_path` acts as a stem, e.g.
    "stage.pdf" -> stage_source.pdf / stage_signature.pdf .
    """
    # same muted, print-friendly family used by plot_indefiniteness_coverage
    COL = {'R only':   '#c44e52',    # soft red
           'both':     '#8172b3',    # soft purple
           'P only':   '#4c72b0',    # soft blue
           'neither':  '#8c8c8c'}
    LABEL = {'R only': r'$R \prec 0$ only',
             'both':   r'both',
             'P only': r'$P \prec 0$ only'}

    def load_stage_diagnostics(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for c in ['R_min_eig', 'P_min_eig', 'M_min_eig']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['R_min_eig', 'P_min_eig', 'M_min_eig']).copy()

        # Match optional '_MIRROR' between iter and stage
        df['problem'] = df['QP_stage_name'].str.strip().str.replace(
            r'_FU_INDE_RE_INDE_status_\d+_iter_\d+(?:_MIRROR)?_stage_\d+$', '', regex=True)

        Rneg, Pneg = df['R_min_eig'] < 0, df['P_min_eig'] < 0
        df['cls'] = np.select(
            [Rneg & ~Pneg, ~Rneg & Pneg, Rneg & Pneg],
            ['R only', 'P only', 'both'], default='neither')
        return df

    def _save(fig, suffix):
        if save_path is None:
            return
        root, ext = os.path.splitext(save_path)
        ext = ext or '.pdf'
        fig.savefig(f'{root}_{suffix}{ext}', bbox_inches='tight')

    def _despine(ax, keep=('left', 'bottom')):
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_visible(s in keep)

    df = load_stage_diagnostics(csv_path)
    order_cls = ['R only', 'both', 'P only']

    # ------------------------------------------------------------------
    # Figure 1: indefiniteness source (counts over all indefinite stages)
    # ------------------------------------------------------------------
    counts = df['cls'].value_counts()
    freqs = np.array([counts.get(c, 0) for c in order_cls], dtype=float)
    total = freqs.sum()

    fig_s, axs = plt.subplots(figsize=(8, 6))
    x = np.arange(len(order_cls))
    axs.bar(x, freqs, width=0.55, color=[COL[c] for c in order_cls],
            edgecolor='white', linewidth=0.8, zorder=3)

    axs.set_xticks(x)
    axs.set_xticklabels([LABEL[c] for c in order_cls])
    axs.set_xlabel('Indefiniteness source')
    axs.set_ylabel('Number of indefinite stages')
    axs.set_xlim(-0.65, len(order_cls) - 0.35)
    axs.set_ylim(0, freqs.max() * 1.18 if total else 1)
    axs.grid(True, axis='y', ls='-', lw=0.5, color='#ededed', zorder=0)
    axs.set_axisbelow(True)
    axs.tick_params(direction='in', length=4, width=1)
    _despine(axs)

    for xi, f in zip(x, freqs):
        share = f / total if total else 0.0
        axs.annotate(f'{int(f)}\n({share:.0%})', (xi, f), ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points', fontsize=14)
    _save(fig_s, 'source')

    # ------------------------------------------------------------------
    # Figure 2: per-OCP signature (100% stacked shares)
    # ------------------------------------------------------------------
    ct = pd.crosstab(df['problem'], df['cls'])
    for c in order_cls:
        if c not in ct:
            ct[c] = 0
    ct = ct[order_cls]

    frac = ct.div(ct.sum(axis=1), axis=0)
    fam_order = frac.sort_values(['R only', 'both'], ascending=True).index.tolist()
    frac, tot = frac.loc[fam_order], ct.sum(axis=1).loc[fam_order]

    fig_g, axg = plt.subplots(figsize=(9, max(4.0, 0.62 * len(fam_order) + 2.0)))
    y = np.arange(len(fam_order))
    left = np.zeros(len(fam_order))

    for c in order_cls:
        w = frac[c].values
        axg.barh(y, w, left=left, height=0.68, color=COL[c],
                 edgecolor='white', linewidth=0.8, zorder=3)
        # label only segments wide enough to hold the text
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi >= 0.14:
                axg.text(li + wi / 2, yi, f'{wi:.0%}', ha='center', va='center',
                         fontsize=12, color='white', zorder=4)
        left = left + w

    axg.set_yticks(y)
    axg.set_yticklabels([f"{f.replace('_', ' ')}  (n={int(tot[f])})" for f in fam_order])
    axg.set_ylim(-0.6, len(fam_order) - 0.4)
    axg.set_xlim(0, 1)
    axg.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    axg.set_xlabel('Share of indefinite stages')
    axg.grid(True, axis='x', ls='-', lw=0.5, color='#ededed', zorder=0)
    axg.set_axisbelow(True)
    axg.tick_params(axis='x', direction='in', length=4, width=1)
    axg.tick_params(axis='y', length=0)
    _despine(axg, keep=('bottom',))

    axg.legend(handles=[Patch(facecolor=COL[c], label=LABEL[c]) for c in order_cls],
               loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
               frameon=False, columnspacing=1.4, handletextpad=0.5)
    _save(fig_g, 'signature')

    plt.show()
    return fig_s, fig_g

# Entry point
if __name__ == "__main__":
    set_ieee_style()

    iter = 1
    if iter:
        plot_charts('qp_results_riccati.csv') if Path('qp_results_riccati.csv').exists() else print("qp_results_riccati.csv not found, skipping.")
        plot_charts_pos('qp_results_pos.csv') if Path('qp_results_pos.csv').exists() else print("qp_results_pos.csv not found, skipping.")
        plot_decomp_flop_worstcase('qp_results_riccati.csv') if Path('qp_results_riccati.csv').exists() else print("qp_results_riccati.csv not found, skipping.")
    else:
        plot_indefiniteness_coverage('qp_diagonstics.csv', save_path='indefiniteness_coverage.pdf') if Path('qp_diagonstics.csv').exists() else print("qp_diagonstics.csv not found, skipping.")
        plot_stage_indefiniteness_source('qp_stage_diagnostics.csv', save_path='stage_indefiniteness.pdf') if Path('qp_stage_diagnostics.csv').exists() else print("qp_stage_diagnostics.csv not found, skipping.")