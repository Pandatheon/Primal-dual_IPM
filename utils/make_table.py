import pandas as pd
import numpy as np
from pathlib import Path


def compute_summary(
    filename='qp_results_riccati.csv',
    success_status=0,      # Status that means "genuinely converged"
    maxiter_codes=(2,),    # Status value(s) that mean "hit the iteration cap"
    atol=1e-6,             # absolute tolerance for "same minimum"
    rtol=1e-4,             # relative tolerance for "same minimum"
    exclude=('NO_REGULARIZE',),
    order=None,            # explicit method order; None = sort by headline score
):
    """
    Categorize every (problem, method) outcome for non-convex QP results and
    return one row per method.

    A run earns quality credit ONLY if it genuinely converged
    (Status == success_status). The per-problem best minimum is taken over
    converged runs only; runs that hit the iteration cap (Status in
    maxiter_codes) or failed otherwise are never credited, regardless of the
    approximate cost they report.

    Columns returned:
        solved_best          converged AND within tol of the best minimum
        solved_suboptimal    converged, but to a worse local minimum
        maxiter              stopped at the iteration limit (not solved)
        failed               other non-convergence            (not solved)
        solve_rate_%         100 * (solved_best + solved_suboptimal) / n_prob
        converged_and_best_% 100 * solved_best / n_prob          <-- headline
        best_given_solved_%  100 * solved_best / (solved runs)
    """
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'string':
            df[col] = df[col].str.strip()
    df = df[~df['Regularization'].isin(exclude)].copy()

    n_prob = df['QP_name'].nunique()
    best = df[df['Status'] == success_status].groupby('QP_name')['Cost'].min()

    def classify(row):
        if row['Status'] != success_status:
            return 'maxiter' if row['Status'] in maxiter_codes else 'failed'
        qp = row['QP_name']
        if qp not in best.index:                # unreachable safety net
            return 'solved_suboptimal'
        b = best[qp]
        return 'solved_best' if row['Cost'] <= b + atol + rtol * abs(b) \
            else 'solved_suboptimal'

    df['Outcome'] = df.apply(classify, axis=1)

    cats = ['solved_best', 'solved_suboptimal', 'maxiter', 'failed']
    s = (df.groupby(['Regularization', 'Outcome']).size()
           .unstack(fill_value=0)
           .reindex(columns=cats, fill_value=0))

    solved = s['solved_best'] + s['solved_suboptimal']
    s['solve_rate_%'] = (100 * solved / n_prob).round(1)
    s['converged_and_best_%'] = (100 * s['solved_best'] / n_prob).round(1)
    s['best_given_solved_%'] = (100 * s['solved_best'] / solved.replace(0, np.nan)).round(1)

    if order is None:
        order = s['converged_and_best_%'].sort_values(ascending=False).index.tolist()
    s = s.reindex(order)
    s.attrs['n_prob'] = n_prob
    return s


def _esc(text):
    """Escape LaTeX-special characters in method names."""
    for a, b in [('\\', r'\textbackslash{}'), ('_', r'\_'), ('&', r'\&'),
                 ('%', r'\%'), ('#', r'\#'), ('$', r'\$')]:
        text = text.replace(a, b)
    return text


def to_latex_table(summary,
                   caption=None,
                   label='tab:solve_quality',
                   highlight_best=True):
    """
    Render the summary DataFrame as a booktabs LaTeX table (string).
    Requires \\usepackage{booktabs} and \\usepackage{makecell} in the preamble.
    """
    n_prob = summary.attrs.get('n_prob', '')
    if caption is None:
        caption = (f"Solve quality across {n_prob} non-convex problems. A run is "
                   r"credited only if it genuinely converged (Status~0); runs that "
                   r"hit the iteration cap or failed otherwise are not credited, "
                   r"regardless of the approximate cost they report. "
                   r"``Conv.\ \& best'' is the headline score.")

    headers = [
        'Method',
        r'\makecell{Conv.\\\& best}',
        r'\makecell{Conv.\\worse}',
        r'\makecell{Max\\iter}',
        'Failed',
        r'\makecell{Solve\\rate (\%)}',
        r'\makecell{Conv.\ \&\\best (\%)}',
        r'\makecell{Best $\mid$\\solved (\%)}',
    ]

    best_idx = summary['converged_and_best_%'].idxmax() if highlight_best else None

    lines = []
    for method, r in summary.iterrows():
        cells = [
            _esc(method),
            f"{int(r['solved_best'])}",
            f"{int(r['solved_suboptimal'])}",
            f"{int(r['maxiter'])}",
            f"{int(r['failed'])}",
            f"{r['solve_rate_%']:.1f}",
            f"{r['converged_and_best_%']:.1f}",
            "--" if pd.isna(r['best_given_solved_%']) else f"{r['best_given_solved_%']:.1f}",
        ]
        if method == best_idx:                       # bold the headline winner
            cells[0] = r'\textbf{' + cells[0] + '}'
            cells[6] = r'\textbf{' + cells[6] + '}'
        lines.append('  ' + ' & '.join(cells) + r' \\')

    tex = []
    tex.append(r'\begin{table}[t]')
    tex.append(r'  \centering')
    tex.append(r'  \caption{' + caption + '}')
    tex.append(r'  \label{' + label + '}')
    tex.append(r'  \begin{tabular}{l rrrr rrr}')
    tex.append(r'    \toprule')
    tex.append('    ' + ' & '.join(headers) + r' \\')
    tex.append(r'    \midrule')
    tex.extend('  ' + ln for ln in lines)
    tex.append(r'    \bottomrule')
    tex.append(r'  \end{tabular}')
    tex.append(r'\end{table}')
    return '\n'.join(tex)


if __name__ == '__main__':
    summary = compute_summary('qp_results_riccati.csv')
    print(summary.to_string(), '\n')

    tex = to_latex_table(summary)
    print(tex)

    out = Path('solve_quality_table.tex')
    out.write_text(tex, encoding='utf-8')
    print(f'\nsaved: {out.resolve()}')