from fileinput import filename
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np

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

        # Global Filters
        df = df[~df['Regularization'].isin(['NO_REGULARIZE', 'ACADOS_PROJECT', 'IPOPT'])]

        if df.empty:
            print("Error: DataFrame is empty.")
            return

        methods = df['Regularization'].unique()
        qp_names = df['QP_name'].unique()

        # --- 2. Styles (Shared to ensure consistency) ---
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
        marker_cycle = itertools.cycle(markers)
        color_cycle = itertools.cycle(plt.cm.tab10.colors)

        style_map = {}
        for name in qp_names:
            style_map[name] = {'marker': next(marker_cycle), 'color': next(color_cycle)}

        # ==============================================================================
        # PART B: Iteration Performance Profile (Data Profile) - Status Based
        # ==============================================================================
        print("Generating Iteration Performance Profile (Status == 0)...")

        if 'Iteration' in df.columns and 'Status' in df.columns:
            # 1. 准备数据
            iter_df = df.copy()
            metric = 'Iteration'

            # 确保 status 是数值类型，防止字符串 '0' 导致过滤失败
            if iter_df['Status'].dtype == object:
                iter_df['Status'] = pd.to_numeric(iter_df['Status'], errors='coerce')

            iter_methods = [m for m in methods if m in iter_df['Regularization'].unique()]
            total_problems = iter_df['QP_name'].nunique()

            # 2. 颜色映射
            method_palette = sns.color_palette("tab10", n_colors=len(iter_methods))
            method_colors = dict(zip(iter_methods, method_palette))

            # 3. 计算统一的 X 轴右边界 (Right Wall)
            # 我们只看 status == 0 (成功) 的数据
            success_data = iter_df[iter_df['Status'] == 0]

            if not success_data.empty:
                # 让图表右边界比最大的成功迭代次数稍微大一点 (例如 1.05 倍)
                # 这样可以看清最后一条线是否还有上升
                max_iter_val = success_data[metric].max()
                max_x_limit = max_iter_val * 1.05
            else:
                max_x_limit = 100 # 默认 fallback

            plt.figure(figsize=(10, 7))

            # 4. 绘制每个方法的阶梯图
            for method in iter_methods:
                method_data = iter_df[iter_df['Regularization'] == method]

                # --- [核心修改] 只筛选 status == 0 的数据 ---
                valid_mask = (method_data['Status'] == 0)
                solved_iters = method_data[valid_mask][metric].values

                # 排序 (从小到大)
                solved_iters.sort()

                # --- 构建延伸数据 (Extend Forward) ---
                # 目的：让线条画到最右边，而不是断在最后一次成功的地方

                # X轴: [0, t1, t2, ..., tn, max_x_limit]
                x_plot = np.concatenate(([0], solved_iters, [max_x_limit]))

                # Y轴: [0, 0,  1,   ..., n-1, n] -> 对应 where='post'
                current_solved_count = len(solved_iters)
                # 注意：最后一个点 (max_x_limit) 的 Y 值必须等于 current_solved_count
                # 表示从最后一次解决问题后，解决数量保持不变直到图表尽头
                y_plot = np.concatenate(([0], np.arange(1, current_solved_count + 1), [current_solved_count]))

                # 绘制阶梯图
                plt.step(x_plot, y_plot, where='post', label=method,
                         color=method_colors[method], linewidth=2.5, alpha=0.85)

                # (可选) 在真实的最后一个成功点画圈，区分“延伸线”和“真实数据”
                if len(solved_iters) > 0:
                    plt.plot(solved_iters[-1], current_solved_count, marker='o',
                             color=method_colors[method], markersize=5)

            # 5. 辅助线与装饰
            # 总问题数虚线
            plt.axhline(total_problems, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.text(0, total_problems, f' Total Problems: {total_problems}',
                     va='bottom', ha='left', fontsize=9, color='gray')

            plt.xlabel("Iteration Count", fontsize=12)
            plt.ylabel("Solved Problems (Count)", fontsize=12)
            plt.title("Performance Profile", fontsize=14, fontweight='bold')

            # X轴范围
            plt.xlim(0, max_x_limit)
            plt.ylim(0, total_problems * 1.05)

            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(title="Method", loc="lower right", frameon=True, fontsize=10)

            plt.tight_layout()
            plt.savefig("iteration_performance_profile.png", bbox_inches='tight')

        else:
            print("Skipping Iteration Chart: 'Iteration' or 'Status' column missing.")

        print("Displaying all windows...")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def plot_anaysis_charts(filename='qp_results.csv'):
    """
    Hybrid Analysis Chart:
    1. M/P Metrics: Side-by-Side (1x2) with Independent Y-axis.
    2. Iteration: Restored to ORIGINAL Baseline (Single large plot).
    """
    try:
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()

        df = df[~df['Regularization'].isin(['NO_REGULARIZE', 'ACADOS_PROJECT', 'IPOPT'])]

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

        OVERFLOW_THRESHOLD = 1e16

        # ==============================================================================
        # PART A: M & P Side-by-Side Logic (Corrected for Min/Max Analysis)
        # ==============================================================================
        def _draw_mp_subplot(ax, data, col_name, method_order, use_log, show_legend=False):
            chart_df = data[~data['Regularization'].isin(['PROJECT', 'MIRROR'])].copy()
            current_methods = [m for m in method_order if m in chart_df['Regularization'].unique()]

            valid_mask = chart_df[col_name].abs() < OVERFLOW_THRESHOLD
            valid_data = chart_df[valid_mask]

            sns.boxplot(
                data=valid_data, x='Regularization', y=col_name, order=current_methods,
                color='lightgray', showfliers=False, ax=ax,
                boxprops=dict(alpha=0.3), width=0.5
            )

            if not valid_data.empty:
                max_val = valid_data[col_name].max()
                min_val = valid_data[col_name].min()
                data_range = max_val - min_val
                if data_range == 0: data_range = abs(max_val) * 0.1 if max_val != 0 else 1.0

                clamp_top = max_val + data_range * 0.2
                clamp_bottom = min_val - data_range * 0.2

                if use_log:
                    clamp_top = max_val * 10 if max_val > 0 else max_val * 0.1
                    clamp_bottom = min_val * 10 if min_val < 0 else min_val * 0.1

                    if pd.isna(clamp_top): clamp_top = 1e5
                    if pd.isna(clamp_bottom): clamp_bottom = -1e5
            else:
                clamp_top = 1e5
                clamp_bottom = -1e5

            for name in qp_names:
                subset = chart_df[(chart_df['QP_name'] == name) & (chart_df['Regularization'].isin(current_methods))]
                if subset.empty: continue
                style = style_map[name]

                sub_val = subset[subset[col_name].abs() < OVERFLOW_THRESHOLD]
                if not sub_val.empty:
                    ax.plot(sub_val['Regularization'], sub_val[col_name], linestyle='None',
                            marker=style['marker'], color=style['color'], label=name,
                            markersize=8, alpha=0.85, markeredgecolor='white', markeredgewidth=0.5)

                sub_over_pos = subset[subset[col_name] >= OVERFLOW_THRESHOLD]
                if not sub_over_pos.empty:
                    ax.plot(sub_over_pos['Regularization'], [clamp_top]*len(sub_over_pos), linestyle='None',
                            marker='X', color='red', markersize=9, alpha=1.0,
                            markeredgecolor='black', markeredgewidth=0.5)

                sub_over_neg = subset[subset[col_name] <= -OVERFLOW_THRESHOLD]
                if not sub_over_neg.empty:
                    ax.plot(sub_over_neg['Regularization'], [clamp_bottom]*len(sub_over_neg), linestyle='None',
                            marker='X', color='red', markersize=9, alpha=1.0,
                            markeredgecolor='black', markeredgewidth=0.5)

            if use_log:
                ax.set_yscale('symlog', linthresh=1.0)

            has_pos_overflow = (chart_df[col_name].max() >= OVERFLOW_THRESHOLD)
            has_neg_overflow = (chart_df[col_name].min() <= -OVERFLOW_THRESHOLD)

            current_ylim = ax.get_ylim()
            new_top = clamp_top * (2 if use_log else 1.1) if has_pos_overflow else current_ylim[1]
            new_bottom = clamp_bottom * (2 if use_log else 1.1) if has_neg_overflow else current_ylim[0]

            ax.set_ylim(bottom=new_bottom, top=new_top)

            if has_pos_overflow:
                ax.axhline(clamp_top, color='red', linestyle=':', alpha=0.4)
            if has_neg_overflow:
                ax.axhline(clamp_bottom, color='red', linestyle=':', alpha=0.4)

        mp_configs = [
            ("Reduction", 
             "Max M Condition Number Reduction", 
             "Max P Condition Number Reduction"),

            ("Adjustment", 
             "Max M Adjustment", 
             "Max P Adjustment"),
        ]

        print("Generating Side-by-Side M/P Charts...")
        for main_title, col_m, col_p in mp_configs:
            if col_m not in df.columns or col_p not in df.columns:
                print(f"Skipping {main_title}: Columns '{col_m}' or '{col_p}' not found in CSV.")
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            _draw_mp_subplot(axes[0], df, col_m, methods, use_log=True, show_legend=False)
            axes[0].set_title("Reduced Hessian M", fontsize=10, fontweight='bold')

            _draw_mp_subplot(axes[1], df, col_p, methods, use_log=True, show_legend=False)
            axes[1].set_title("Matrix P", fontsize=10, fontweight='bold')

            axes[0].set_ylabel("Adjustment Value" if "Adjustment" in main_title else "Reduction Value", fontsize=10)
            axes[0].set_xlabel("")
            axes[0].grid(True, which="both", ls="--", alpha=0.3)
            axes[0].tick_params(axis='x', labelsize=8, rotation=45)

            axes[1].set_ylabel("Adjustment Value" if "Adjustment" in main_title else "Reduction Value", fontsize=10)
            axes[1].set_xlabel("")
            axes[1].grid(True, which="both", ls="--", alpha=0.3)
            axes[1].tick_params(axis='x', labelsize=8, rotation=45)

            plt.tight_layout()
            plt.savefig(f"{main_title.replace(' ', '_').lower()}.png", bbox_inches='tight')

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    plt.show()

def plot_cost(filename='qp_results.csv'):
    from matplotlib.lines import Line2D
    """
    Plots the 'Cost' metric categorized by QP_name patterns.
    Differentiates 'ACADOS_PROJECT' and 'IPOPT' truth values using distinct line styles:
    - ACADOS_PROJECT: Dashed line (--)
    - IPOPT: Dotted line (:)
    Each reference line maintains the color of its corresponding QP case.
    """
    try:
        # --- 1. Load and Clean Data ---
        df = pd.read_csv(filename, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()

        # Define truth solvers and their specific line styles
        truth_config = {
            'ACADOS_PROJECT': {'ls': '--', 'label': 'ACADOS Truth'},
            'IPOPT': {'ls': ':', 'label': 'IPOPT Truth'}
        }

        # Get unique regularization methods for X-axis (excluding truth solvers and baseline)
        all_methods = [m for m in df['Regularization'].unique()
                      if m not in truth_config.keys() and
                      m in ["PROJECT", "MIRROR", "PM_STAGE_MIRROR", "RP_STAGE_MIRROR"] ]

        # --- 2. Define Category Patterns ---
        categories = {
            'M_pos': 'M_pos',
            'M_inde_R_inde_P_pos': 'M_inde_R_inde_P_pos',
            'M_inde_R_pos_P_inde': 'M_inde_R_pos_P_inde',
            'M_inde_R_inde_P_inde': 'M_inde_R_inde_P_inde',
            'Pendulum Cases': 'Pendulum'
        }

        # --- 3. Plotting Loop ---
        for title, keyword in categories.items():
            cat_df = df[df['QP_name'].str.contains(keyword, case=False, na=False)]

            if cat_df.empty:
                continue

            case_names = cat_df['QP_name'].unique()
            plt.figure(figsize=(12, 7), num=f"Cost Analysis: {title}")

            marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'])
            color_cycle = itertools.cycle(plt.cm.tab10.colors)

            for case in case_names:
                case_color = next(color_cycle)
                case_marker = next(marker_cycle)

                # --- Plot Regularization Methods (Scatter + Line) ---
                case_subset = cat_df[cat_df['QP_name'] == case]
                plot_data = case_subset[case_subset['Regularization'].isin(all_methods)]
                plot_data = plot_data.set_index('Regularization').reindex(all_methods)

                x_vals = all_methods
                y_vals = plot_data['Cost'].values
                mask = ~np.isnan(y_vals.astype(float))

                plt.plot(np.array(x_vals)[mask], y_vals[mask],
                         linestyle='None',
                         marker=case_marker, color=case_color,
                         label=case, markersize=8, alpha=0.8)

                # # --- Plot Truth Reference Lines (Separated Styles) ---
                # for solver, config in truth_config.items():
                #     truth_val = case_subset[case_subset['Regularization'] == solver]['Cost']
                #     if not truth_val.empty:
                #         val = truth_val.values[0]
                #         # Plot horizontal line with specific style (dashed or dotted)
                #         plt.axhline(y=val, color=case_color, linestyle=config['ls'],
                #                     alpha=0.6, linewidth=1.2)

            # # --- 4. Plot Aesthetics & Legend ---
            # plt.title(f"Cost Comparison with Separated Truths: {title}", fontsize=14)
            # plt.ylabel("Cost Value")
            # plt.xlabel("Regularization Method")
            # plt.grid(True, which="both", ls="--", alpha=0.4)
            # plt.xticks(rotation=45, ha="right")

            # # Create custom legend entries for the two truth styles
            # handles, labels = plt.gca().get_legend_handles_labels()

            # # Add proxy artists for the truth line styles
            # acados_proxy = Line2D([0], [0], color='gray', linestyle='--', lw=1.2)
            # ipopt_proxy = Line2D([0], [0], color='gray', linestyle=':', lw=1.2)

            # handles.extend([acados_proxy, ipopt_proxy])
            # labels.extend(['ACADOS_PROJECT Truth (--)', 'IPOPT Truth (:)'])

            # plt.legend(handles=handles, labels=labels, title="Cases & Truth References",
            #            bbox_to_anchor=(1.02, 1), loc='upper left')

            # # --- 4. Plot Aesthetics & Legend ---
            plt.title(f"Cost Comparison: {title}", fontsize=14)
            plt.ylabel("Cost Value")
            plt.xlabel("Regularization Method")
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Cases", bbox_to_anchor=(1.02, 1), loc='upper left')

            plt.tight_layout()
            print(f"Figure generated for Category: {title}")
            plt.savefig(f"cost_comparison_{keyword.lower()}.png", bbox_inches='tight')

        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_reg_counter(filename='qp_results.csv'):
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']):
        df[col] = df[col].str.strip()

    targets = [m for m in df['Regularization'].unique() if m != 'NO_REGULARIZE']
    df_plot = df[(df['Status'] == 0) & (df['Regularization'].isin(targets))]

    sns.set_theme(style="ticks", font_scale=1.1)
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data=df_plot, x='Regularization', y='Regularization Count', hue='Regularization',
        order=targets, palette='deep', edgecolor='black', linewidth=1.2,
        capsize=0.1, errorbar='se', width=0.5, dodge=False, legend=False
    )

    ax.set_title('Average Regularization Count (Success Cases)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Average Count ± SE', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels([t.replace('_', ' ') for t in targets], fontsize=10)
    ax.tick_params(direction='in', length=4, width=1)
    ax.tick_params(axis='x', rotation=45)
    sns.despine()

    plt.tight_layout()
    plt.savefig('regularization_avg_se.png', dpi=300, bbox_inches='tight')
    plt.show()

# Entry point
if __name__ == "__main__":
    plot_charts()
    # plot_cost()
    plot_anaysis_charts()
    plot_reg_counter()