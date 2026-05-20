import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import ast
import warnings
import textwrap
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, rankdata, norm
import math
import itertools

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def pairwise_chi2_with_bonferroni(exploded_df, attr_col, target_col):
    results = []
    values = exploded_df[attr_col].unique()
    for v1, v2 in itertools.combinations(values, 2):
        sub = exploded_df[exploded_df[attr_col].isin([v1, v2])]
        table = pd.crosstab(sub[attr_col], sub[target_col])
        if table.shape[1] < 2:   
            continue
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        if (expected < 5).sum() / expected.size > 0.2:
            _, p = fisher_exact(table, alternative='two-sided')
            chi2 = np.nan
            dof = 1
        n = table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(table.shape)-1))) if not np.isnan(chi2) else np.nan
        results.append({
            'Group1': v1,
            'Group2': v2,
            'p_uncorrected': p,
            'cramer_v': cramer_v,
            'n': n
        })
    if results:
        df_res = pd.DataFrame(results)
        n_comp = len(df_res)
        df_res['p_corrected'] = df_res['p_uncorrected'] * n_comp
        df_res['p_corrected'] = df_res['p_corrected'].clip(upper=1)
        return df_res
    else:
        return pd.DataFrame()

def dunn_test_with_cliffs_delta(groups, group_names):
    n_groups = len(groups)
    all_data = np.concatenate(groups)
    all_ranks = rankdata(all_data)
    n = [len(g) for g in groups]
    rank_sums = []
    idx = 0
    for g in groups:
        rank_sums.append(np.sum(all_ranks[idx:idx+len(g)]))
        idx += len(g)
    
    results = []
    for i, j in itertools.combinations(range(n_groups), 2):
        z_num = (rank_sums[i]/n[i] - rank_sums[j]/n[j])
        total_n = sum(n)
        var = (total_n * (total_n + 1) / 12) * (1/n[i] + 1/n[j])
        z = z_num / np.sqrt(var)
        p = 2 * (1 - norm.cdf(abs(z)))
        # Cliff's delta
        x = groups[i]
        y = groups[j]
        n_x = len(x)
        n_y = len(y)
        gt = sum(1 for xi in x for yj in y if xi > yj)
        lt = sum(1 for xi in x for yj in y if xi < yj)
        delta = (gt - lt) / (n_x * n_y)
        results.append({
            'Group1': group_names[i],
            'Group2': group_names[j],
            'z': z,
            'p_uncorrected': p,
            'cliffs_delta': delta
        })
    if results:
        df_res = pd.DataFrame(results)
        n_comp = len(df_res)
        df_res['p_corrected'] = df_res['p_uncorrected'] * n_comp
        df_res['p_corrected'] = df_res['p_corrected'].clip(upper=1)
        return df_res
    else:
        return pd.DataFrame()

def load_and_preprocess_data():
    df = pd.read_excel('gfi_issues_classified.xlsx')

    column_mapping = {
        'issue_type_classification': 'task_type',
        'priority_classification': 'priority',
        'difficulty_classification': 'difficulty',
        'technology_classification': 'domain'
    }
    existing_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
    if existing_mapping:
        df.rename(columns=existing_mapping, inplace=True)

    classification_tree = {
        "Problem Understanding and Comprehension": {
            "Problem Reproduction and Validation": [
                "Visual Evidence", "Bug Reproduction Steps and Guidance",
                "Environment-Specific Reproduction", "Error Reproduction Examples",
                "Status Confirmation", "Problem Diagnosis"
            ],
            "Context and Root Cause Analysis": [
                "Root Cause Analysis", "Problem Context and Impact",
                "Implementation Context", "Behavioral Context",
                "Historical Context", "Domain Terminology Guidance",
                "Behavioral Evidence"
            ],
            "Expected Behavior and Scope": [
                "Problem Specification", "Behavior Specification", "Scope Definition"
            ]
        },
        "Solution Design and Approach": {
            "Solution Strategy Guidance": [
                "Solution Approach", "Architectural Guidance",
                "Problem-Solving Strategy", "Alternative Solution Evaluation",
                "Implementation Strategy", "Solution Goal Clarification"
            ],
            "Implementation Reference and Planning": [
                "Design Consistency Guidance", "Implementation Techniques and Patterns",
                "Implementation Plan", "Alternative Solution Options",
                "Prior Work References", "Workaround Demonstration",
                "Implementation Examples and References"
            ],
            "Solution Validation and Approval": [
                "Solution Feasibility Validation", "Validation Criteria"
            ]
        },
        "Implementation and Verification Support": {
            "Technical Implementation Guidance": [
                "Code Location Guidance", "Solution Correction",
                "Exact Code Fix Suggestion", "Task Clarification",
                "Implementation Guidance"
            ],
            "Testing and Verification": [
                "Testing Configuration", "Verification Requirements",
                "Test Implementation Guidance", "Test Case Construction",
                "Testing Procedure", "Reproducible Test Cases",
                "Validation Tool References", "Solution Verification Methods",
                "Error Identification", "Prevention Guidance"
            ]
        },
        "Project Environment and Knowledge Foundation": {
            "Quality and Standards": [
                "Code Quality and Standards", "Contribution Requirements",
                "Quality Assurance Requirements", "Coding Standards",
                "Technical Conventions"
            ],
            "Documentation and References": [
                "Technical Documentation References", "External Resource Links",
                "Project Overview and Process References", "Documentation Standards"
            ],
            "Workflow and Process Support": [
                "Contribution Process Guidance", "Task Management and Prioritization",
                "Workflow Best Practice", "Task Assignment and Collaboration"
            ],
            "Environment and Tool Setup": [
                "Environment Configuration", "Setup Instructions",
                "Tool Usage Guidance"
            ]
        }
    }

    level3_to_level1 = {}
    for l1, l2dict in classification_tree.items():
        for l2, l3list in l2dict.items():
            for l3 in l3list:
                level3_to_level1[l3] = l1

    def parse_item(item):
        if pd.isna(item) or item == '':
            return []
        if isinstance(item, list):
            return item
        item_str = str(item).strip()
        if item_str.startswith('[') and item_str.endswith(']'):
            try:
                parsed = ast.literal_eval(item_str)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except:
                pass
        if ',' in item_str:
            return [elem.strip() for elem in item_str.split(',') if elem.strip() != '']
        return [item_str]

    cols_to_parse = [c for c in df.columns if c not in ['row_data', 'data_dimensions']]
    for col in cols_to_parse:
        df[col] = df[col].apply(parse_item)

    def map_row_to_dimensions(val):
        if pd.isna(val) or val == '':
            return []
        if isinstance(val, list):
            items = val
        else:
            s = str(val).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    items = ast.literal_eval(s)
                    if not isinstance(items, list):
                        items = [s]
                except:
                    items = [s]
            else:
                items = [s]
        dims = []
        for item in items:
            item_str = str(item).strip()
            if item_str in level3_to_level1:
                dims.append(level3_to_level1[item_str])
        return dims

    if 'row_data' in df.columns:
        df['data_dimensions'] = df['row_data'].apply(map_row_to_dimensions)
    else:
        raise ValueError("miss 'row_data'")

    def parse_row(val):
        if pd.isna(val) or val == '':
            return []
        if isinstance(val, list):
            items = val
        else:
            s = str(val).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    items = ast.literal_eval(s)
                    if not isinstance(items, list):
                        items = [s]
                except:
                    items = [s]
            else:
                items = [s]
        return [str(item).strip() for item in items]

    df['row_data_parsed'] = df['row_data'].apply(parse_row)
    # -------------------------------------------------------------

    if 'priority' in df.columns:
        priority_mapping = {
            'Medium Priority': 'Medium/High Priority',
            'High Priority': 'Medium/High Priority',
            'Critical/High Priority': 'Medium/High Priority'
        }
        df['priority'] = df['priority'].apply(
            lambda lst: list(set([priority_mapping.get(item, item) for item in lst]))
        )
    if 'difficulty' in df.columns:
        difficulty_mapping = {
            'Medium Complexity': 'Medium/High Complexity',
            'Advanced/Complex': 'Medium/High Complexity',
            'Complex': 'Medium/High Complexity'
        }
        df['difficulty'] = df['difficulty'].apply(
            lambda lst: list(set([difficulty_mapping.get(item, item) for item in lst]))
        )

    dimension_names = [
        'Problem Understanding and Comprehension',
        'Implementation and Verification Support',
        'Solution Design and Approach',
        'Project Environment and Knowledge Foundation'
    ]
    for dim in dimension_names:
        df[f'has_{dim}'] = df['data_dimensions'].apply(lambda lst: 1 if dim in lst else 0)
        df[f'count_{dim}'] = df['data_dimensions'].apply(lambda lst: lst.count(dim))

    return df, dimension_names, classification_tree

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def print_taxonomy_statistics(df, classification_tree):
    level1_map = {}
    level2_map = {}
    for l1, l2dict in classification_tree.items():
        for l2, l3list in l2dict.items():
            for l3 in l3list:
                level1_map[l3] = l1
                level2_map[l3] = l2

    level3_counts = Counter()
    for items in df['row_data_parsed']:
        for item in items:
            if item in level1_map:   
                level3_counts[item] += 1

    level2_counts = Counter()
    level1_counts = Counter()
    for l3, cnt in level3_counts.items():
        l2 = level2_map[l3]
        l1 = level1_map[l3]
        level2_counts[l2] += cnt
        level1_counts[l1] += cnt

    print("\n" + "=" * 70)
    print("TAXONOMY INSTANCE COUNTS (across loaded dataset)")
    print("=" * 70)

    dimension_order = [
        "Implementation and Verification Support",
        "Problem Understanding and Comprehension",
        "Solution Design and Approach",
        "Project Environment and Knowledge Foundation"
    ]

    for l1 in dimension_order:
        if l1 not in classification_tree:
            continue
        total_l1 = level1_counts[l1]
        print(f"\n{l1}: {total_l1} instances")
        for l2, l3list in classification_tree[l1].items():
            if not l3list:
                continue
            total_l2 = level2_counts[l2]
            pct_l2 = (total_l2 / total_l1 * 100) if total_l1 > 0 else 0
            print(f"  {l2}: {total_l2} instances ({pct_l2:.1f}%)")
            for l3 in l3list:
                cnt_l3 = level3_counts.get(l3, 0)
                pct_l3 = (cnt_l3 / total_l1 * 100) if total_l1 > 0 else 0
                print(f"    {l3} ({cnt_l3}, {pct_l3:.1f}%)")
    print("\n")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def add_presence_tests_with_posthoc(df, dimension_names):
    attributes = ['task_type', 'priority', 'difficulty', 'domain']
    global_results = []
    posthoc_results = []

    for attr in attributes:
        if attr not in df.columns:
            continue
        temp_df = df[df[attr].apply(lambda x: len(x) > 0)]
        exploded = temp_df.explode(attr)

        for dim in dimension_names:
            col = f'has_{dim}'
            table = pd.crosstab(exploded[attr], exploded[col])
            if table.shape[0] < 2:
                continue
            chi2, p, dof, expected = chi2_contingency(table)
            n = table.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * (min(table.shape)-1)))
            global_results.append({
                'Attribute': attr,
                'Dimension': dim,
                'Test': 'Presence (Chi-square)',
                'Statistic': chi2,
                'df': dof,
                'p_value': p,
                'Effect_size': cramer_v,
                'n': n
            })
            if p < 0.05:
                posthoc_df = pairwise_chi2_with_bonferroni(exploded, attr, col)
                if not posthoc_df.empty:
                    posthoc_df['Attribute'] = attr
                    posthoc_df['Dimension'] = dim
                    posthoc_results.append(posthoc_df)
    if posthoc_results:
        posthoc_df_all = pd.concat(posthoc_results, ignore_index=True)
    else:
        posthoc_df_all = pd.DataFrame()
    return pd.DataFrame(global_results), posthoc_df_all

def add_frequency_tests_with_posthoc(df, dimension_names):
    attributes = ['task_type', 'priority', 'difficulty', 'domain']
    global_results = []
    posthoc_results = []

    for attr in attributes:
        if attr not in df.columns:
            continue
        temp_df = df[df[attr].apply(lambda x: len(x) > 0)]
        exploded = temp_df.explode(attr)

        for dim in dimension_names:
            count_col = f'count_{dim}'
            groups = []
            group_names = []
            for val in exploded[attr].unique():
                subset = exploded[exploded[attr] == val]
                if len(subset) > 0:
                    groups.append(subset[count_col].values)
                    group_names.append(val)
            if len(groups) < 2:
                continue
            if len(groups) == 2:
                stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                test_name = 'Mann-Whitney U'
                dof = 1
            else:
                stat, p = kruskal(*groups)
                test_name = 'Kruskal-Wallis'
                dof = len(groups)-1
            global_results.append({
                'Attribute': attr,
                'Dimension': dim,
                'Test': f'Frequency ({test_name})',
                'Statistic': stat,
                'df': dof,
                'p_value': p,
                'n': len(exploded)
            })
            if p < 0.05:
                if len(groups) > 2:
                    posthoc_df = dunn_test_with_cliffs_delta(groups, group_names)
                    if not posthoc_df.empty:
                        posthoc_df['Attribute'] = attr
                        posthoc_df['Dimension'] = dim
                        posthoc_results.append(posthoc_df)
                elif len(groups) == 2:
                    x = groups[0]
                    y = groups[1]
                    gt = sum(1 for xi in x for yj in y if xi > yj)
                    lt = sum(1 for xi in x for yj in y if xi < yj)
                    delta = (gt - lt) / (len(x)*len(y))
                    posthoc_df = pd.DataFrame([{
                        'Group1': group_names[0],
                        'Group2': group_names[1],
                        'p_uncorrected': p,
                        'cliffs_delta': delta
                    }])
                    posthoc_df['Attribute'] = attr
                    posthoc_df['Dimension'] = dim
                    posthoc_results.append(posthoc_df)
    if posthoc_results:
        posthoc_df_all = pd.concat(posthoc_results, ignore_index=True)
        corrected_list = []
        for (attr, dim), group in posthoc_df_all.groupby(['Attribute','Dimension']):
            n_comp = len(group)
            group = group.copy()
            group['p_corrected'] = group['p_uncorrected'] * n_comp
            group['p_corrected'] = group['p_corrected'].clip(upper=1)
            corrected_list.append(group)
        posthoc_df_all = pd.concat(corrected_list, ignore_index=True)
    else:
        posthoc_df_all = pd.DataFrame()
    return pd.DataFrame(global_results), posthoc_df_all

def apply_bonferroni(df_results):
    if df_results.empty:
        return df_results
    n_tests = len(df_results)
    df_results['p_corrected'] = df_results['p_value'] * n_tests
    df_results['p_corrected'] = df_results['p_corrected'].clip(upper=1)
    return df_results

def print_stat_results(df_pres, df_freq):
    print("\n" + "="*80)
    print("STATISTICAL RESULTS (after Bonferroni correction)")
    print("="*80)
    if not df_pres.empty:
        print("\n--- Presence (Chi-square) ---")
        sig_pres = df_pres[df_pres['p_corrected'] < 0.05]
        if not sig_pres.empty:
            for _, row in sig_pres.iterrows():
                print(f"  {row['Attribute']} vs {row['Dimension']}: χ²={row['Statistic']:.2f}, "
                      f"p_corr={row['p_corrected']:.4f}, V={row['Effect_size']:.3f}")
        else:
            print("  No significant associations after correction.")
    if not df_freq.empty:
        print("\n--- Frequency (Kruskal-Wallis / Mann-Whitney U) ---")
        sig_freq = df_freq[df_freq['p_corrected'] < 0.05]
        if not sig_freq.empty:
            for _, row in sig_freq.iterrows():
                print(f"  {row['Attribute']} vs {row['Dimension']}: {row['Test']} H/U={row['Statistic']:.2f}, "
                      f"p_corr={row['p_corrected']:.4f}")
        else:
            print("  No significant frequency differences after correction.")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def wrap_labels(labels, max_length=20):
    wrapped_labels = []
    for label in labels:
        if len(str(label)) > max_length:
            wrapped = textwrap.fill(str(label), width=max_length)
            wrapped_labels.append(wrapped)
        else:
            wrapped_labels.append(str(label))
    return wrapped_labels

def create_category_distribution(df, dimension_names):
    all_categories = []
    for categories in df['data_dimensions']:
        all_categories.extend(categories)
    category_counts = Counter(all_categories)
    categories = [dim for dim in dimension_names if dim in category_counts]
    if len(categories) >= 2:
        categories[0], categories[1] = categories[1], categories[0]
    counts = [category_counts[dim] for dim in categories]

    plt.figure(figsize=(14, 8))
    y_pos = np.arange(len(categories))
    bars = plt.barh(y_pos, counts, color=sns.color_palette("viridis", len(categories)))
    plt.grid(True, axis='both', color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    for bar, count in zip(bars, counts):
        plt.text(count - max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{count}', ha='right', va='center', fontsize=20)
    plt.xlim(0, max(counts)*1.05)
    wrapped = wrap_labels(categories, max_length=20)
    plt.yticks(y_pos, wrapped, fontsize=20)
    plt.xlabel('Count', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return category_counts

def create_correlation_analysis(df, dimension_names, stats_pres_df=None):
    all_categories = []
    for categories in df['data_dimensions']:
        all_categories.extend(categories)
    common_categories = [dim for dim in dimension_names if dim in Counter(all_categories)]
    colors = sns.color_palette("husl", len(common_categories))

    dimensions = ['task_type', 'domain']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)

    for i, attr in enumerate(dimensions):
        if attr not in df.columns:
            axes[i].set_visible(False)
            continue
        exploded_df = df.explode(attr)
        exploded_df = exploded_df[exploded_df[attr].notna() & (exploded_df[attr] != '')]
        attr_counts = exploded_df[attr].value_counts()
        top_vals = attr_counts.head(5).index.tolist()

        freq_by_attr = {}
        for dim in common_categories:
            freqs = []
            for val in top_vals:
                subset = exploded_df[exploded_df[attr] == val]
                if len(subset) > 0:
                    cnt = subset['data_dimensions'].apply(lambda lst: lst.count(dim)).mean()
                    freqs.append(cnt)
                else:
                    freqs.append(0)
            freq_by_attr[dim] = freqs

        x = np.arange(len(top_vals))
        bottom = np.zeros(len(top_vals))
        for j, dim in enumerate(common_categories):
            values = freq_by_attr[dim]
            bars = axes[i].bar(x, values, label=dim, bottom=bottom, color=colors[j])
            for k, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.05:
                    y_pos = bottom[k] + val/2
                    axes[i].text(bar.get_x() + bar.get_width()/2, y_pos,
                                 f'{val:.2f}', ha='center', va='center', fontsize=10, color='white')
            bottom += np.array(values)

        axes[i].set_ylim(0, 4)
        axes[i].grid(True, axis='both', color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        for spine in axes[i].spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        axes[i].set_ylabel('Average occurrences per issue', fontsize=12)
        title = 'Issue Type' if attr == 'task_type' else 'Domain'
        axes[i].set_title(f'GFI Categorization by {title}', fontsize=12, fontweight='bold', y=1.05)
        x_labels = [f'{val} ({attr_counts[val]})' for val in top_vals]
        wrapped = wrap_labels(x_labels, max_length=14)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(wrapped, rotation=0, ha='center')
        axes[i].tick_params(axis='both', labelsize=10)

        if stats_pres_df is not None and not stats_pres_df.empty:
            attr_stats = stats_pres_df[stats_pres_df['Attribute'] == attr]
            if not attr_stats.empty:
                p_texts = []
                for _, row in attr_stats.iterrows():
                    dim_short = row['Dimension'].replace('Problem Understanding and Comprehension', 'PU') \
                                 .replace('Implementation and Verification Support', 'IV') \
                                 .replace('Solution Design and Approach', 'SD') \
                                 .replace('Project Environment and Knowledge Foundation', 'PE')
                    p_val = row['p_corrected']
                    p_str = f"{dim_short}: p<0.001" if p_val < 0.001 else f"{dim_short}: p={p_val:.3f}"
                    p_texts.append(p_str)
                axes[i].text(0.5, 1.02, '; '.join(p_texts), transform=axes[i].transAxes,
                             ha='center', fontsize=9, style='italic', color='gray')

    
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(common_categories))]
    fig.legend(handles, common_categories, loc='lower center',
               bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=10, frameon=False)
    
    plt.show()

def create_variability_analysis(df):
    dimension_sets = [set(dim_list) for dim_list in df['data_dimensions']]
    num_dims = [len(dims) for dims in dimension_sets]
    num_counts = Counter(num_dims)
    total = len(num_dims)          
    print("\n--- Dimension Count Distribution ---")
    for n in [0, 1, 2, 3, 4]:
        cnt = num_counts.get(n, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {n} dimension(s): {cnt} GFIs ({pct:.1f}%)")
    # -------------------------------------------------

    all_nums = [0, 1, 2, 3, 4]
    counts = [num_counts.get(n, 0) for n in all_nums]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(all_nums, counts, color=sns.color_palette("Blues_d", 5))
    plt.xlabel('Number of Information Dimensions per GFI', fontsize=14)
    plt.ylabel('Number of GFIs', fontsize=14)
    plt.xticks(all_nums)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{cnt}',
                 ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    df, dimension_names, classification_tree = load_and_preprocess_data()
    if len(df) == 0:
        return

    print_taxonomy_statistics(df, classification_tree)

    df_pres_global, df_pres_posthoc = add_presence_tests_with_posthoc(df, dimension_names)
    df_pres_global = apply_bonferroni(df_pres_global)

    df_freq_global, df_freq_posthoc = add_frequency_tests_with_posthoc(df, dimension_names)
    df_freq_global = apply_bonferroni(df_freq_global)

    df_pres_global.to_csv('presence_global.csv', index=False)
    if not df_pres_posthoc.empty:
        df_pres_posthoc.to_csv('presence_posthoc.csv', index=False)
    df_freq_global.to_csv('frequency_global.csv', index=False)
    if not df_freq_posthoc.empty:
        df_freq_posthoc.to_csv('frequency_posthoc.csv', index=False)

    print_stat_results(df_pres_global, df_freq_global)
    if not df_pres_posthoc.empty:
        print("\n--- Presence Post-hoc Comparisons (significant pairs, p_corrected < 0.05) ---")
        sig_post = df_pres_posthoc[(df_pres_posthoc['p_corrected'] < 0.05) & (df_pres_posthoc['cramer_v'] >= 0.1)]
        if not sig_post.empty:
            print(sig_post[['Attribute','Dimension','Group1','Group2','p_corrected','cramer_v']].to_string())
    if not df_freq_posthoc.empty:
        print("\n--- Frequency Post-hoc Comparisons (significant pairs, p_corrected < 0.05) ---")
        sig_post = df_freq_posthoc[(df_freq_posthoc['p_corrected'] < 0.05) & (df_freq_posthoc['cliffs_delta'].abs() >= 0.147)]
        if not sig_post.empty:
            print(sig_post[['Attribute','Dimension','Group1','Group2','p_corrected','cliffs_delta']].to_string())

    create_category_distribution(df, dimension_names)
    create_correlation_analysis(df, dimension_names, df_pres_global)
    create_variability_analysis(df)

if __name__ == "__main__":
    main()