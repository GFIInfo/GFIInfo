import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast
import warnings
import textwrap
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, rankdata, norm
import itertools

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

TARGET_CATEGORIES = [
    'Technical Implementation Guidance',
    'Testing and Verification',
    'Expected Behavior and Scope',
    'Context and Root Cause Analysis',
    'Problem Reproduction and Validation'
]

def build_level3_to_level2():
    classification_tree = {
        "Problem Understanding and Comprehension": {
            "Problem Reproduction and Validation": [
                "Visual Evidence",
                "Bug Reproduction Steps and Guidance",
                "Environment-Specific Reproduction",
                "Error Reproduction Examples",
                "Status Confirmation",
                "Problem Diagnosis"
            ],
            "Context and Root Cause Analysis": [
                "Root Cause Analysis",
                "Problem Context and Impact",
                "Implementation Context",
                "Behavioral Context",
                "Historical Context",
                "Domain Terminology Guidance",
                "Behavioral Evidence"
            ],
            "Expected Behavior and Scope": [
                "Problem Specification",
                "Behavior Specification",
                "Scope Definition"
            ]
        },
        "Solution Design and Approach": {
            "Solution Strategy Guidance": [
                "Solution Approach",
                "Architectural Guidance",
                "Problem-Solving Strategy",
                "Alternative Solution Evaluation",
                "Implementation Strategy",
                "Solution Goal Clarification"
            ],
            "Implementation Reference and Planning": [
                "Design Consistency Guidance",
                "Implementation Techniques and Patterns",
                "Implementation Plan",
                "Alternative Solution Options",
                "Prior Work References",
                "Workaround Demonstration",
                "Implementation Examples and References"
            ],
            "Solution Validation and Approval": [
                "Solution Feasibility Validation",
                "Validation Criteria"
            ]
        },
        "Implementation and Verification Support": {
            "Technical Implementation Guidance": [
                "Code Location Guidance",
                "Solution Correction",
                "Exact Code Fix Suggestion",
                "Task Clarification",
                "Implementation Guidance"
            ],
            "Testing and Verification": [
                "Testing Configuration",
                "Verification Requirements",
                "Test Implementation Guidance",
                "Test Case Construction",
                "Testing Procedure",
                "Reproducible Test Cases",
                "Validation Tool References",
                "Solution Verification Methods",
                "Error Identification",
                "Prevention Guidance"
            ]
        },
        "Project Environment and Knowledge Foundation": {
            "Quality and Standards": [
                "Code Quality and Standards",
                "Contribution Requirements",
                "Quality Assurance Requirements",
                "Coding Standards",
                "Technical Conventions"
            ],
            "Documentation and References": [
                "Technical Documentation References",
                "External Resource Links",
                "Project Overview and Process References",
                "Documentation Standards"
            ],
            "Workflow and Process Support": [
                "Contribution Process Guidance",
                "Task Management and Prioritization",
                "Workflow Best Practice",
                "Task Assignment and Collaboration"
            ],
            "Environment and Tool Setup": [
                "Environment Configuration",
                "Setup Instructions",
                "Tool Usage Guidance"
            ]
        }
    }

    l3_to_l2 = {}
    for l2_dict in classification_tree.values():
        for l2_name, l3_list in l2_dict.items():
            for l3_name in l3_list:
                l3_to_l2[l3_name] = l2_name
    return l3_to_l2

# --------------------------------------------------------------------

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
    print(f"Data loaded successfully, {len(df)} rows")

    column_mapping = {
        'issue_type_classification': 'task_type',
        'technology_classification': 'domain'
    }
    existing_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
    if existing_mapping:
        df.rename(columns=existing_mapping, inplace=True)
        print("Renamed columns:", existing_mapping)

    level3_to_level2 = build_level3_to_level2()

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

    for col in df.columns:
        df[col] = df[col].apply(parse_item)

    if 'row_data' not in df.columns:
        raise ValueError("Column 'row_data' not found in the Excel file.")

    def map_secondary(categories):
        return [level3_to_level2[cat] for cat in categories if cat in level3_to_level2]

    df['data_categories'] = df['row_data'].apply(map_secondary)

    def filter_target(categories):
        return [cat for cat in categories if cat in TARGET_CATEGORIES]
    df['data_categories'] = df['data_categories'].apply(filter_target)

    for cat in TARGET_CATEGORIES:
        df[f'has_{cat}'] = df['data_categories'].apply(lambda lst: 1 if cat in lst else 0)
        df[f'count_{cat}'] = df['data_categories'].apply(lambda lst: lst.count(cat))

    print(f"Preprocessing completed, {len(df)} rows remaining")
    return df

def add_presence_tests_with_posthoc(df):
    attributes = ['task_type', 'domain']
    global_results = []
    posthoc_results = []

    for attr in attributes:
        if attr not in df.columns:
            continue
        temp_df = df[df[attr].apply(lambda x: len(x) > 0)]
        exploded = temp_df.explode(attr)

        for cat in TARGET_CATEGORIES:
            col = f'has_{cat}'
            table = pd.crosstab(exploded[attr], exploded[col])
            if table.shape[0] < 2:
                continue
            chi2, p, dof, expected = chi2_contingency(table)
            n = table.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * (min(table.shape)-1)))
            global_results.append({
                'Attribute': attr,
                'Category': cat,
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
                    posthoc_df['Category'] = cat
                    posthoc_results.append(posthoc_df)
    if posthoc_results:
        posthoc_df_all = pd.concat(posthoc_results, ignore_index=True)
    else:
        posthoc_df_all = pd.DataFrame()
    return pd.DataFrame(global_results), posthoc_df_all

def add_frequency_tests_with_posthoc(df):
    attributes = ['task_type', 'domain']
    global_results = []
    posthoc_results = []

    for attr in attributes:
        if attr not in df.columns:
            continue
        temp_df = df[df[attr].apply(lambda x: len(x) > 0)]
        exploded = temp_df.explode(attr)

        for cat in TARGET_CATEGORIES:
            count_col = f'count_{cat}'
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
                'Category': cat,
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
                        posthoc_df['Category'] = cat
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
                    posthoc_df['Category'] = cat
                    posthoc_results.append(posthoc_df)
    if posthoc_results:
        posthoc_df_all = pd.concat(posthoc_results, ignore_index=True)
        corrected_list = []
        for (attr, cat), group in posthoc_df_all.groupby(['Attribute','Category']):
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
    print("SUBCATEGORY STATISTICAL RESULTS (Bonferroni corrected)")
    print("="*80)
    if not df_pres.empty:
        print("\n--- Presence (Chi-square) ---")
        sig = df_pres[df_pres['p_corrected'] < 0.05]
        if not sig.empty:
            for _, row in sig.iterrows():
                print(f"  {row['Attribute']} vs {row['Category']}: χ²={row['Statistic']:.2f}, "
                      f"p_corr={row['p_corrected']:.4f}, V={row['Effect_size']:.3f}")
        else:
            print("  No significant presence associations.")
    if not df_freq.empty:
        print("\n--- Frequency (Kruskal-Wallis / Mann-Whitney U) ---")
        sig = df_freq[df_freq['p_corrected'] < 0.05]
        if not sig.empty:
            for _, row in sig.iterrows():
                print(f"  {row['Attribute']} vs {row['Category']}: {row['Test']} H/U={row['Statistic']:.2f}, "
                      f"p_corr={row['p_corrected']:.4f}")
        else:
            print("  No significant frequency differences.")

def wrap_labels(labels, max_length=20):
    wrapped_labels = []
    for label in labels:
        if len(str(label)) > max_length:
            wrapped = textwrap.fill(str(label), width=max_length)
            wrapped_labels.append(wrapped)
        else:
            wrapped_labels.append(str(label))
    return wrapped_labels

def create_category_distribution_sub(df):
    all_cats = []
    for cats in df['data_categories']:
        all_cats.extend(cats)
    counts = [all_cats.count(cat) for cat in TARGET_CATEGORIES]
    plt.figure(figsize=(14,8))
    y_pos = np.arange(len(TARGET_CATEGORIES))
    bars = plt.barh(y_pos, counts, color=sns.color_palette("viridis", len(TARGET_CATEGORIES)))
    plt.grid(True, axis='both', color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    for bar, cnt in zip(bars, counts):
        plt.text(cnt - max(counts)*0.01, bar.get_y()+bar.get_height()/2, f'{cnt}',
                 ha='right', va='center', fontsize=20)
    plt.xlim(0, max(counts)*1.05)
    wrapped = wrap_labels(TARGET_CATEGORIES, max_length=20)
    plt.yticks(y_pos, wrapped, fontsize=20)
    plt.xlabel('Count', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def create_correlation_analysis_sub(df, stats_pres_df=None):
    all_cats = []
    for cats in df['data_categories']:
        all_cats.extend(cats)
    present_cats = [cat for cat in TARGET_CATEGORIES if cat in all_cats]
    colors = sns.color_palette("husl", len(present_cats))

    attributes = ['task_type', 'domain']
    fig, axes = plt.subplots(1, 2, figsize=(18,6))
    plt.subplots_adjust(bottom=0.2, wspace=0.3)

    for i, attr in enumerate(attributes):
        if attr not in df.columns:
            axes[i].set_visible(False)
            continue
        exploded = df.explode(attr)
        exploded = exploded[exploded[attr].notna() & (exploded[attr] != '')]
        attr_counts = exploded[attr].value_counts()
        top_vals = attr_counts.head(5).index.tolist()

        freq_by_cat = {}
        for cat in present_cats:
            freqs = []
            for val in top_vals:
                subset = exploded[exploded[attr] == val]
                if len(subset) > 0:
                    mean_cnt = subset['data_categories'].apply(lambda lst: lst.count(cat)).mean()
                    freqs.append(mean_cnt)
                else:
                    freqs.append(0)
            freq_by_cat[cat] = freqs

        x = np.arange(len(top_vals))
        bottom = np.zeros(len(top_vals))
        for j, cat in enumerate(present_cats):
            values = freq_by_cat[cat]
            bars = axes[i].bar(x, values, label=cat, bottom=bottom, color=colors[j])
            for k, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.05:
                    y_pos = bottom[k] + val/2
                    axes[i].text(bar.get_x()+bar.get_width()/2, y_pos, f'{val:.2f}',
                                 ha='center', va='center', fontsize=12, color='white')
            bottom += np.array(values)

        axes[i].set_ylim(0, 3)
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
        axes[i].tick_params(axis='both', labelsize=12)

        if stats_pres_df is not None and not stats_pres_df.empty:
            attr_stats = stats_pres_df[stats_pres_df['Attribute'] == attr]
            if not attr_stats.empty:
                p_texts = []
                for _, row in attr_stats.iterrows():
                    cat_short = row['Category'].replace('Technical Implementation Guidance','TIG') \
                                 .replace('Testing and Verification','TV') \
                                 .replace('Expected Behavior and Scope','EBS') \
                                 .replace('Context and Root Cause Analysis','CRC') \
                                 .replace('Problem Reproduction and Validation','PRV')
                    p_val = row['p_corrected']
                    p_str = f"{cat_short}: p<0.001" if p_val < 0.001 else f"{cat_short}: p={p_val:.3f}"
                    p_texts.append(p_str)
                axes[i].text(0.5, 1.02, '; '.join(p_texts), transform=axes[i].transAxes,
                             ha='center', fontsize=9, style='italic', color='gray')

    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(present_cats))]
    fig.legend(handles, present_cats, loc='lower center',
               bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=12, frameon=False)
    plt.show()

def create_dimension_count_distribution(df):
    if 'data_dimensions' not in df.columns:
        print("data_dimensions column not found, skipping dimension count distribution chart")
        return
    dim_counts = [len(set(dims)) for dims in df['data_dimensions']]
    count_dist = Counter(dim_counts)
    x_vals = [1,2,3,4]
    y_vals = [count_dist.get(i,0) for i in x_vals]
    plt.figure(figsize=(8,6))
    bars = plt.bar(x_vals, y_vals, color=sns.color_palette("Blues_d",4))
    plt.xlabel('Number of Information Dimensions per GFI', fontsize=14)
    plt.ylabel('Number of GFIs', fontsize=14)
    plt.xticks(x_vals)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, cnt in zip(bars, y_vals):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f'{cnt}',
                 ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    df = load_and_preprocess_data()
    if len(df) == 0:
        print("No valid data")
        return

    df_pres_global, df_pres_posthoc = add_presence_tests_with_posthoc(df)
    df_pres_global = apply_bonferroni(df_pres_global)

    df_freq_global, df_freq_posthoc = add_frequency_tests_with_posthoc(df)
    df_freq_global = apply_bonferroni(df_freq_global)

    df_pres_global.to_csv('sub_presence_global.csv', index=False)
    if not df_pres_posthoc.empty:
        df_pres_posthoc.to_csv('sub_presence_posthoc.csv', index=False)
    df_freq_global.to_csv('sub_frequency_global.csv', index=False)
    if not df_freq_posthoc.empty:
        df_freq_posthoc.to_csv('sub_frequency_posthoc.csv', index=False)

    print_stat_results(df_pres_global, df_freq_global)
    if not df_pres_posthoc.empty:
        print("\n--- Subcategory Presence Post-hoc (significant pairs) ---")
        sig = df_pres_posthoc[(df_pres_posthoc['p_corrected'] < 0.05) & (df_pres_posthoc['cramer_v'] >= 0.1)]
        if not sig.empty:
            print(sig[['Attribute','Category','Group1','Group2','p_corrected','cramer_v']].to_string())
    if not df_freq_posthoc.empty:
        print("\n--- Subcategory Frequency Post-hoc (significant pairs) ---")
        sig = df_freq_posthoc[(df_freq_posthoc['p_corrected'] < 0.05) & (df_freq_posthoc['cliffs_delta'].abs() >= 0.147)]
        if not sig.empty:
            print(sig[['Attribute','Category','Group1','Group2','p_corrected','cliffs_delta']].to_string())

    create_category_distribution_sub(df)
    create_correlation_analysis_sub(df, df_pres_global)

if __name__ == "__main__":
    main()