import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
import ast
from collections import defaultdict
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

class FirthLogisticRegressionCustom:

    def __init__(self, max_iter=1000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.se_ = None
        self.p_values_ = None
        self.conf_int_ = None

    def fit(self, X, y):
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])
        from sklearn.linear_model import LogisticRegression as LR
        lr = LR(penalty='l2', C=1e6, solver='lbfgs', max_iter=5000)
        lr.fit(X, y)
        init = np.concatenate([[lr.intercept_[0]], lr.coef_[0]])
        
        def neg_penalized_loglik(beta):
            eta = X_design @ beta
            mu = expit(eta)
            loglik = np.sum(y * np.log(mu + 1e-10) + (1 - y) * np.log(1 - mu + 1e-10))
            W = mu * (1 - mu)
            W = np.clip(W, 1e-10, 1 - 1e-10)
            Fisher = (X_design.T * W) @ X_design
            sign, logdet = np.linalg.slogdet(Fisher)
            if sign <= 0:
                return -loglik - (-1e10)
            penalty = 0.5 * logdet
            return -(loglik + penalty)
        
        res = minimize(neg_penalized_loglik, init, method='L-BFGS-B',
                       options={'maxiter': self.max_iter, 'gtol': self.tol})
        if not res.success:
            print(f"Firth optimization warning: {res.message}")
        beta_hat = res.x
        self.intercept_ = beta_hat[0]
        self.coef_ = beta_hat[1:]
        
        eta = X_design @ beta_hat
        mu = expit(eta)
        W = np.clip(mu * (1 - mu), 1e-10, 1 - 1e-10)
        def penalized_loglik(beta):
            eta = X_design @ beta
            mu = expit(eta)
            loglik = np.sum(y * np.log(mu + 1e-10) + (1 - y) * np.log(1 - mu + 1e-10))
            W_local = np.clip(mu * (1 - mu), 1e-10, 1 - 1e-10)
            Fisher = (X_design.T * W_local) @ X_design
            sign, logdet = np.linalg.slogdet(Fisher)
            if sign <= 0:
                return -1e10
            penalty = 0.5 * logdet
            return loglik + penalty
        
        Fisher_mat = (X_design.T * W) @ X_design
        try:
            cov_matrix = np.linalg.inv(Fisher_mat)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(Fisher_mat)
        se = np.sqrt(np.diag(cov_matrix))
        self.se_ = se[1:]
        self.intercept_se_ = se[0]
        
        z_scores = beta_hat / se
        self.p_values_ = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        self.intercept_p_ = 2 * (1 - stats.norm.cdf(np.abs(z_scores[0])))
        self.p_values_ = self.p_values_[1:]
        
        alpha = 0.05
        z_alpha = stats.norm.ppf(1 - alpha/2)
        self.conf_int_ = np.column_stack([beta_hat - z_alpha * se, beta_hat + z_alpha * se])
        self.feature_conf_int_ = self.conf_int_[1:, :]
        self.intercept_conf_int_ = self.conf_int_[0, :].reshape(1, -1)
        
        return self

    def predict_proba(self, X):
        eta = np.dot(X, self.coef_) + self.intercept_
        prob = expit(eta)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

CLASSIFIERS = ["random_forest", "xgboost", "logistic_regression", "mlp", "knn"]
RANDOM_SEED = 42
N_SPLITS = 10

DIMENSIONS = [
    "Problem Understanding and Comprehension",
    "Solution Design and Approach",
    "Implementation and Verification Support",
    "Project Environment and Knowledge Foundation"
]
DIM_SHORT = ['PU', 'SD', 'IV', 'PE']

def hosmer_lemeshow(y, y_pred_proba, g=10):
    data = pd.DataFrame({'y': y, 'prob': y_pred_proba})
    data['decile'] = pd.qcut(data['prob'], g, duplicates='drop')
    observed = data.groupby('decile')['y'].sum()
    expected = data.groupby('decile')['prob'].sum()
    n = data.groupby('decile').size()
    hl_stat = np.sum((observed - expected) ** 2 / (expected * (1 - expected / n)))
    df = len(observed) - 2
    p_val = 1 - stats.chi2.cdf(hl_stat, df)
    return hl_stat, p_val

def load_and_merge_data(success_path, fail_path):
    df_success = pd.read_excel(success_path)
    df_fail = pd.read_excel(fail_path)
    df_success['success'] = 1
    df_fail['success'] = 0
    df = pd.concat([df_success, df_fail], ignore_index=True)
    print(f"Successful GFI count: {len(df_success)}")
    print(f"Failed GFI count: {len(df_fail)}")
    return df

def parse_dimensions(dim_str):
    if pd.isna(dim_str) or dim_str == "" or dim_str == "[]":
        return []
    try:
        if isinstance(dim_str, str):
            return ast.literal_eval(dim_str)
        elif isinstance(dim_str, list):
            return dim_str
        else:
            return []
    except:
        clean = dim_str.strip("[]").replace("'", "").replace('"', '')
        return [d.strip() for d in clean.split(",") if d.strip()]

def extract_dimension_features(df):
    X_dim = np.zeros((len(df), len(DIMENSIONS)))
    for i, dim_str in enumerate(df['data_dimensions']):
        dims = parse_dimensions(dim_str)
        for j, dim in enumerate(DIMENSIONS):
            if dim in dims:
                X_dim[i, j] = 1
    dim_feat_names = [f"dim_{short}" for short in DIM_SHORT]
    return X_dim, dim_feat_names

def extract_label_features(df):
    label_cols = [
        'issue_type_classification',
        'priority_classification',
        'difficulty_classification',
        'impact_area_classification',
        'theme_classification',
        'status_classification',
        'technology_classification'
    ]
    unique_categories = defaultdict(set)
    for col in label_cols:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            if isinstance(val, str):
                cats = [c.strip() for c in val.split(',') if c.strip()]
            elif isinstance(val, list):
                cats = val
            else:
                cats = []
            for cat in cats:
                unique_categories[col].add(cat)
    
    feature_names = []
    for col, cats in unique_categories.items():
        for cat in cats:
            feature_names.append(f"{col}::{cat}")
    
    X_label = np.zeros((len(df), len(feature_names)))
    for i, row in df.iterrows():
        feat_idx = 0
        for col, cats in unique_categories.items():
            val = row.get(col, '')
            if pd.isna(val):
                val = ''
            if isinstance(val, str):
                row_cats = [c.strip() for c in val.split(',') if c.strip()]
            elif isinstance(val, list):
                row_cats = val
            else:
                row_cats = []
            for cat in cats:
                if cat in row_cats:
                    X_label[i, feat_idx] = 1
                feat_idx += 1
    return X_label, feature_names

def prepare_all_features(df):
    X_dim, dim_names = extract_dimension_features(df)
    X_label, label_names = extract_label_features(df)
    X = np.hstack([X_dim, X_label])
    feature_names = dim_names + label_names
    print(f"Total features: {len(feature_names)}")
    y = df['success'].values
    return X, y, feature_names, dim_names, label_names

def create_classifier(name):
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=RANDOM_SEED, eval_metric='logloss',
                                 verbosity=0, use_label_encoder=False)
    elif name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                             early_stopping=True, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier: {name}")

def evaluate_model(X, y, classifier_name, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = create_classifier(classifier_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        if y_proba is not None:
            metrics['auc'].append(roc_auc_score(y_test, y_proba))
    results = {}
    for k, v in metrics.items():
        if v:
            results[k] = (np.mean(v), np.std(v))
    return results

def analyze_feature_importance(X, y, feature_names, classifier_name="random_forest"):
    model = create_classifier(classifier_name)
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        return None
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return imp_df.sort_values('importance', ascending=False)

def univariate_analysis(df, dim_names, dim_short, label_names):
    results = []
    for i, dname in enumerate(DIMENSIONS):
        col = f"presence_{dim_short[i]}"
        presence = np.zeros(len(df))
        for idx, dim_str in enumerate(df['data_dimensions']):
            if dname in parse_dimensions(dim_str):
                presence[idx] = 1
        df[col] = presence
    
    group_success = df[df['success'] == 1]
    group_fail = df[df['success'] == 0]
    
    for i, dname in enumerate(DIMENSIONS):
        col = f"presence_{dim_short[i]}"
        table = pd.crosstab(df[col], df['success'])
        if table.shape == (2,2):
            chi2, p, dof, ex = stats.chi2_contingency(table)
        else:
            chi2, p = np.nan, np.nan
        p_succ = group_success[col].mean()
        p_fail = group_fail[col].mean()
        diff = p_succ - p_fail
        results.append({
            'feature': col,
            'full_name': dname,
            'type': 'dimension',
            'success_prop': p_succ,
            'fail_prop': p_fail,
            'prop_diff': diff,
            'chi2': chi2,
            'p_value': p
        })
    
    X_label, label_names_list = extract_label_features(df)
    for i, name in enumerate(label_names_list):
        presence = X_label[:, i]
        table = pd.crosstab(presence, df['success'])
        if table.shape == (2,2):
            chi2, p, dof, ex = stats.chi2_contingency(table)
        else:
            chi2, p = np.nan, np.nan
        p_succ = presence[df['success'] == 1].mean()
        p_fail = presence[df['success'] == 0].mean()
        results.append({
            'feature': name,
            'full_name': name,
            'type': 'label',
            'success_prop': p_succ,
            'fail_prop': p_fail,
            'prop_diff': p_succ - p_fail,
            'chi2': chi2,
            'p_value': p
        })
    
    df_results = pd.DataFrame(results)
    pvals = df_results['p_value'].dropna().values
    if len(pvals) > 0:
        _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
        df_results.loc[df_results['p_value'].notna(), 'q_value'] = qvals
    else:
        df_results['q_value'] = np.nan
    return df_results

def logistic_regression_analysis(X, y, feature_names, dim_names):
    X_sm = sm.add_constant(X)
    
    null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
    result = None
    try:
        model = sm.Logit(y, X_sm)
        result = model.fit(disp=0, method='bfgs', maxiter=500)
        print("Standard logistic regression fitted successfully.")
    except Exception as e:
        print(f"statsmodels fitting failed: {e}")
    
    print("\n===== Separation Diagnosis (Dimension Predictors) =====")
    sep_warnings = []
    for j, dname in enumerate(dim_names):
        idx = feature_names.index(dname)
        col_name = f"dim_{DIM_SHORT[j]}"
        tab = pd.crosstab(X[:, idx], y.astype(int))
        print(f"\n{dname} vs success:")
        print(tab)
        min_cell = tab.min().min()
        if min_cell < 5:
            msg = f"{dname}: Minimum cell count = {min_cell} (<5), possible quasi-separation."
            sep_warnings.append(msg)
            print(f"  *** {msg}")
        if tab.shape[0] == 2 and tab.shape[1] == 2:
            if (tab.iloc[0,0] == 0 and tab.iloc[1,1] == 0) or \
               (tab.iloc[0,1] == 0 and tab.iloc[1,0] == 0):
                msg = f"{dname}: Perfect prediction (separation)!"
                sep_warnings.append(msg)
                print(f"  *** {msg}")
    if not sep_warnings:
        print("\nNo obvious separation/quasi-separation issues detected.")
    else:
        print("\n*** Summary: Possible separation/quasi-separation, standard regression OR and p-values may be unreliable. ***")
    
    print("\n===== Custom Firth Logistic Regression =====")
    firth = FirthLogisticRegressionCustom()
    firth.fit(X, y)
    firth_coef = np.insert(firth.coef_, 0, firth.intercept_)
    firth_se = np.insert(firth.se_, 0, firth.intercept_se_)
    firth_pvals = np.insert(firth.p_values_, 0, firth.intercept_p_)
    firth_ci = np.row_stack([firth.intercept_conf_int_, firth.feature_conf_int_])
    
    firth_coef_df = pd.DataFrame({
        'feature': ['const'] + feature_names,
        'coef_firth': firth_coef,
        'odds_ratio_firth': np.exp(firth_coef),
        'ci_lower_firth': np.exp(firth_ci[:, 0]),
        'ci_upper_firth': np.exp(firth_ci[:, 1]),
        'p_value_firth': firth_pvals
    })
    firth_coef_df.loc[0, ['odds_ratio_firth', 'ci_lower_firth', 'ci_upper_firth']] = np.nan
    print("Firth regression complete.")
    
    coef_df = None
    fit_stats = None
    if result is not None:
        params = np.array(result.params)
        bse = np.array(result.bse)
        pvalues = np.array(result.pvalues)
        conf_int = result.conf_int()
        if hasattr(conf_int, 'iloc'):
            ci_lower = np.array(conf_int.iloc[:, 0])
            ci_upper = np.array(conf_int.iloc[:, 1])
        else:
            ci_lower = conf_int[:, 0]
            ci_upper = conf_int[:, 1]
        
        pvals_no_const = pvalues[1:]
        _, qvals_no_const, _, _ = multipletests(pvals_no_const, method='fdr_bh')
        qvals = np.insert(qvals_no_const, 0, np.nan)
        
        coef_df = pd.DataFrame({
            'feature': ['const'] + feature_names,
            'coef': params,
            'std_err': bse,
            'p_value': pvalues,
            'q_value': qvals,
            'odds_ratio': np.exp(params),
            'ci_lower': np.exp(ci_lower),
            'ci_upper': np.exp(ci_upper)
        })
        coef_df.loc[0, ['odds_ratio', 'ci_lower', 'ci_upper']] = np.nan
        
        ll_full = result.llf
        ll_null = null_model.llf
        mcfadden_r2 = 1 - ll_full / ll_null
        y_pred_proba = result.predict()
        hl_stat, hl_p = hosmer_lemeshow(y, y_pred_proba)
        
        fit_stats = {
            'aic_null': null_model.aic,
            'aic_full': result.aic,
            'mcfadden_r2': mcfadden_r2,
            'hl_stat': hl_stat,
            'hl_p': hl_p
        }
    else:
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X, y)
        coef_df = pd.DataFrame({
            'feature': ['const'] + feature_names,
            'coef': [lr.intercept_[0]] + list(lr.coef_[0]),
            'odds_ratio': np.exp([lr.intercept_[0]] + list(lr.coef_[0]))
        })
        coef_df[['std_err', 'p_value', 'q_value', 'ci_lower', 'ci_upper']] = np.nan
        coef_df.loc[0, ['odds_ratio', 'ci_lower', 'ci_upper']] = np.nan
        fit_stats = {}
    
    return coef_df, fit_stats, firth_coef_df, sep_warnings

def print_univariate_table(df_res, top_n=20):
    df_sig = df_res[df_res['p_value'] < 0.05].sort_values('p_value')
    if len(df_sig) == 0:
        print("\nNo significant features in univariate tests.")
        return
    print("\n% LaTeX table for Univariate Analysis (significant features)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Univariate comparison ...}")
    print("\\label{tab:rq3-univariate}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Feature} & \\textbf{Success} & \\textbf{Failure} & \\textbf{Diff} & $\\chi^2$ & \\textbf{p} & \\textbf{q} \\\\")
    print("\\midrule")
    for _, row in df_sig.head(top_n).iterrows():
        feat = row['feature'].replace('_', '\\_').replace('::', ': ')
        print(f"{feat} & {row['success_prop']:.3f} & {row['fail_prop']:.3f} & {row['prop_diff']:.3f} & {row['chi2']:.2f} & {row['p_value']:.3e} & {row['q_value']:.3e} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def print_logistic_table(coef_df, fit_stats, firth_coef_df=None, feature_names_subset=None):
    if feature_names_subset is None:
        dim_feats = [f"dim_{s}" for s in DIM_SHORT]
        label_feats = coef_df[coef_df['feature'].str.contains('::', na=False) & (coef_df['q_value'] < 0.05)]['feature'].tolist()
        subset = ['const'] + dim_feats + label_feats[:10]
    else:
        subset = feature_names_subset
    df_sub = coef_df[coef_df['feature'].isin(subset)].copy()
    df_sub['signif'] = ''
    mask_sig = (df_sub['q_value'] < 0.05) & (~df_sub['feature'].str.contains('const'))
    df_sub.loc[mask_sig, 'signif'] = '$^*$'
    
    print("\n% LaTeX table for Logistic Regression (Standard)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Multivariate logistic regression predicting GFI success. Features with $^*$ remain significant after FDR correction.}")
    print("\\label{tab:rq3-logistic}")
    print("\\begin{tabular}{lccccr}")
    print("\\toprule")
    print("\\textbf{Feature} & \\textbf{Odds Ratio} & \\textbf{95\\% CI} & \\textbf{p-value} & \\textbf{q-value} \\\\")
    print("\\midrule")
    for _, row in df_sub.iterrows():
        feat = row['feature'].replace('_', '\\_').replace('::', ': ')
        if feat == 'const':
            continue
        or_val = f"{row['odds_ratio']:.3f}" if not pd.isna(row['odds_ratio']) else "NA"
        ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if not pd.isna(row['ci_lower']) else "NA"
        p = f"{row['p_value']:.3e}" if not pd.isna(row['p_value']) and row['p_value'] < 0.001 else f"{row['p_value']:.3f}" if not pd.isna(row['p_value']) else "NA"
        q = f"{row['q_value']:.3e}" if not pd.isna(row['q_value']) else "NA"
        sig_mark = row['signif']
        print(f"{feat}{sig_mark} & {or_val} & {ci} & {p} & {q} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    if fit_stats:
        print("\n\\textbf{Model fit statistics:}")
        print(f"Null AIC: {fit_stats['aic_null']:.2f}")
        print(f"Full AIC: {fit_stats['aic_full']:.2f}")
        print(f"McFadden Pseudo $R^2$: {fit_stats['mcfadden_r2']:.3f}")
        print(f"Hosmer‑Lemeshow $\\chi^2$: {fit_stats['hl_stat']:.2f}, p = {fit_stats['hl_p']:.3f}")
    print("\\end{table}")
    
    if firth_coef_df is not None:
        print("\n% LaTeX table for Firth Logistic Regression")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Firth logistic regression (penalized) – dimensions only.}")
        print("\\label{tab:rq3-firth}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("\\textbf{Feature} & \\textbf{Odds Ratio} & \\textbf{95\\% CI} & \\textbf{p-value} \\\\")
        print("\\midrule")
        for _, row in firth_coef_df.iterrows():
            feat = row['feature'].replace('_', '\\_').replace('::', ': ')
            if feat == 'const':
                continue
            or_val = f"{row['odds_ratio_firth']:.3f}"
            ci = f"[{row['ci_lower_firth']:.3f}, {row['ci_upper_firth']:.3f}]"
            p = f"{row['p_value_firth']:.3e}" if not pd.isna(row['p_value_firth']) else "NA"
            print(f"{feat} & {or_val} & {ci} & {p} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
    
    non_sig_after_fdr = coef_df[(coef_df['p_value'] < 0.05) & (coef_df['q_value'] >= 0.05) & (~coef_df['feature'].str.contains('const'))]
    if len(non_sig_after_fdr) > 0:
        print("\n\\textbf{Note:} The following variables had raw $p<0.05$ but were not significant after FDR correction:")
        for _, row in non_sig_after_fdr.iterrows():
            feat = row['feature'].replace('_', '\\_').replace('::', ': ')
            print(f"{feat} (p={row['p_value']:.3f}, q={row['q_value']:.3f})")
        print()

def print_latex_results(all_results):
    print("\n% LaTeX table for RQ3: Classifier Performance")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Predictive performance ...}")
    print("\\label{tab:rq3-classifier-performance}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Classifier} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} \\\\")
    print("\\midrule")
    name_map = {"random_forest": "Random Forest", "xgboost": "XGBoost", "logistic_regression": "Logistic Regression", "mlp": "MLP", "knn": "KNN"}
    for clf in CLASSIFIERS:
        if clf not in all_results:
            continue
        res = all_results[clf]
        row = name_map.get(clf, clf)
        row += f" & {res['accuracy'][0]:.3f}$\\pm${res['accuracy'][1]:.3f}"
        row += f" & {res['precision'][0]:.3f}$\\pm${res['precision'][1]:.3f}"
        row += f" & {res['recall'][0]:.3f}$\\pm${res['recall'][1]:.3f}"
        row += f" & {res['f1'][0]:.3f}$\\pm${res['f1'][1]:.3f}"
        if 'auc' in res:
            row += f" & {res['auc'][0]:.3f}$\\pm${res['auc'][1]:.3f}"
        else:
            row += " & --"
        row += " \\\\"
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def print_feature_importance_table(imp_df, top_n=15):
    print("\n% LaTeX table for Feature Importance")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Top 15 important features (Random Forest).}")
    print("\\label{tab:rq3-feature-importance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Feature} & \\textbf{Importance} \\\\")
    print("\\midrule")
    for _, row in imp_df.head(top_n).iterrows():
        feat = row['feature'].replace('_', '\\_').replace('::', ': ')
        print(f"{feat} & {row['importance']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    df = load_and_merge_data("gfi_issues_classified.xlsx", "rq3/gfi_fail_data.xlsx")
    
    X, y, feature_names, dim_names, label_names = prepare_all_features(df)
    print(f"Feature matrix shape: {X.shape}, Positive ratio: {y.mean():.3f}")
    
    print("\n" + "="*80)
    print("Univariate statistical test (Chi-square test)")
    print("="*80)
    df_uni = univariate_analysis(df, dim_names, DIM_SHORT, label_names)
    dim_res = df_uni[df_uni['type'] == 'dimension']
    print("\nDimension differences:")
    print(dim_res[['full_name', 'success_prop', 'fail_prop', 'prop_diff', 'chi2', 'p_value', 'q_value']].to_string(index=False))
    print_univariate_table(df_uni)
    
    print("\n" + "="*80)
    print("Multivariate logistic regression and separation diagnosis")
    print("="*80)
    coef_df, fit_stats, firth_coef_df, sep_warnings = logistic_regression_analysis(
        X, y, feature_names, dim_names
    )
    
    if coef_df is not None:
        dim_coef = coef_df[coef_df['feature'].isin(dim_names)]
        print("\nDimension standard logistic regression results:")
        print(dim_coef[['feature', 'odds_ratio', 'ci_lower', 'ci_upper', 'p_value', 'q_value']].to_string(index=False))
    if firth_coef_df is not None:
        dim_firth = firth_coef_df[firth_coef_df['feature'].isin(dim_names)]
        print("\nDimension Firth logistic regression results:")
        print(dim_firth[['feature', 'odds_ratio_firth', 'ci_lower_firth', 'ci_upper_firth', 'p_value_firth']].to_string(index=False))
    
    print_logistic_table(coef_df, fit_stats, firth_coef_df)
    
    print("\n" + "="*80)
    print("Classifier predictive performance")
    print("="*80)
    np.random.seed(RANDOM_SEED)
    shuffle_idx = np.random.permutation(len(X))
    X_shuf, y_shuf = X[shuffle_idx], y[shuffle_idx]
    all_results = {}
    for clf in CLASSIFIERS:
        print(f"Evaluating {clf}...", end=' ', flush=True)
        res = evaluate_model(X_shuf, y_shuf, clf)
        all_results[clf] = res
        print(f"F1={res['f1'][0]:.3f}, AUC={res.get('auc', [np.nan])[0]:.3f}")
    print_latex_results(all_results)
    
    print("\n" + "="*80)
    print("Feature importance analysis (Random Forest)")
    print("="*80)
    imp_df = analyze_feature_importance(X, y, feature_names, "random_forest")
    if imp_df is not None:
        print(imp_df.head(15).to_string(index=False))
        print_feature_importance_table(imp_df)
    
    print("\nAll analyses completed!")
    if sep_warnings:
        print("\n⚠️ Note: Separation/quasi-separation warnings exist, standard regression OR and p-values may be unreliable. Please refer to Firth regression results.")

if __name__ == "__main__":
    main()