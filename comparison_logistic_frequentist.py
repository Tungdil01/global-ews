import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
seed = 1234
np.random.seed(seed)

def set_plot_style():
    plt.rcParams.update({'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16,'axes.titlesize':18,'font.size':16,'lines.linewidth':1.5,'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'black','axes.linewidth':1.5})

def create_directories():
    os.makedirs('csv',exist_ok=True)
    os.makedirs('img',exist_ok=True)

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    parts = str(species_name).strip().split()
    if len(parts)>=2:
        return f"{parts[0]} {parts[1]}"
    return species_name

def categorize_bifurcation(row, threshold):
    if row['best_probability'] < threshold:
        return 'undetermined'
    if row['raw_tipping_point_type'] == 'no-bifurcation':
        return 'no-bifurcation'
    return 'bifurcation'

def classify_iucn_status(s):
    if s in ['LEAST_CONCERN']:
        return 'non-threatened'
    elif s in ['NEAR_THREATENED', 'VULNERABLE','ENDANGERED','CRITICALLY_ENDANGERED']:
        return 'threatened'
    elif s in ['EXTINCT', 'EXTINCT_IN_THE_WILD']:
        return 'excluded'
    else:
        return 'unknown'

def load_and_preprocess_bifurcation(threshold=0.7):
    df = pd.read_csv('csv/bifurcation_results_short.csv')
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['tipping_category'] = df.apply(lambda row: categorize_bifurcation(row, threshold), axis=1)
    valid = ['bifurcation','no-bifurcation','undetermined']
    df = df[df['tipping_category'].isin(valid)]
    print(f"[bifurcation] Applied probability threshold {threshold}:")
    print(f"  - Total species: {len(df)}")
    print(f"  - Distribution:")
    for cat in valid:
        print(f"    {cat}: {(df['tipping_category']==cat).sum()}")
    return df

def load_and_preprocess_iucn():
    df = pd.read_csv('csv/iucn_species.csv', dtype=str)
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['IUCN_Status_raw'] = df['iucn_status'].astype(str).fillna('').str.strip().str.upper()
    df['conservation_status'] = df['IUCN_Status_raw'].apply(classify_iucn_status)
    dedup = df[['species_short','conservation_status']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    dedup = dedup[dedup['conservation_status'] != 'excluded'].reset_index(drop=True)
    print(f"[iucn] Preprocessed {len(dedup)} species: threatened={(dedup['conservation_status']=='threatened').sum()}, non-threatened={(dedup['conservation_status']=='non-threatened').sum()}, unknown={(dedup['conservation_status']=='unknown').sum()}")
    return dedup

def merge_datasets(bifurcation_df, iucn_df):
    per_species = bifurcation_df[['species_short','tipping_category']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    merged = pd.merge(per_species, iucn_df[['species_short','conservation_status']], on='species_short', how='inner')
    print(f"Merged dataset: {len(merged)} species with conservation status.")
    return merged

def compute_dl_dev_agreement(df, threshold=0.7):
    df = df[df['transition_year'].notna()].copy()
    dev_present = df['early_warning'].notna()
    if dev_present.sum() == 0:
        print("No DEV results found. Skipping DL-DEV agreement.")
        return None
    df_sub = df[dev_present].copy()
    df_sub['dl_bif'] = ((df_sub['best_probability'] >= threshold) & (df_sub['raw_tipping_point_type'] != 'no-bifurcation')).astype(bool)
    df_sub['dev_bif'] = ((df_sub['early_warning'] == True) & (df_sub['bifurcation_type'] != 'no-bifurcation')).astype(bool)
    cross = pd.crosstab(df_sub['dl_bif'], df_sub['dev_bif'])
    print("\nDL vs DEV bifurcation classification (threshold={}):".format(threshold))
    print(cross)
    kappa = cohen_kappa_score(df_sub['dl_bif'], df_sub['dev_bif'])
    print(f"Cohen's kappa: {kappa:.3g}")
    chi2, p, dof, expected = stats.chi2_contingency(cross)
    print(f"Chi-square test: chi2={chi2:.3g}, p={p:.3g}, df={dof}")
    return df_sub

def plot_dl_dev_agreement(df_sub, threshold, outcome_name='conservation'):
    if df_sub is None or len(df_sub)==0:
        print("No data for DL-DEV agreement plot.")
        return
    cross = pd.crosstab(df_sub['dl_bif'], df_sub['dev_bif'])
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.imshow(cross.values, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Dev No-bif', 'Dev Bif'])
    ax.set_yticklabels(['DL No-bif', 'DL Bif'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cross.values[i,j]}", ha='center', va='center', color='black' if cross.values[i,j] < np.max(cross.values)/2 else 'white', fontsize=14)
    ax.set_xlabel('DEV classification')
    ax.set_ylabel('DL classification')
    plt.tight_layout()
    filename = f'img/dl_dev_agreement_{outcome_name}.pdf'
    print(f"Saving figure: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Done: {filename}")

def plot_dl_dev_roc(df_sub, outcome_name='conservation'):
    if df_sub is None or len(df_sub)==0:
        print("No data for ROC plot.")
        return
    y = df_sub['dev_bif'].astype(int).values
    if np.sum(y)==0 or np.sum(y)==len(y):
        print("No variation in DEV classification, cannot compute ROC.")
        return
    X = df_sub[['best_probability']].copy()
    X = sm.add_constant(X)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)
    coef = res.params['best_probability']
    pval = res.pvalues['best_probability']
    or_ = np.exp(coef)
    fpr, tpr, _ = roc_curve(y, df_sub['best_probability'].values)
    roc_auc = auc(fpr, tpr)
    print(f"Logistic regression: DEV_bif ~ DL_best_probability, OR={or_:.3g}, p={pval:.3g}, AUC={roc_auc:.3g}")
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = {:.2f})'.format(roc_auc))
    ax.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    plt.tight_layout()
    filename = f'img/dl_dev_roc_{outcome_name}.pdf'
    print(f"Saving figure: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Done: {filename}")

def abbreviate_category_name(col_name):
    mapping = {'bifurcation': 'bif.', 'no-bifurcation': 'no-bif.', 'undetermined': 'undet.', 'non-threatened': 'non-threat.', 'threatened': 'threat.', 'unknown': 'unknown', }
    if col_name in mapping:
        return mapping[col_name]
    return col_name

def create_odds_ratio_plot(odds_ratios_dict, tipping_col, method_name, ref_cat, outcome_name='conservation'):
    set_plot_style()
    level_map = {0: 'non-threatened', 1: 'threatened', 2: 'unknown'}
    for outcome_level, or_df in odds_ratios_dict.items():
        if outcome_level == 0:
            continue
        if outcome_level not in level_map:
            continue
        level_name = level_map[outcome_level]
        print(f"Processing outcome level {outcome_level} -> '{level_name}'")
        if or_df.empty:
            print(f"  No tipping rows for '{level_name}', skipping.")
            continue
        tipping_rows = or_df[or_df.index.str.contains(tipping_col + '_')]
        if tipping_rows.empty:
            print(f"  No tipping rows for '{level_name}', skipping.")
            continue
        cats = tipping_rows.index.str.replace(tipping_col + '_', '')
        cat_labels = [abbreviate_category_name(c) for c in cats]
        or_vals = tipping_rows['odds_ratio'].values
        ci_low = tipping_rows['ci_lower'].values
        ci_high = tipping_rows['ci_upper'].values
        pvals = tipping_rows['p_value'].values
        y_pos = np.arange(len(cats))
        fig, ax = plt.subplots(1,1,figsize=(5,max(3,0.4*len(cats)+2)))
        for i, (orv, low, high, p) in enumerate(zip(or_vals, ci_low, ci_high, pvals)):
            color = 'red' if p < 0.05 else 'gray'
            ax.plot(orv, i, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
            ax.hlines(i, low, high, color=color, linewidth=1.5)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cat_labels)
        ax.set_xlabel(f'Odds Ratio (vs. {abbreviate_category_name(ref_cat)})')
        finite_high = ci_high[np.isfinite(ci_high)]
        if len(finite_high) > 0:
            max_ci = max(1.5, np.max(finite_high) * 1.2)
        else:
            max_ci = 2.0
        ax.set_xlim(0, max_ci)
        ax.set_ylim(-0.5, len(cats)-0.5)
        plt.tight_layout()
        filename = f'img/odds_ratios_{method_name}_{outcome_name}_{level_name}.pdf'
        print(f"Saving figure: {filename}")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Done: {filename}")

def create_contrast_plot(contrast_res, method_name, outcome_name='conservation', lrt_p=None):
    set_plot_style()
    if 0 not in contrast_res:
        print("Threatened outcome not found in contrast results, skipping contrast plot.")
        return
    res = contrast_res[0]
    or_val = res['or']
    low = res['ci_lower']
    high = res['ci_upper']
    p = lrt_p if lrt_p is not None else res['p']
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    color = 'red' if p < 0.05 else 'gray'
    ax.plot(or_val, 0, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    if np.isfinite(low) and np.isfinite(high):
        ax.hlines(0, low, high, color=color, linewidth=1.5)
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(['Threatened'])
    ax.set_xlabel('Odds Ratio\n(bif. vs. no-bif.)')
    max_x = max(1.5, high*1.2) if np.isfinite(high) else 25
    ax.set_xlim(0, max_x)
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    filename = f'img/contrast_odds_ratios_{method_name}_{outcome_name}.pdf'
    print(f"Saving figure: {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Done: {filename}")

def _fit_mnlogit_constrained(model, R, q, start_params, disp=False):
    def obj(x):
        return -model.loglike(x)
    cons = {'type':'eq', 'fun': lambda x: R.dot(x) - q}
    res = opt.minimize(obj, start_params, method='SLSQP', constraints=cons, options={'disp': disp})
    class ConstrainedResult:
        pass
    cr = ConstrainedResult()
    cr.llf = -res.fun if res.success else -np.inf
    return cr

def _profile_loglike(val, model, full_ll, coef_idx, flat_order, start_params=None):
    R = np.zeros((1, len(flat_order)))
    R[0, coef_idx] = 1.0
    q = np.array([val])
    try:
        if start_params is None:
            unconstrained = model.fit(disp=False)
            start_params = unconstrained.params.values.ravel('F')
        res_constrained = _fit_mnlogit_constrained(model, R, q, start_params, disp=False)
        return res_constrained.llf
    except:
        return -np.inf

def _profile_func(val, model, full_ll, coef_idx, flat_order, chi2_crit, start_params=None):
    ll = _profile_loglike(val, model, full_ll, coef_idx, flat_order, start_params)
    if not np.isfinite(ll):
        return 1.0
    return -2*(ll - full_ll) - chi2_crit

def get_profile_ci_p(model_result, var_name, outcome_level, alpha=0.05):
    model = model_result.model
    full_ll = model_result.llf
    params_full = model_result.params
    flat_order = params_full.values.ravel('F')
    idx_map = {}
    k_vars = params_full.shape[0]
    k_outcomes = params_full.shape[1]
    for j, out in enumerate(params_full.columns):
        for i, var in enumerate(params_full.index):
            idx_map[(var, out)] = j * k_vars + i
    if (var_name, outcome_level) not in idx_map:
        raise ValueError(f"Coefficient ({var_name}, {outcome_level}) not found.")
    coef_idx = idx_map[(var_name, outcome_level)]
    chi2_crit = stats.chi2.ppf(1-alpha, df=1)
    start_params_flat = model_result.params.values.ravel('F')
    coef_est = params_full.loc[var_name, outcome_level]
    step = 0.1
    found_low = False
    for i in range(1, 10000):
        val = coef_est - step * i
        f_val = _profile_func(val, model, full_ll, coef_idx, flat_order, chi2_crit, start_params_flat)
        if np.isfinite(f_val) and f_val > 0:
            low_bracket = (coef_est, val)
            found_low = True
            break
    if not found_low:
        raise RuntimeError("Could not find lower CI bound.")
    lower = opt.brentq(lambda v: _profile_func(v, model, full_ll, coef_idx, flat_order, chi2_crit, start_params_flat), low_bracket[0], low_bracket[1])
    found_high = False
    for i in range(1, 10000):
        val = coef_est + step * i
        f_val = _profile_func(val, model, full_ll, coef_idx, flat_order, chi2_crit, start_params_flat)
        if np.isfinite(f_val) and f_val > 0:
            upper_bracket = (coef_est, val)
            found_high = True
            break
    if not found_high:
        raise RuntimeError("Could not find upper CI bound.")
    upper = opt.brentq(lambda v: _profile_func(v, model, full_ll, coef_idx, flat_order, chi2_crit, start_params_flat), upper_bracket[0], upper_bracket[1])
    ll0 = _profile_loglike(0, model, full_ll, coef_idx, flat_order, start_params_flat)
    if np.isfinite(ll0):
        lr_stat = -2 * (ll0 - full_ll)
        p = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        p = np.nan
    return lower, upper, p

class full_statistical_analysis:
    def __init__(self, df):
        self.df = df
    def _prepare_data(self, include_undetermined):
        df = self.df.copy()
        if not include_undetermined:
            df = df[df['tipping_category'] != 'undetermined'].copy()
        df['y'] = df['conservation_status'].map({'non-threatened':0, 'threatened':1, 'unknown':2})
        if not include_undetermined:
            df['bif_binary'] = (df['tipping_category'] == 'bifurcation').astype(int)
        if include_undetermined:
            X = pd.get_dummies(df['tipping_category'], drop_first=True)
        else:
            X = df[['bif_binary']].copy()
            X.rename(columns={'bif_binary': 'tipping_category_bifurcation'}, inplace=True)
        X = sm.add_constant(X)
        y = df['y']
        return X, y, df
    def run_multinomial_lr(self, include_undetermined=True):
        print("Method: Multinomial unpenalized ML + Likelihood Ratio tests. Reference: Hosmer et al. (2013, ch.8); mclogit package.")
        X, y, _ = self._prepare_data(include_undetermined)
        model_full = sm.MNLogit(y, X)
        res_full = model_full.fit(disp=False)
        ll_full = res_full.llf
        null_model = sm.MNLogit(y, sm.add_constant(np.ones(len(y))))
        null_res = null_model.fit(disp=False)
        lr_stat_omni = -2 * (null_res.llf - ll_full)
        df_omni = (len(res_full.params.columns) - 1) * len(res_full.params.index)
        omnibus_p = 1 - stats.chi2.cdf(lr_stat_omni, df_omni)
        lrt_contrast_p = np.nan
        wald_contrast_p = np.nan
        contrast_or = np.nan
        if include_undetermined == False:
            bif_col = [c for c in X.columns if 'bifurcation' in c and 'no' not in c][0]
            coef_bif = res_full.params.loc[bif_col, 1]
            se_bif = res_full.bse.loc[bif_col, 1]
            contrast_or = np.exp(coef_bif)
            params_full = res_full.params
            flat_order = params_full.values.ravel('F')
            k_vars = params_full.shape[0]
            k_outcomes = params_full.shape[1]
            idx_map = {}
            for j, out in enumerate(params_full.columns):
                for i, var in enumerate(params_full.index):
                    idx_map[(var, out)] = j * k_vars + i
            coef_idx = idx_map[(bif_col, 1)]
            R = np.zeros((1, len(flat_order)))
            R[0, coef_idx] = 1.0
            q = np.array([0.0])
            res_constrained = _fit_mnlogit_constrained(model_full, R, q, start_params=res_full.params.values.ravel('F'), disp=False)
            ll_constrained = res_constrained.llf
            lr_stat = -2 * (ll_constrained - ll_full)
            lrt_contrast_p = 1 - stats.chi2.cdf(lr_stat, df=1)
            wald_stat = (coef_bif / se_bif)**2
            wald_contrast_p = 1 - stats.chi2.cdf(wald_stat, df=1)
        return {'converged': True, 'omnibus_p': omnibus_p, 'lrt_contrast_p': lrt_contrast_p, 'wald_contrast_p': wald_contrast_p, 'contrast_or': contrast_or, 'model': res_full, 'X': X, 'y': y, 'cat_order': ['no-bifurcation','bifurcation'] if not include_undetermined else ['undetermined','no-bifurcation','bifurcation']}

def run_analysis(probability_threshold=0.7):
    create_directories()
    bif_df = load_and_preprocess_bifurcation(threshold=probability_threshold)
    bif_df['dev_tipping_category'] = np.nan
    dev_mask = bif_df['early_warning'].notna()
    bif_df.loc[dev_mask, 'dev_tipping_category'] = 'no-bifurcation'
    bif_mask = dev_mask & (bif_df['early_warning'] == True) & (bif_df['bifurcation_type'] != 'no-bifurcation')
    bif_df.loc[bif_mask, 'dev_tipping_category'] = 'bifurcation'
    df_sub = compute_dl_dev_agreement(bif_df, threshold=probability_threshold)
    plot_dl_dev_agreement(df_sub, threshold=probability_threshold, outcome_name='frequentist_conservation')
    plot_dl_dev_roc(df_sub, outcome_name='frequentist_conservation')
    iucn_df = load_and_preprocess_iucn()
    merged_df = merge_datasets(bif_df, iucn_df)
    merged_df = pd.merge(merged_df, bif_df[['species_short', 'dev_tipping_category']], on='species_short', how='left')
    if merged_df.empty:
        print("No data after merging. Exiting.")
        return
    analysis = full_statistical_analysis(merged_df)
    res = analysis.run_multinomial_lr(include_undetermined=False)
    model = res['model']
    X = res['X']
    bif_col = [c for c in X.columns if 'bifurcation' in c and 'no' not in c][0]
    low, high, p_contrast = get_profile_ci_p(model, bif_col, 1)
    or_contrast = np.exp(model.params.loc[bif_col, 1])
    ci_lower = np.exp(low)
    ci_upper = np.exp(high)
    print("\n" + "="*100)
    print("FINAL RESULT: Multinomial LR")
    print("="*100)
    print(f"Omnibus LR test: p = {res['omnibus_p']:.3g}")
    print(f"Contrast (bifurcation vs. no-bifurcation) for threatened vs. non-threatened:")
    print(f"  OR = {res['contrast_or']:.3g}")
    print(f"  95% CI: [{ci_lower:.3g}, {ci_upper:.3g}]")
    print(f"  LRT p = {res['lrt_contrast_p']:.3g}")
    print(f"  Wald p = {res['wald_contrast_p']:.3g}")
    cat_order = res['cat_order']
    params = model.params
    odds_ratios = {}
    for outcome_level in params.columns:
        coef_series = params[outcome_level]
        or_series = np.exp(coef_series)
        ci_lower_series = []
        ci_upper_series = []
        pvals_series = []
        for var_name in params.index:
            if var_name == 'const':
                continue
            low2, high2, p2 = get_profile_ci_p(model, var_name, outcome_level)
            ci_lower_series.append(np.exp(low2))
            ci_upper_series.append(np.exp(high2))
            pvals_series.append(p2)
        const_coef = params.loc['const', outcome_level]
        const_se = model.bse.loc['const', outcome_level]
        const_or = np.exp(const_coef)
        const_low = np.exp(const_coef - 1.96 * const_se)
        const_high = np.exp(const_coef + 1.96 * const_se)
        const_p = 2 * (1 - stats.norm.cdf(np.abs(const_coef / const_se)))
        or_df = pd.DataFrame({
            'coefficient': np.concatenate([[const_coef], coef_series[1:].values]),
            'odds_ratio': np.concatenate([[const_or], np.exp(coef_series[1:].values)]),
            'ci_lower': np.concatenate([[const_low], ci_lower_series]),
            'ci_upper': np.concatenate([[const_high], ci_upper_series]),
            'p_value': np.concatenate([[const_p], pvals_series])
        }, index=params.index)
        odds_ratios[outcome_level] = or_df
    create_odds_ratio_plot(odds_ratios, 'tipping_category', 'multinomial_lr_final', cat_order[0], 'frequentist_conservation')
    contrast_res = {0: {'contrast': model.params.loc[bif_col, 1], 'se': model.bse.loc[bif_col, 1], 'p': p_contrast, 'or': or_contrast, 'ci_lower': ci_lower, 'ci_upper': ci_upper}}
    create_contrast_plot(contrast_res, 'multinomial_lr_final', 'frequentist_conservation', lrt_p=res['lrt_contrast_p'])

def main():
    probability_threshold = 0.7
    run_analysis(probability_threshold)

if __name__ == "__main__":
    main()
