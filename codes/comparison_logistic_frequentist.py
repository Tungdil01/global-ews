import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from firthmodels import FirthLogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_curve, auc
from joblib import Parallel, delayed
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import warnings
warnings.filterwarnings('ignore')
from main_orchestrator import probability_threshold
from main_dl import EXCLUDED_GROUPS, EXCLUDED_SPECIES_LABELS, get_cached_taxonomic_group
seed = 1234
np.random.seed(seed)
CONSERVATION_STATUS_CODES = {'non-threatened': 0, 'threatened': 1, 'unknown': 2}
CONSERVATION_STATUS_NAMES = {v: k for k, v in CONSERVATION_STATUS_CODES.items()}

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
    if row['tipping_point_type'] == 'undetermined':
        return 'undetermined'
    if row['best_probability'] < threshold:
        return 'undetermined'
    if row['tipping_point_type'] == 'no-bifurcation':
        return 'no-bifurcation'
    return 'bifurcation'

def classify_iucn_status(s):
    if s in ['LEAST_CONCERN', 'NEAR_THREATENED']:
        return 'non-threatened'
    elif s in ['VULNERABLE','ENDANGERED','CRITICALLY_ENDANGERED']:
        return 'threatened'
    elif s in ['EXTINCT', 'EXTINCT_IN_THE_WILD']:
        return 'excluded'
    else:
        return 'unknown'

def load_and_preprocess_bifurcation(threshold=probability_threshold):
    df = pd.read_csv('csv/bifurcation_results_short.csv')
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['tipping_category'] = df.apply(lambda row: categorize_bifurcation(row, threshold), axis=1)
    short_species = set(df['species_short'].dropna())
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_df['species_short'] = large_df['species'].apply(preprocess_species_name)
    large_df['best_probability'] = pd.to_numeric(large_df['best_probability'], errors='coerce').fillna(0.0)
    large_species = set(large_df['species_short'].dropna())
    low_conf = large_df[(large_df['best_probability'] < threshold) & (~large_df['species_short'].isin(short_species))].drop_duplicates(subset=['species_short'])
    low_conf_rows = low_conf[['species', 'species_short', 'dataset', 'timeseries_id', 'best_probability', 'tipping_point_type', 'taxonomic_group']].copy()
    low_conf_rows['tipping_category'] = 'undetermined'
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    if 'reason' in failed_df.columns:
        failed_df = failed_df[failed_df['reason'].astype(str).str.strip() != 'taxonomic_excluded']
    failed_df['species_short'] = failed_df['species'].apply(preprocess_species_name)
    failed_only = failed_df[~failed_df['species_short'].isin(large_species)].drop_duplicates(subset=['species_short'])
    failed_rows = pd.DataFrame({
        'species': failed_only['species'].values,
        'species_short': failed_only['species_short'].values,
        'dataset': 'failed_species',
        'timeseries_id': 'POOLED',
        'best_probability': 0.0,
        'tipping_point_type': 'undetermined',
        'tipping_category': 'undetermined'
    })
    failed_rows['taxonomic_group'] = failed_rows['species'].apply(get_cached_taxonomic_group)
    failed_rows = failed_rows[~failed_rows['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    failed_rows = failed_rows[~failed_rows['species'].isin(EXCLUDED_SPECIES_LABELS)]
    failed_rows = failed_rows[failed_rows['taxonomic_group'] != 'Other']
    valid = ['bifurcation', 'no-bifurcation', 'undetermined']
    df = df[df['tipping_category'].isin(valid)]
    combined = pd.concat([df, low_conf_rows, failed_rows], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=['species_short'], keep='first').reset_index(drop=True)
    print(f"[bifurcation] Applied probability threshold {threshold}:")
    print(f"  - Total species: {len(combined)}")
    print(f"  - Distribution:")
    for cat in valid:
        print(f"    {cat}: {(combined['tipping_category']==cat).sum()}")
    return combined

def load_and_preprocess_iucn():
    df = pd.read_csv('csv/iucn_species.csv', dtype=str)
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['IUCN_Status_raw'] = df['iucn_status'].astype(str).fillna('').str.strip().str.upper()
    df['conservation_status'] = df['IUCN_Status_raw'].apply(classify_iucn_status)
    dedup = df[['species_short','conservation_status','IUCN_Status_raw']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    short_df = pd.read_csv('csv/bifurcation_results_short.csv', dtype=str)
    short_species = set(short_df['species'].apply(preprocess_species_name).dropna())
    excluded_rows = dedup[(dedup['conservation_status'] == 'excluded') & (dedup['species_short'].isin(short_species))]
    for status in sorted(excluded_rows['IUCN_Status_raw'].unique()):
        species_list = sorted(excluded_rows.loc[excluded_rows['IUCN_Status_raw'] == status, 'species_short'].tolist())
        print(f"[iucn] Excluded species in short file ({status}): {species_list}")
    n_excluded = (dedup['conservation_status'] == 'excluded').sum()
    dedup = dedup[dedup['conservation_status'] != 'excluded'].reset_index(drop=True)
    print(f"[iucn] Preprocessed {len(dedup)} species: threatened={(dedup['conservation_status']=='threatened').sum()}, non-threatened={(dedup['conservation_status']=='non-threatened').sum()}, unknown={(dedup['conservation_status']=='unknown').sum()}, excluded={n_excluded}")
    return dedup

def merge_datasets(bifurcation_df, iucn_df):
    per_species = bifurcation_df[['species_short','tipping_category']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    short_set = set(per_species['species_short'])
    iucn_set = set(iucn_df['species_short'])
    missing = sorted(short_set - iucn_set)
    if len(missing) > 0:
        print(f"[merge] {len(missing)} species from short not in IUCN (inner join drops them):")
        for sp in missing:
            print(f"[merge]   {sp}")
    else:
        print(f"[merge] All {len(short_set)} short species have IUCN status.")
    merged = pd.merge(per_species, iucn_df[['species_short','conservation_status']], on='species_short', how='inner')
    print(f"Merged dataset: {len(merged)} species with conservation status.")
    return merged

def print_validation_diagnostics(bif_df, merged_df, firth_results, resampling_results, permutation_results=None):
    print("\n" + "="*100)
    print("VALIDATION DIAGNOSTICS")
    print("="*100)
    print(f"[val] short unique species: {bif_df['species_short'].nunique()}")
    print(f"[val] short rows (segments): {len(bif_df)}")
    print(f"[val] short tipping_category distribution:")
    for cat, cnt in bif_df['tipping_category'].value_counts().sort_index().items():
        print(f"[val]   {cat}: {cnt}")
    ct_short = pd.crosstab(bif_df['tipping_category'], bif_df['taxonomic_group'] if 'taxonomic_group' in bif_df.columns else bif_df['tipping_category'])
    print(f"[val] merged analysis set n: {len(merged_df)}")
    print(f"[val] merged conservation_status distribution:")
    for st, cnt in merged_df['conservation_status'].value_counts().sort_index().items():
        print(f"[val]   {st}: {cnt}")
    ct = pd.crosstab(merged_df['tipping_category'], merged_df['conservation_status'])
    print(f"[val] crosstab tipping_category vs conservation_status:")
    print(ct.to_string())
    if firth_results is not None:
        for model in ['binary', 'multinomial']:
            r = firth_results.get(model)
            if r is None:
                continue
            if 'error' in r:
                print(f"[val] firth_{model}: SKIPPED - {r['error']}")
                continue
            print(f"[val] firth_{model}: OR={r['or']:.3g} CI=[{r['ci_lower']:.3g}, {r['ci_upper']:.3g}] p={r['p_value']:.3g}")
    if resampling_results is not None:
        for label, r in resampling_results.items():
            if 'error' in r:
                continue
            print(f"[val] resample_{label}: OR_full={r['or_full']:.3g} CI=[{r['ci_lower']:.3g}, {r['ci_upper']:.3g}] p_boot={r['p_value']:.3g} n_valid={r['n_valid']}")
    if permutation_results is not None:
        for label, r in permutation_results.items():
            if 'error' in r:
                continue
            print(f"[val] permutation_{label}: OR_full={r['or_full']:.3g} CI=[{r['ci_lower']:.3g}, {r['ci_upper']:.3g}] p_perm={r['p_value']:.3g} n_valid={r['n_valid']}")
    print("="*100 + "\n")

def dev_columns_present(df):
    return ('early_warning' in df.columns) and ('bifurcation_type' in df.columns)

def compute_dl_dev_agreement(df, threshold=probability_threshold):
    if not dev_columns_present(df):
        print("No DEV columns present. Skipping DL-DEV agreement.")
        return None
    df = df[df['transition_year'].notna()].copy()
    dev_present = df['early_warning'].notna()
    if dev_present.sum() == 0:
        print("No DEV results found. Skipping DL-DEV agreement.")
        return None
    df_sub = df[dev_present].copy()
    df_sub['dl_bif'] = ((df_sub['best_probability'] >= threshold) & (df_sub['tipping_point_type'] != 'no-bifurcation')).astype(bool)
    df_sub['dev_bif'] = ((df_sub['early_warning'] == True) & (df_sub['bifurcation_type'] != 'no-bifurcation')).astype(bool)
    cross = pd.crosstab(df_sub['dl_bif'], df_sub['dev_bif'])
    print("\nDL vs DEV bifurcation classification (threshold={}):".format(threshold))
    print(cross)
    kappa = cohen_kappa_score(df_sub['dl_bif'], df_sub['dev_bif'])
    print(f"Cohen's kappa: {kappa:.3g}")
    chi2, p, dof, expected = stats.chi2_contingency(cross)
    print(f"Chi-square test: chi2={chi2:.3g}, p={p:.3g}, dof={dof}")
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
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = {:.2g})'.format(roc_auc))
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

def _print_class_mapping(y_series, label):
    y_arr = np.asarray(y_series)
    present_classes = sorted(int(v) for v in np.unique(y_arr) if not pd.isna(v))
    print(f"[{label}] y coding (y value -> conservation status):")
    for v in present_classes:
        print(f"[{label}]   {v} -> {CONSERVATION_STATUS_NAMES.get(v, 'UNKNOWN')}")
    print(f"[{label}] Model column order (non-reference classes):")
    for i, v in enumerate(present_classes[1:]):
        print(f"[{label}]   column {i} -> class {v} ('{CONSERVATION_STATUS_NAMES.get(v, 'UNKNOWN')}')")
    ref_vals = [present_classes[0]]
    ref_names = [CONSERVATION_STATUS_NAMES.get(present_classes[0], 'UNKNOWN')]
    print(f"[{label}] Reference class (implicit): {ref_vals} ({ref_names})")
    return present_classes, ref_vals, ref_names

def _assert_reference_is(ref_names, expected_reference, label, ref_vals=None):
    if expected_reference not in ref_names:
        codes_str = f" (codes {ref_vals})" if ref_vals is not None else ""
        raise RuntimeError(f"[{label}] Reference class mismatch: got {ref_names}{codes_str}, expected '{expected_reference}'. CONSERVATION_STATUS_CODES = {CONSERVATION_STATUS_CODES}. statsmodels MNLogit chose a different reference class than intended; reorder CONSERVATION_STATUS_CODES so that '{expected_reference}' has the smallest code. Aborting.")

def _check_design_health(X, y_label, bif_label, model_label, verbose=True):
    rank = np.linalg.matrix_rank(X.values)
    crosstab = pd.crosstab(bif_label, y_label)
    if rank < X.shape[1]:
        if verbose:
            print(f"[{model_label}] Design matrix rank {rank} < {X.shape[1]} columns. Crosstab:")
            print(crosstab)
        raise RuntimeError(f"[{model_label}] Design matrix is rank-deficient (rank={rank}, columns={X.shape[1]}).")
    if (crosstab.values == 0).any():
        if verbose:
            print(f"[{model_label}] Empty cell(s) in crosstab. Crosstab:")
            print(crosstab)
        raise RuntimeError(f"[{model_label}] Crosstab has empty cell(s); fit Hessian would be singular.")

def _fit_or(df_samp, n_classes, verbose=False):
    unknown_code = CONSERVATION_STATUS_CODES['unknown']
    threatened_code = CONSERVATION_STATUS_CODES['threatened']
    non_threatened_code = CONSERVATION_STATUS_CODES['non-threatened']
    if n_classes == 2:
        df_fit = df_samp[df_samp['y'] != unknown_code].copy()
    else:
        df_fit = df_samp
    X = sm.add_constant(df_fit['bif'].astype(float), has_constant='add')
    y = df_fit['y'].astype(int).values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _check_design_health(X, df_fit['conservation_status'], df_fit['tipping_category'], 'resample', verbose=False)
        if n_classes == 2:
            y_bin = (y == threatened_code).astype(int)
            res = sm.Logit(y_bin, X).fit(disp=False, method='bfgs', skip_hessian=True)
            if verbose:
                print(f"[logit_binary verbose] y_bin: 0='{CONSERVATION_STATUS_NAMES[non_threatened_code]}' (reference), 1='{CONSERVATION_STATUS_NAMES[threatened_code]}'")
            return float(np.exp(res.params['bif']))
        else:
            res = sm.MNLogit(y, X).fit(disp=False, method='bfgs', skip_hessian=True)
            present_classes = sorted(int(v) for v in np.unique(y) if not pd.isna(v))
            non_ref_classes = present_classes[1:]
            res.params = res.params.set_axis(non_ref_classes, axis=1)
            if verbose:
                _, ref_vals, ref_names = _print_class_mapping(y, 'resample_mnlogit')
                _assert_reference_is(ref_names, 'non-threatened', 'resample_mnlogit', ref_vals)
            if threatened_code not in res.params.columns:
                raise RuntimeError(f"Threatened class (code {threatened_code}) not in resample MNLogit params columns: {list(res.params.columns)}")
            return float(np.exp(res.params.loc['bif', threatened_code]))

def _resample_single_or(df, idx_threatened, idx_non_threatened, idx_unknown, n_threatened, replace_non, replace_unk, seed_i, n_classes):
    rng = np.random.RandomState(seed_i)
    samp_non = rng.choice(idx_non_threatened, size=n_threatened, replace=replace_non)
    samp_unk = rng.choice(idx_unknown, size=n_threatened, replace=replace_unk)
    samp_idx = np.concatenate([idx_threatened, samp_non, samp_unk])
    df_samp = df.loc[samp_idx]
    try:
        return _fit_or(df_samp, n_classes)
    except (RuntimeError, PerfectSeparationError, np.linalg.LinAlgError) as e:
        print(f"[resample worker seed={seed_i}] caught {type(e).__name__}: {e}")
        return np.nan

def resampling_analysis(merged_df, n_iter=10000, seed=1234, n_classes=2, combo_label='baseline'):
    print("\n" + "="*100)
    print(f"RESAMPLING ANALYSIS [{combo_label}]")
    print(f"n_classes={n_classes}")
    print("="*100)
    threatened_code = CONSERVATION_STATUS_CODES['threatened']
    unknown_code = CONSERVATION_STATUS_CODES['unknown']
    non_threatened_code = CONSERVATION_STATUS_CODES['non-threatened']
    if n_classes >= 3:
        min_code = min(CONSERVATION_STATUS_CODES.values())
        if non_threatened_code != min_code:
            raise RuntimeError(f"[{combo_label}] non-threatened code ({non_threatened_code}) is not the smallest CONSERVATION_STATUS_CODES value ({min_code}). statsmodels MNLogit uses the smallest class code as reference; reference must be non-threatened. Check CONSERVATION_STATUS_CODES. Aborting before resampling.")
    df = merged_df[merged_df['tipping_category'].isin(['bifurcation', 'no-bifurcation'])].copy()
    df['y'] = df['conservation_status'].map(CONSERVATION_STATUS_CODES)
    df['bif'] = (df['tipping_category'] == 'bifurcation').astype(int)
    df = df.dropna(subset=['y', 'bif']).reset_index(drop=True)
    n_threatened = (df['y'] == threatened_code).sum()
    print(f"n_threatened = {n_threatened}")
    if n_threatened == 0:
        print("No threatened species, aborting resampling.")
        return None, None, None, None, None
    idx_threatened = df.index[df['y'] == threatened_code].tolist()
    idx_non_threatened = df.index[df['y'] == non_threatened_code].tolist()
    idx_unknown = df.index[df['y'] == unknown_code].tolist()
    n_non = len(idx_non_threatened)
    n_unk = len(idx_unknown)
    print(f"Available: non-threatened={n_non}, threatened={n_threatened}, unknown={n_unk}")
    replace_non = n_non < n_threatened
    replace_unk = n_unk < n_threatened
    if replace_non:
        print("Warning: non-threatened group smaller than n_threatened; sampling with replacement.")
    if replace_unk:
        print("Warning: unknown group smaller than n_threatened; sampling with replacement.")
    seeds = [seed + i for i in range(n_iter)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ors_all = Parallel(n_jobs=-1)(delayed(_resample_single_or)(df, idx_threatened, idx_non_threatened, idx_unknown, n_threatened, replace_non, replace_unk, s, n_classes) for s in seeds)
    ors_all = np.array(ors_all, dtype=float)
    n_failed = int(np.sum(np.isnan(ors_all)))
    ors = ors_all[~np.isnan(ors_all)]
    pct_failed = 100.0 * n_failed / n_iter if n_iter > 0 else 0.0
    print(f"Completed {len(ors)} valid iterations out of {n_iter} ({n_failed} failed, {pct_failed:.3g}%).")
    if len(ors) == 0:
        print("All iterations failed; cannot report.")
        return None, None, None, None, None
    try:
        or_full = _fit_or(df, n_classes, verbose=True)
    except RuntimeError as e:
        print(f"Full-data fit failed: {type(e).__name__}: {e}")
        or_full = np.nan
    print(f"Full data OR: {or_full:.3g}")
    lower = np.percentile(ors, 2.5)
    upper = np.percentile(ors, 97.5)
    print(f"Resampled 95% CI: [{lower:.3g}, {upper:.3g}]")
    print(f"Proportion of ORs < 1: {np.mean(ors < 1):.3g}")
    print(f"Proportion of ORs > 1: {np.mean(ors > 1):.3g}")
    count_lt_one = int(np.sum(ors < 1.0))
    count_gt_one = int(np.sum(ors > 1.0))
    p_left = count_lt_one / len(ors)
    p_right = count_gt_one / len(ors)
    p_boot = float(min(1.0, 2.0 * min(p_left, p_right)))
    print(f"Bootstrap percentile diagnostic p (H0: OR = 1, robustness only): {p_boot:.3g}")
    print(f"  count OR < 1 = {count_lt_one}")
    print(f"  count OR > 1 = {count_gt_one}")
    if p_boot == 0.0:
        print("  WARNING: p_boot = 0 boundary value; no resampled OR crossed 1. Interpret as boundary artifact, not evidence of p < 1/(2*n_valid).")
    print(f"OR distribution diagnostics:")
    print(f"  min = {np.min(ors):.3g}")
    print(f"  median = {np.median(ors):.3g}")
    print(f"  mean = {np.mean(ors):.3g}")
    print(f"  max = {np.max(ors):.3g}")
    for p in [75, 90, 95, 97.5, 99, 99.5, 99.9]:
        print(f"  percentile {p} = {np.percentile(ors, p):.3g}")
    print(f"  count > 5 = {int(np.sum(ors > 5))}")
    print(f"  count > 10 = {int(np.sum(ors > 10))}")
    print(f"  count > 100 = {int(np.sum(ors > 100))}")
    print(f"  count > 1e6 = {int(np.sum(ors > 1e6))}")
    set_plot_style()
    if np.isfinite(or_full):
        xmax_plot = max(or_full * 5.0, 2.0)
    else:
        xmax_plot = 2.0
    n_above = int(np.sum(ors > xmax_plot))
    pct_above = 100.0 * n_above / len(ors)
    print(f"Plot xmax = {xmax_plot:.3g}; {n_above} ORs ({pct_above:.3g}%) lie beyond the visible range.")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(ors, bins=100, range=(0, xmax_plot), density=True, alpha=0.7, color='skyblue', edgecolor='black')
    if np.isfinite(or_full):
        ax.axvline(or_full, color='red', linestyle='--', linewidth=2, label=f'Full data OR = {or_full:.2g}')
    ax.axvline(1.0, color='black', linestyle='-', linewidth=1, label='OR = 1')
    ax.axvline(lower, color='gray', linestyle=':', linewidth=1.5, label=f'2.5th pct = {lower:.2g}')
    ax.axvline(upper, color='gray', linestyle=':', linewidth=1.5, label=f'97.5th pct = {upper:.2g}')
    ax.set_xlabel('Odds Ratio (bifurcation vs no-bifurcation)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, xmax_plot)
    ax.legend()
    plt.tight_layout()
    filename = f'img/resampling_odds_ratios_{combo_label}.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved histogram to {filename}")
    pd.DataFrame({'odds_ratio': ors}).to_csv(f'csv/resampling_odds_ratios_{combo_label}.csv', index=False)
    return ors, or_full, lower, upper, p_boot

def run_resampling_combos(merged_df, do_binary, n_iter=10000, seed=1234):
    combos = [('binary_baseline', 2), ('multinomial_baseline', 3)] if do_binary else [('multinomial_baseline', 3)]
    results = {}
    for label, n_classes in combos:
        try:
            ors, or_full, lower, upper, p_boot = resampling_analysis(merged_df, n_iter=n_iter, seed=seed, n_classes=n_classes, combo_label=label)
            if ors is not None:
                results[label] = {'n_classes': n_classes, 'or_full': float(or_full) if np.isfinite(or_full) else np.nan, 'ci_lower': float(lower), 'ci_upper': float(upper), 'p_value': float(p_boot), 'n_valid': int(len(ors)), 'mean_ors': float(np.mean(ors)), 'median_ors': float(np.median(ors))}
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")
            results[label] = {'error': f"{type(e).__name__}: {e}"}
    print("\n" + "="*100)
    print("SUMMARY OF COMBOS")
    print("="*100)
    for label, r in results.items():
        if 'error' in r:
            print(f"{label}: FAILED - {r['error']}")
        else:
            print(f"{label} (n_classes={r['n_classes']}): full_OR={r['or_full']:.3g}, CI=[{r['ci_lower']:.3g}, {r['ci_upper']:.3g}], p_boot={r['p_value']:.3g}, n={r['n_valid']}, mean={r['mean_ors']:.3g}, median={r['median_ors']:.3g}")
    return results

def run_exact_logistic_binary(merged_df):
    df = merged_df[merged_df['tipping_category'].isin(['bifurcation', 'no-bifurcation'])].copy()
    df['y'] = df['conservation_status'].map(CONSERVATION_STATUS_CODES)
    df['bif'] = (df['tipping_category'] == 'bifurcation').astype(int)
    df = df.dropna(subset=['y', 'bif']).reset_index(drop=True)
    df = df[df['y'] != CONSERVATION_STATUS_CODES['unknown']].copy()
    table = pd.crosstab(df['bif'], df['y'])
    or_val, p_val = stats.fisher_exact(table.values)
    ci = stats.contingency.odds_ratio(table.values, kind='conditional').confidence_interval(confidence_level=0.95)
    low_ci = float(ci.low)
    high_ci = float(ci.high)
    print(f"[Exact binary] Fisher exact OR = {or_val:.3g}, 95% exact CI = [{low_ci:.3g}, {high_ci:.3g}], p = {p_val:.3g}")
    pd.DataFrame([{'arm':'D_Exact','model':'binary','odds_ratio':float(or_val),'ci_lower':low_ci,'ci_upper':high_ci,'p_value':float(p_val)}]).to_csv('csv/exact_logistic_results.csv', index=False)
    return {'binary': {'or': float(or_val), 'ci_lower': low_ci, 'ci_upper': high_ci, 'p_value': float(p_val)}}

def _permute_single_or(df, seed_i, n_classes):
    rng = np.random.RandomState(seed_i)
    df_samp = df.copy()
    perm = rng.permutation(df_samp['bif'].values)
    df_samp['bif'] = perm
    df_samp['tipping_category'] = np.where(perm == 1, 'bifurcation', 'no-bifurcation')
    try:
        return _fit_or(df_samp, n_classes)
    except (RuntimeError, PerfectSeparationError, np.linalg.LinAlgError) as e:
        print(f"[permute worker seed={seed_i}] caught {type(e).__name__}: {e}")
        return np.nan

def permutation_analysis(merged_df, n_iter=10000, seed=1234, n_classes=2, combo_label='permutation_binary'):
    print("\n" + "="*100)
    print(f"PERMUTATION ANALYSIS [{combo_label}]")
    print(f"n_classes={n_classes}")
    print("="*100)
    threatened_code = CONSERVATION_STATUS_CODES['threatened']
    unknown_code = CONSERVATION_STATUS_CODES['unknown']
    non_threatened_code = CONSERVATION_STATUS_CODES['non-threatened']
    if n_classes >= 3:
        min_code = min(CONSERVATION_STATUS_CODES.values())
        if non_threatened_code != min_code:
            raise RuntimeError(f"[{combo_label}] non-threatened code ({non_threatened_code}) is not the smallest CONSERVATION_STATUS_CODES value ({min_code}). statsmodels MNLogit uses the smallest class code as reference; reference must be non-threatened. Check CONSERVATION_STATUS_CODES. Aborting before permutation.")
    df = merged_df[merged_df['tipping_category'].isin(['bifurcation', 'no-bifurcation'])].copy()
    df['y'] = df['conservation_status'].map(CONSERVATION_STATUS_CODES)
    df['bif'] = (df['tipping_category'] == 'bifurcation').astype(int)
    df = df.dropna(subset=['y', 'bif']).reset_index(drop=True)
    n_threatened = (df['y'] == threatened_code).sum()
    print(f"n_threatened = {n_threatened}")
    if n_threatened == 0:
        print("No threatened species, aborting permutation.")
        return None, None, None, None, None
    try:
        or_full = _fit_or(df, n_classes, verbose=True)
    except RuntimeError as e:
        print(f"Full-data fit failed: {type(e).__name__}: {e}")
        or_full = np.nan
    print(f"Full data OR: {or_full:.3g}")
    seeds = [seed + i for i in range(n_iter)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ors_all = Parallel(n_jobs=-1)(delayed(_permute_single_or)(df, s, n_classes) for s in seeds)
    ors_all = np.array(ors_all, dtype=float)
    n_failed = int(np.sum(np.isnan(ors_all)))
    ors = ors_all[~np.isnan(ors_all)]
    pct_failed = 100.0 * n_failed / n_iter if n_iter > 0 else 0.0
    print(f"Completed {len(ors)} valid iterations out of {n_iter} ({n_failed} failed, {pct_failed:.3g}%).")
    if len(ors) == 0:
        print("All iterations failed; cannot report.")
        return None, None, None, None, None
    lower = np.percentile(ors, 2.5)
    upper = np.percentile(ors, 97.5)
    print(f"Permuted 95% CI: [{lower:.3g}, {upper:.3g}]")
    if np.isfinite(or_full):
        p_left = float(np.mean(ors <= or_full))
        p_right = float(np.mean(ors >= or_full))
        p_perm = float(min(1.0, 2.0 * min(p_left, p_right)))
    else:
        p_perm = np.nan
    print(f"Permutation two-sided diagnostic p (H0: OR = 1, robustness only): {p_perm:.3g}")
    print(f"OR distribution diagnostics:")
    print(f"  min = {np.min(ors):.3g}")
    print(f"  median = {np.median(ors):.3g}")
    print(f"  mean = {np.mean(ors):.3g}")
    print(f"  max = {np.max(ors):.3g}")
    for p in [75, 90, 95, 97.5, 99, 99.5, 99.9]:
        print(f"  percentile {p} = {np.percentile(ors, p):.3g}")
    set_plot_style()
    if np.isfinite(or_full):
        xmax_plot = max(or_full * 5.0, 2.0)
    else:
        xmax_plot = 2.0
    n_above = int(np.sum(ors > xmax_plot))
    pct_above = 100.0 * n_above / len(ors)
    print(f"Plot xmax = {xmax_plot:.3g}; {n_above} ORs ({pct_above:.3g}%) lie beyond the visible range.")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(ors, bins=100, range=(0, xmax_plot), density=True, alpha=0.7, color='skyblue', edgecolor='black')
    if np.isfinite(or_full):
        ax.axvline(or_full, color='red', linestyle='--', linewidth=2, label=f'Full data OR = {or_full:.2g}')
    ax.axvline(1.0, color='black', linestyle='-', linewidth=1, label='OR = 1')
    ax.axvline(lower, color='gray', linestyle=':', linewidth=1.5, label=f'2.5th pct = {lower:.2g}')
    ax.axvline(upper, color='gray', linestyle=':', linewidth=1.5, label=f'97.5th pct = {upper:.2g}')
    ax.set_xlabel('Odds Ratio (bifurcation vs no-bifurcation)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, xmax_plot)
    ax.legend()
    plt.tight_layout()
    filename = f'img/permutation_odds_ratios_{combo_label}.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved histogram to {filename}")
    pd.DataFrame({'odds_ratio': ors}).to_csv(f'csv/permutation_odds_ratios_{combo_label}.csv', index=False)
    return ors, or_full, lower, upper, p_perm

def run_permutation_combos(merged_df, do_binary, n_iter=10000, seed=1234):
    combos = [('binary_baseline', 2), ('multinomial_baseline', 3)] if do_binary else [('multinomial_baseline', 3)]
    results = {}
    for label, n_classes in combos:
        try:
            ors, or_full, lower, upper, p_perm = permutation_analysis(merged_df, n_iter=n_iter, seed=seed, n_classes=n_classes, combo_label=label)
            if ors is not None:
                results[label] = {'n_classes': n_classes, 'or_full': float(or_full) if np.isfinite(or_full) else np.nan, 'ci_lower': float(lower), 'ci_upper': float(upper), 'p_value': float(p_perm), 'n_valid': int(len(ors)), 'mean_ors': float(np.mean(ors)), 'median_ors': float(np.median(ors))}
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")
            results[label] = {'error': f"{type(e).__name__}: {e}"}
    print("\n" + "="*100)
    print("SUMMARY OF PERMUTATION COMBOS")
    print("="*100)
    rows = []
    for label, r in results.items():
        if 'error' in r:
            print(f"{label}: FAILED - {r['error']}")
        else:
            print(f"{label} (n_classes={r['n_classes']}): full_OR={r['or_full']:.3g}, CI=[{r['ci_lower']:.3g}, {r['ci_upper']:.3g}], p_perm={r['p_value']:.3g}, n={r['n_valid']}, mean={r['mean_ors']:.3g}, median={r['median_ors']:.3g}")
            rows.append({'arm':'C_Permutation','model':label,'odds_ratio':r['or_full'],'ci_lower':r['ci_lower'],'ci_upper':r['ci_upper'],'p_value':r['p_value'],'n_valid':r['n_valid'],'mean_ors':r['mean_ors'],'median_ors':r['median_ors']})
    pd.DataFrame(rows).to_csv('csv/permutation_results.csv', index=False)
    return results

def run_firth_analysis(merged_df, do_binary, outcome_name='frequentist_conservation'):
    print("\n" + "="*100)
    print("ARM A: FIRTH BIAS-REDUCED ESTIMATOR (FULL DATA)")
    print("="*100)
    df = merged_df[merged_df['tipping_category'].isin(['bifurcation', 'no-bifurcation'])].copy()
    df['y'] = df['conservation_status'].map(CONSERVATION_STATUS_CODES)
    df['bif'] = (df['tipping_category'] == 'bifurcation').astype(int)
    df = df.dropna(subset=['y', 'bif']).reset_index(drop=True)
    threatened_code = CONSERVATION_STATUS_CODES['threatened']
    unknown_code = CONSERVATION_STATUS_CODES['unknown']
    results = {}
    if do_binary:
        try:
            df_bin = df[df['y'] != unknown_code].copy()
            X_bin_full = sm.add_constant(df_bin['bif'].astype(float), has_constant='add')
            y_bin = (df_bin['y'] == threatened_code).astype(int).values
            _check_design_health(X_bin_full, df_bin['conservation_status'], df_bin['tipping_category'], 'firth_binary')
            fl = FirthLogisticRegression(fit_intercept=False)
            fl.fit(X_bin_full.values, y_bin)
            fl.lrt(1)
            or_bin = float(np.exp(fl.coef_[1]))
            ci_bin = fl.conf_int(method='pl')
            low_bin = float(ci_bin[1, 0])
            high_bin = float(ci_bin[1, 1])
            p_bin = float(fl.lrt_pvalues_[1])
            print(f"[Firth binary] OR_bif = {or_bin:.3g}, 95% PL CI = [{np.exp(low_bin):.3g}, {np.exp(high_bin):.3g}], PL LRT p = {p_bin:.3g}")
            results['binary'] = {'or': or_bin, 'ci_lower': float(np.exp(low_bin)), 'ci_upper': float(np.exp(high_bin)), 'p_value': float(p_bin)}
            create_contrast_plot({0: {'or': or_bin, 'ci_lower': results['binary']['ci_lower'], 'ci_upper': results['binary']['ci_upper'], 'p': results['binary']['p_value']}}, 'firth_binary', outcome_name, lrt_p=results['binary']['p_value'])
        except Exception as e:
            print(f"[Firth binary] SKIPPED: {type(e).__name__}: {e}")
            results['binary'] = {'error': f"{type(e).__name__}: {e}"}
    try:
        X_multi = sm.add_constant(df['bif'].astype(float), has_constant='add')
        y_multi = df['y'].astype(int).values
        _check_design_health(X_multi, df['conservation_status'], df['tipping_category'], 'firth_multinomial')
        print(f"[Firth multinomial] crosstab (n = {len(df)}):")
        print(pd.crosstab(df['conservation_status'], df['bif']).to_string())
        try:
            res_unpen = sm.MNLogit(y_multi, X_multi).fit(disp=False)
            present = sorted(int(v) for v in np.unique(y_multi) if not pd.isna(v))
            res_unpen.params = res_unpen.params.set_axis(present[1:], axis=1)
            or_unpen_T = float(np.exp(res_unpen.params.loc['bif', threatened_code]))
            print(f"[Firth multinomial] unpenalized MNLogit OR_bif(T vs NT) = {or_unpen_T:.3g}")
        except Exception as e_unpen:
            print(f"[Firth multinomial] unpenalized MNLogit failed: {type(e).__name__}: {e_unpen}")
        importr('pmlr')
        df_r = df[['conservation_status', 'bif']].copy()
        ro.globalenv['df_r'] = ro.DataFrame({'conservation_status': ro.StrVector(df_r['conservation_status'].astype(str).tolist()), 'bif': ro.IntVector(df_r['bif'].astype(int).tolist())})
        r_script = (
            "df_r$conservation_status <- factor(df_r$conservation_status, levels = c('non-threatened', 'threatened', 'unknown'))\n"
            "fit_pmlr <- pmlr::pmlr(conservation_status ~ bif, data = df_r, penalized = TRUE, method = 'likelihood', CI.calculate = TRUE, CI.alpha = 0.05, tol = 1e-4, verbose = FALSE)\n"
            "summ_pmlr <- summary(fit_pmlr)\n"
            "coef_pmlr <- summ_pmlr$coef\n"
            "beta_T_pmlr <- coef_pmlr['bif', 'Estimate', 'threatened']\n"
            "beta_U_pmlr <- coef_pmlr['bif', 'Estimate', 'unknown']\n"
            "or_T_pmlr <- coef_pmlr['bif', 'OR Estimate', 'threatened']\n"
            "ci_low_pmlr <- coef_pmlr['bif', 'Lower CI (OR)', 'threatened']\n"
            "ci_high_pmlr <- coef_pmlr['bif', 'Upper CI (OR)', 'threatened']\n"
            "chisq_pmlr <- coef_pmlr['bif', 'ChiSq', 'threatened']\n"
            "p_pmlr <- coef_pmlr['bif', 'Pr(>ChiSq)', 'threatened']\n"
            "cat(sprintf('R pmlr: beta_T = %.6f, beta_U = %.6f, |beta_T - beta_U| = %.6g\\n', beta_T_pmlr, beta_U_pmlr, abs(beta_T_pmlr - beta_U_pmlr)))\n"
            "cat(sprintf('R pmlr: OR_T = %.6f, profile CI = [%.6f, %.6f], ChiSq = %.6f, p = %.6g\\n', or_T_pmlr, ci_low_pmlr, ci_high_pmlr, chisq_pmlr, p_pmlr))\n"
            "beta_T_pmlr_out <- beta_T_pmlr; beta_U_pmlr_out <- beta_U_pmlr\n"
            "or_pmlr_out <- or_T_pmlr; ci_low_pmlr_out <- ci_low_pmlr; ci_high_pmlr_out <- ci_high_pmlr\n"
            "chisq_pmlr_out <- chisq_pmlr; p_pmlr_out <- p_pmlr\n"
        )
        ro.r(r_script)
        beta_T_pmlr = float(np.array(ro.globalenv['beta_T_pmlr_out'])[0])
        beta_U_pmlr = float(np.array(ro.globalenv['beta_U_pmlr_out'])[0])
        or_pmlr = float(np.array(ro.globalenv['or_pmlr_out'])[0])
        ci_low_pmlr = float(np.array(ro.globalenv['ci_low_pmlr_out'])[0])
        ci_high_pmlr = float(np.array(ro.globalenv['ci_high_pmlr_out'])[0])
        chisq_pmlr = float(np.array(ro.globalenv['chisq_pmlr_out'])[0])
        p_pmlr = float(np.array(ro.globalenv['p_pmlr_out'])[0])
        print(f"[Firth multinomial] pmlr: beta_T = {beta_T_pmlr:.3g}, beta_U = {beta_U_pmlr:.3g}, |beta_T - beta_U| = {abs(beta_T_pmlr - beta_U_pmlr):.3g}")
        print(f"[Firth multinomial] pmlr OR_bif(T vs NT) = {or_pmlr:.3g}, 95% profile CI = [{ci_low_pmlr:.3g}, {ci_high_pmlr:.3g}], LRT ChiSq = {chisq_pmlr:.3g}, LRT p = {p_pmlr:.3g}")
        if np.isnan(ci_low_pmlr) or np.isnan(ci_high_pmlr):
            raise RuntimeError("pmlr profile CI returned NaN even with tol=1e-4. Near-separation may be too severe for finite CI.")
        results['multinomial'] = {'or': or_pmlr, 'ci_lower': ci_low_pmlr, 'ci_upper': ci_high_pmlr, 'p_value': p_pmlr}
        create_contrast_plot({0: {'or': or_pmlr, 'ci_lower': results['multinomial']['ci_lower'], 'ci_upper': results['multinomial']['ci_upper'], 'p': p_pmlr}}, 'firth_multinomial', outcome_name, lrt_p=p_pmlr)
    except Exception as e:
        print(f"[Firth multinomial] SKIPPED: {type(e).__name__}: {e}")
        results['multinomial'] = {'error': f"{type(e).__name__}: {e}"}
    rows = []
    for model in ['binary', 'multinomial']:
        r = results.get(model)
        if r is None or 'error' in r:
            continue
        rows.append({'arm': 'A_Firth', 'model': model, 'odds_ratio': r['or'], 'ci_lower': r['ci_lower'], 'ci_upper': r['ci_upper'], 'p_value': r['p_value']})
    pd.DataFrame(rows).to_csv('csv/firth_results.csv', index=False)
    return results

def write_comparison_csv(firth_results, resampling_results, permutation_results, exact_results):
    rows = []
    for model in ['binary', 'multinomial']:
        fa = firth_results.get(model)
        if fa is not None and 'error' not in fa:
            rows.append({'arm': 'A_Firth', 'model': model, 'odds_ratio': fa['or'], 'ci_lower': fa['ci_lower'], 'ci_upper': fa['ci_upper'], 'p_value': fa['p_value'], 'n_valid': np.nan, 'mean_ors': np.nan, 'median_ors': np.nan})
        rk = f'{model}_baseline'
        rb = resampling_results.get(rk)
        if rb is not None and 'error' not in rb:
            rows.append({'arm': 'B_Resampling', 'model': model, 'odds_ratio': rb['or_full'], 'ci_lower': rb['ci_lower'], 'ci_upper': rb['ci_upper'], 'p_value': rb['p_value'], 'n_valid': rb['n_valid'], 'mean_ors': rb['mean_ors'], 'median_ors': rb['median_ors']})
        rp = permutation_results.get(rk)
        if rp is not None and 'error' not in rp:
            rows.append({'arm': 'C_Permutation', 'model': model, 'odds_ratio': rp['or_full'], 'ci_lower': rp['ci_lower'], 'ci_upper': rp['ci_upper'], 'p_value': rp['p_value'], 'n_valid': rp['n_valid'], 'mean_ors': rp['mean_ors'], 'median_ors': rp['median_ors']})
        rd = exact_results.get(model)
        if rd is not None:
            rows.append({'arm': 'D_Exact', 'model': model, 'odds_ratio': rd['or'], 'ci_lower': rd['ci_lower'], 'ci_upper': rd['ci_upper'], 'p_value': rd['p_value'], 'n_valid': np.nan, 'mean_ors': np.nan, 'median_ors': np.nan})
    df_out = pd.DataFrame(rows)
    df_out.to_csv('csv/four_arm_comparison.csv', index=False)
    print("\n" + "="*100)
    print("FOUR-ARM COMPARISON (Arm A = Firth full data, Arm B = unpenalized balanced resampling, Arm C = permutation, Arm D = exact binary)")
    print("="*100)
    print(df_out.to_string(index=False))
    return df_out

def enrich_with_biomes(df):
    biomes_df = pd.read_csv('csv/biomes.csv')
    biomes_df['species_short'] = biomes_df['species'].apply(preprocess_species_name)
    biomes_df = biomes_df[['species_short', 'biome', 'koppen_zone', 'meow_zone']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    df = pd.merge(df, biomes_df, on='species_short', how='left')
    df = df.rename(columns={'biome': 'biomes_category', 'koppen_zone': 'koppen_category', 'meow_zone': 'meow_category'})
    df['plant_animal_category'] = np.nan
    df['plant_animal_binary'] = np.nan
    plant_mask = df['taxonomic_group'].notna()
    df.loc[plant_mask, 'plant_animal_binary'] = df.loc[plant_mask, 'is_plant']
    df.loc[plant_mask, 'plant_animal_category'] = df.loc[plant_mask, 'is_plant'].map({0: 'animal', 1: 'plant'})
    df = df.drop(columns=['is_plant'])
    tax_others = sorted([v for v in df['taxonomic_group'].dropna().unique() if v != 'Other'])
    tax_codes = pd.Categorical(df['taxonomic_group'], categories=['Other'] + tax_others).codes
    df['taxonomy_binary'] = np.where(tax_codes == -1, np.nan, tax_codes)
    bio_vals = df['biomes_category']
    bio_ref_mask = bio_vals.str.strip().str.lower().isin(['unknown', 'unclassified', 'unknown/unclassified'])
    bio_clean = bio_vals.where(~bio_ref_mask, 'Unknown')
    bio_others = sorted([v for v in bio_clean.dropna().unique() if v != 'Unknown'])
    bio_codes = pd.Categorical(bio_clean, categories=['Unknown'] + bio_others).codes
    df['biomes_binary'] = np.where(bio_codes == -1, np.nan, bio_codes)
    kop_vals = df['koppen_category']
    largest_koppen = kop_vals.value_counts().idxmax()
    kop_others = sorted([v for v in kop_vals.dropna().unique() if v != largest_koppen])
    kop_codes = pd.Categorical(kop_vals, categories=[largest_koppen] + kop_others).codes
    df['koppen_binary'] = np.where(kop_codes == -1, np.nan, kop_codes)
    meow_vals = df['meow_category']
    largest_meow = meow_vals.value_counts().idxmax()
    meow_others = sorted([v for v in meow_vals.dropna().unique() if v != largest_meow])
    meow_codes = pd.Categorical(meow_vals, categories=[largest_meow] + meow_others).codes
    df['meow_binary'] = np.where(meow_codes == -1, np.nan, meow_codes)
    print(f"[biomes] Mapped biomes for {len(df)} species. Largest koppen: {largest_koppen}; largest meow: {largest_meow}.")
    print(f"[biomes] biomes_binary distribution: {df['biomes_binary'].value_counts(dropna=False).to_dict()}")
    return df

def run_analysis(probability_threshold, do_resampling, do_exact, do_permutation, do_binary, n_iter=10000, seed=1234):
    create_directories()
    bif_df = load_and_preprocess_bifurcation(threshold=probability_threshold)
    pre_dedup = len(bif_df)
    species_counts = bif_df['species_short'].value_counts()
    duplicated_short = species_counts[species_counts > 1]
    if len(duplicated_short) > 0:
        print(f"[dedup] Collapsing {int(duplicated_short.sum()) - len(duplicated_short)} duplicate rows across {len(duplicated_short)} species_short:")
        for sp, cnt in duplicated_short.items():
            print(f"[dedup]   {sp}: {int(cnt)} rows")
    else:
        print(f"[dedup] No duplicate species_short in short file ({pre_dedup} rows).")
    bif_df = bif_df.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    print(f"[dedup] After dedup: {len(bif_df)} unique species_short (dropped {pre_dedup - len(bif_df)})")
    has_dev = dev_columns_present(bif_df)
    bif_df['dev_tipping_category'] = np.nan
    if has_dev:
        dev_mask = bif_df['early_warning'].notna()
        bif_df.loc[dev_mask, 'dev_tipping_category'] = 'no-bifurcation'
        bif_mask = dev_mask & (bif_df['early_warning'] == True) & (bif_df['bifurcation_type'] != 'no-bifurcation')
        bif_df.loc[bif_mask, 'dev_tipping_category'] = 'bifurcation'
        df_sub = compute_dl_dev_agreement(bif_df, threshold=probability_threshold)
        plot_dl_dev_agreement(df_sub, threshold=probability_threshold, outcome_name='frequentist_conservation')
        plot_dl_dev_roc(df_sub, outcome_name='frequentist_conservation')
    else:
        print("DEV columns not present in bifurcation_results_short.csv. Skipping all DEV statistics.")
    iucn_df = load_and_preprocess_iucn()
    merged_df = merge_datasets(bif_df, iucn_df)
    extra_cols = ['species_short', 'dev_tipping_category', 'tipping_point_type', 'taxonomic_group']
    extra_cols = [c for c in extra_cols if c in bif_df.columns]
    merged_df = pd.merge(merged_df, bif_df[extra_cols], on='species_short', how='left')
    merged_df['is_plant'] = merged_df['taxonomic_group'].isin({'Bryophyta', 'Chlorophyta', 'Tracheophyta'}).astype(int)
    merged_df = enrich_with_biomes(merged_df)
    if merged_df.empty:
        print("No data after merging. Exiting.")
        return
    ct = pd.crosstab(merged_df['tipping_category'], merged_df['conservation_status'])
    print("[pre-analysis] Crosstab of tipping_category vs conservation_status:")
    print(ct)
    count_no_bif_threat = int(ct.loc['no-bifurcation', 'threatened']) if ('no-bifurcation' in ct.index and 'threatened' in ct.columns) else 0
    print(f"[pre-analysis] Count of no-bifurcation + threatened = {count_no_bif_threat}")
    firth_results = run_firth_analysis(merged_df, do_binary, outcome_name='frequentist_conservation')
    exact_results = {}
    if do_exact:
        exact_results = run_exact_logistic_binary(merged_df)
    resampling_results = {}
    permutation_results = {}
    if do_permutation:
        permutation_results = run_permutation_combos(merged_df, do_binary, n_iter=n_iter, seed=seed)
    if do_resampling:
        resampling_results = run_resampling_combos(merged_df, do_binary, n_iter=n_iter, seed=seed)
    else:
        print("\n[resampling] do_resampling=False - skipping resampling analysis.")
    print_validation_diagnostics(bif_df, merged_df, firth_results, resampling_results, permutation_results)
    write_comparison_csv(firth_results, resampling_results, permutation_results, exact_results)

def main():
    probability_threshold_local = probability_threshold
    do_exact = False
    do_binary = False
    do_resampling = False
    do_permutation = False
    n_iter = 10000
    seed = 1234
    run_analysis(probability_threshold_local, do_resampling, do_exact, do_permutation, do_binary, n_iter, seed)

if __name__ == "__main__":
    main()
