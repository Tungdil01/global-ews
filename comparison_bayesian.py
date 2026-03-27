import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
seed = 1234
np.random.seed(seed)
import pytensor
pytensor.config.rng_seed = seed

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14, 'axes.titlesize': 18, 'font.size': 14, 'lines.linewidth': 1.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 1.5
    })

def create_directories():
    os.makedirs('csv', exist_ok=True)
    os.makedirs('img', exist_ok=True)

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    parts = str(species_name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return species_name

def load_and_preprocess_bifurcation(threshold=0.3):
    df = pd.read_csv('csv/bifurcation_results_short.csv')
    df = df[df['transition_year'].notna()]
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['tipping_point_type'] = df.apply(
        lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no bifurcation', axis=1
    )
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation', 'no bifurcation']
    df = df[df['tipping_point_type'].isin(valid_types)]
    print(f"[bifurcation] Applied probability threshold {threshold} and kept species with transition_year:")
    print(f"- Total species: {len(df)}")
    print(f"- Tipping point distribution:")
    for tip_type in valid_types:
        count = (df['tipping_point_type'] == tip_type).sum()
        print(f"{tip_type}: {count} species")
    n_hopf_before = (df['tipping_point_type'] == 'Hopf bifurcation').sum()
    df = df[df['tipping_point_type'] != 'Hopf bifurcation']
    print(f"- Excluded {n_hopf_before} Hopf bifurcation species from analysis (uncertain label)")
    if 'taxonomic_group' not in df.columns:
        print("ERROR: 'taxonomic_group' column missing from bifurcation data. Cannot proceed with random effects analysis.")
        sys.exit(1)
    if df['taxonomic_group'].isna().all():
        print("ERROR: 'taxonomic_group' column contains only NaN values. Incorrectly loaded.")
        sys.exit(1)
    print(f"- Taxonomic groups available: {df['taxonomic_group'].nunique()}")
    return df

def load_and_preprocess_invasive():
    df = pd.read_csv('csv/invasive_species.csv', dtype=str)
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['invasive_raw'] = df['invasive_status'].astype(str).fillna('').str.strip()
    df['is_invasive'] = df['invasive_raw'] != 'Not Listed'
    dedup = df[['species_short', 'invasive_raw', 'is_invasive']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    total_unique = int(dedup['species_short'].nunique())
    n_not_listed = int((dedup['invasive_raw'] == 'Not Listed').sum())
    n_invasive_marked = int(dedup['is_invasive'].sum())
    print(f"[invasive] Loaded invasive CSV: unique species = {total_unique}; Not Listed (non-invasive) = {n_not_listed}; invasive = {n_invasive_marked}")
    return dedup

def load_and_preprocess_iucn():
    df = pd.read_csv('csv/iucn_species.csv', dtype=str)
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['IUCN_Status_raw'] = df['iucn_status'].astype(str).fillna('').str.strip()
    valid_statuses = ['LEAST_CONCERN', 'NEAR_THREATENED', 'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED']
    iucn_present = df[df['IUCN_Status_raw'].str.upper().str.strip().isin(valid_statuses)].copy()
    dedup = iucn_present[['species_short', 'IUCN_Status_raw']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    dedup['iucn_status_up'] = dedup['IUCN_Status_raw'].str.upper().str.strip()
    non_threat_mask = dedup['iucn_status_up'].isin(['LEAST_CONCERN', 'NEAR_THREATENED'])
    threatened_mask = dedup['iucn_status_up'].isin(['VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED'])
    total_iucn = int(len(dedup))
    n_non_threat = int(non_threat_mask.sum())
    n_threat = int(threatened_mask.sum())
    status_mapping = {
        'LEAST_CONCERN': 'LC', 'NEAR_THREATENED': 'NT', 'VULNERABLE': 'VU', 'ENDANGERED': 'EN', 'CRITICALLY_ENDANGERED': 'CR'
    }
    dedup['iucn_status'] = dedup['iucn_status_up'].map(status_mapping)
    dedup['is_threatened'] = threatened_mask.astype(int)
    out = dedup[['species_short', 'iucn_status', 'is_threatened']].copy()
    print(f"[threatened] Preprocessed {len(out)} species for conservation status analysis: non-threatened={n_non_threat}, threatened={n_threat}")
    return out

def merge_datasets(bifurcation_df, invasive_df, iucn_df):
    if 'taxonomic_group' in bifurcation_df.columns:
        per_species_tipping = bifurcation_df[['species_short', 'tipping_point_type', 'taxonomic_group']].copy()
    else:
        per_species_tipping = bifurcation_df[['species_short', 'tipping_point_type']].copy()
    per_species_tipping = per_species_tipping.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    invasive_unique = invasive_df[['species_short', 'invasive_raw', 'is_invasive']].copy()
    invasive_unique = invasive_unique.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    invasive_for_test = pd.merge(per_species_tipping, invasive_unique, on='species_short', how='inner')
    invasive_for_test['invasive_raw'] = invasive_for_test['invasive_raw'].fillna('').astype(str)
    invasive_for_test['invasive_status_known'] = invasive_for_test['invasive_raw'].str.strip() != ''
    invasive_for_test['is_invasive'] = invasive_for_test['is_invasive'].fillna(False).astype(bool)
    iucn_unique = iucn_df[['species_short', 'iucn_status', 'is_threatened']].copy()
    iucn_unique = iucn_unique.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    threatened_for_test = pd.merge(per_species_tipping, iucn_unique, on='species_short', how='inner')
    threatened_for_test['iucn_status_known'] = True
    print(f"Merged datasets - species with bifurcation: {len(per_species_tipping)}; invasive analysis: {len(invasive_for_test)}; IUCN analysis: {len(threatened_for_test)}")
    if 'taxonomic_group' in invasive_for_test.columns:
        print(f"Taxonomic groups in invasive analysis: {invasive_for_test['taxonomic_group'].nunique()}")
    if 'taxonomic_group' in threatened_for_test.columns:
        print(f"Taxonomic groups in threatened analysis: {threatened_for_test['taxonomic_group'].nunique()}")
    return invasive_for_test, threatened_for_test

def load_and_preprocess_biome():
    df = pd.read_csv('csv/biomes.csv')
    df['species_short'] = df['species'].apply(preprocess_species_name)
    dedup = df[['species_short', 'biome']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    if dedup.empty:
        print("ERROR: Biome data is empty. Cannot proceed.")
        sys.exit(1)
    if 'biome' not in dedup.columns:
        print("ERROR: 'biome' column missing from biome data.")
        sys.exit(1)
    if dedup['biome'].isna().all():
        print("ERROR: 'biome' column contains only NaN values.")
        sys.exit(1)
    print(f"[biome] Loaded {len(dedup)} species with biome information.")
    return dedup

def create_combined_dataset(invasive_data, threatened_data, biome_df):
    merged = pd.merge(invasive_data, threatened_data, on='species_short', how='inner', suffixes=('_inv', '_thr'))
    merged['tipping_point_type'] = merged['tipping_point_type_inv']
    if 'taxonomic_group_inv' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_inv']
    elif 'taxonomic_group_thr' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_thr']
    else:
        merged['taxonomic_group'] = np.nan
    merged['category'] = np.nan
    mask_inv = (merged['is_invasive'] == 1) & (merged['is_threatened'] == 0)
    mask_thr = (merged['is_invasive'] == 0) & (merged['is_threatened'] == 1)
    mask_neu = (merged['is_invasive'] == 0) & (merged['is_threatened'] == 0)
    merged.loc[mask_inv, 'category'] = 1
    merged.loc[mask_thr, 'category'] = 2
    merged.loc[mask_neu, 'category'] = 0
    merged = merged.dropna(subset=['category'])
    merged['category'] = merged['category'].astype(int)
    if biome_df is not None:
        merged = pd.merge(merged, biome_df, on='species_short', how='left')
    if 'taxonomic_group' not in merged.columns or merged['taxonomic_group'].isna().all():
        print("ERROR: 'taxonomic_group' missing or all NaN in combined dataset.")
        sys.exit(1)
    if 'biome' not in merged.columns or merged['biome'].isna().all():
        print("ERROR: 'biome' missing or all NaN in combined dataset.")
        sys.exit(1)
    print(f"[combined] Created combined dataset: {len(merged)} species")
    print(f"[combined] Category counts: neutral={sum(merged['category']==0)}, invasive={sum(merged['category']==1)}, threatened={sum(merged['category']==2)}")
    return merged

def prepare_multinomial_data(data_df, biome_df, outcome_col):
    data = data_df.copy()
    if biome_df is not None and 'biome' not in data.columns:
        data = pd.merge(data, biome_df, on='species_short', how='left')
    cat = data['tipping_point_type'].astype('category')
    all_types = ['no bifurcation', 'fold bifurcation', 'Hopf bifurcation', 'transcritical bifurcation']
    present_types = [t for t in all_types if t in cat.cat.categories]
    cat = cat.cat.reorder_categories(present_types, ordered=True)
    data['tipping_point_type'] = cat
    print(f"[prepare_multinomial_data] Tipping point counts for {outcome_col}:")
    print(data['tipping_point_type'].value_counts())
    dummies = pd.get_dummies(data['tipping_point_type'], prefix='tipping_point_type', drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    feature_cols = list(dummies.columns)
    return data, feature_cols

def compute_multinomial_summaries(trace, X_columns, category_names):
    summaries = {'effects': {}}
    posterior = trace.posterior
    beta_samples = posterior['beta'].values
    n_cat = beta_samples.shape[-2]
    for i, col in enumerate(X_columns):
        key = col
        summaries['effects'][key] = {}
        for k in range(n_cat):
            samples_beta = beta_samples[:, :, k, i].reshape(-1)
            samples_or = np.exp(samples_beta)
            mean_or = np.mean(samples_or)
            hdi_or = az.hdi(samples_or, hdi_prob=0.95)
            prob_dir = np.mean(samples_beta > 0)
            cat_label = category_names[k]
            summaries['effects'][key][cat_label] = {
                'odds_ratio': {'mean': mean_or, 'hdi_lower': hdi_or[0], 'hdi_upper': hdi_or[1]},
                'bayesian_evidence': {'probability_direction': prob_dir}
            }
    if 'sigma_group' in posterior:
        sigma_group = posterior['sigma_group'].values
        sigma_samples = sigma_group.reshape(-1)
        summaries['group_sd'] = {'mean': np.mean(sigma_samples), 'hdi': az.hdi(sigma_samples, hdi_prob=0.95).tolist()}
    return summaries

def run_multinomial_model_fixed(data, feature_cols, outcome_col, model_name='fixed'):
    y = data[outcome_col].astype(int).values
    X = data[feature_cols].values.astype(float)
    n_obs = len(data)
    n_predictors = X.shape[1]
    n_categories = len(np.unique(y))
    try:
        with pm.Model() as model:
            intercept = pm.Normal('intercept', mu=0, sigma=5, shape=n_categories-1)
            beta = pm.Normal('beta', mu=0, sigma=5, shape=(n_categories-1, n_predictors))
            eta = intercept + pm.math.dot(X, beta.T)
            exp_eta = pm.math.exp(eta)
            p0 = 1 / (1 + pm.math.sum(exp_eta, axis=-1, keepdims=True))
            p = pm.math.concatenate([p0, exp_eta * p0], axis=-1)
            y_obs = pm.Categorical('y_obs', p=p, observed=y)
            trace = pm.sample(draws=1000, tune=1000, cores=4, target_accept=0.95, return_inferencedata=True, random_seed=seed)
        category_names = ['neutral', 'invasive', 'threatened']
        summaries = compute_multinomial_summaries(trace, feature_cols, category_names[1:])
        error = None
    except Exception as e:
        trace = None
        summaries = None
        error = str(e)
        print(f"Error in {model_name}: {error}")
    result = {
        'outcome': outcome_col,
        'model': model_name,
        'trace': trace,
        'X_columns': feature_cols,
        'bayesian_summaries': summaries,
        'error': error
    }
    return result

def run_multinomial_model_hierarchical(data, feature_cols, outcome_col, group_col, model_name='hierarchical'):
    data_model = data.dropna(subset=[group_col]).copy()
    if len(data_model) < len(data):
        print(f"Dropped {len(data)-len(data_model)} rows with missing {group_col}")
    group_vals = data_model[group_col].astype('category')
    group_idx = group_vals.cat.codes.values
    n_groups = len(group_vals.cat.categories)
    y = data_model[outcome_col].astype(int).values
    X = data_model[feature_cols].values.astype(float)
    n_obs = len(data_model)
    n_predictors = X.shape[1]
    n_categories = len(np.unique(y))
    try:
        with pm.Model() as model:
            sigma_group = pm.HalfNormal('sigma_group', sigma=2, shape=n_categories-1)
            z_group = pm.Normal('z_group', mu=0, sigma=1, shape=(n_categories-1, n_groups))
            group_offset = z_group * sigma_group[:, None]
            intercept = pm.Normal('intercept', mu=0, sigma=5, shape=n_categories-1)
            beta = pm.Normal('beta', mu=0, sigma=5, shape=(n_categories-1, n_predictors))
            eta = intercept + pm.math.dot(X, beta.T) + group_offset[:, group_idx].T
            exp_eta = pm.math.exp(eta)
            p0 = 1 / (1 + pm.math.sum(exp_eta, axis=-1, keepdims=True))
            p = pm.math.concatenate([p0, exp_eta * p0], axis=-1)
            y_obs = pm.Categorical('y_obs', p=p, observed=y)
            trace = pm.sample(draws=1000, tune=1000, cores=4, target_accept=0.98, return_inferencedata=True, random_seed=seed)
        category_names = ['neutral', 'invasive', 'threatened']
        summaries = compute_multinomial_summaries(trace, feature_cols, category_names[1:])
        error = None
    except Exception as e:
        trace = None
        summaries = None
        error = str(e)
        print(f"Error in {model_name}: {error}")
    result = {
        'outcome': outcome_col,
        'model': model_name,
        'trace': trace,
        'X_columns': feature_cols,
        'bayesian_summaries': summaries,
        'error': error
    }
    return result

def create_multinomial_inference_figure(bayesian_results, outcome_name, model_name):
    set_plot_style()
    result = bayesian_results[(bayesian_results['outcome'] == outcome_name) & (bayesian_results['model'] == model_name)]
    if len(result) == 0:
        print("No result found for this outcome/model combination")
        return
    result = result.iloc[0]
    if 'error' in result and result['error'] is not None:
        print(f"Skipping due to error: {result['error']}")
        return
    is_fixed = 'fixed' in model_name
    if is_fixed:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if 'bayesian_summaries' in result and result['bayesian_summaries'] is not None:
        summaries = result['bayesian_summaries']
        if 'effects' in summaries and summaries['effects']:
            effect_data = {}
            coef_names = []
            for coef_key, cat_effects in summaries['effects'].items():
                if isinstance(cat_effects, dict):
                    clean_name = coef_key.replace('tipping_point_type_', '').replace('_', ' ')
                    clean_name = clean_name.replace('bifurcation', '').replace('transcritical', 'Transc.').replace('Hopf', 'Hopf').replace('fold', 'Fold').strip()
                    coef_names.append(clean_name)
                    for cat, vals in cat_effects.items():
                        if cat not in effect_data:
                            effect_data[cat] = []
                        or_mean = vals.get('odds_ratio', {}).get('mean', 1.0)
                        or_lower = vals.get('odds_ratio', {}).get('hdi_lower', 1.0)
                        or_upper = vals.get('odds_ratio', {}).get('hdi_upper', 1.0)
                        effect_data[cat].append((or_mean, or_lower, or_upper))
            if effect_data:
                cats = list(effect_data.keys())
                y_pos = np.arange(len(coef_names))
                colors_cat = {'invasive': '#1f77b4', 'threatened': '#d62728'}
                offsets = {'invasive': 0.2, 'threatened': -0.2}
                for cat in cats:
                    data = effect_data[cat]
                    for i, (mean, low, high) in enumerate(data):
                        y = y_pos[i] + offsets[cat]
                        axes[0].plot(mean, y, 'o', color=colors_cat[cat], markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=cat if i==0 else "")
                        axes[0].hlines(y, low, high, color='black', linewidth=1.5)
                axes[0].axvline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                axes[0].set_yticks(y_pos)
                axes[0].set_yticklabels(coef_names)
                axes[0].set_xlabel('Odds Ratio (relative to neutral)')
                axes[0].legend(loc='lower right')
                max_val = max([max([d[2] for d in effect_data[cat]]) for cat in cats]) if cats else 2.0
                axes[0].set_xlim(0, max(max_val * 1.1, 2.0))
                axes[0].set_ylim(-0.5, len(coef_names) - 0.5)
            else:
                axes[0].text(0.5, 0.5, 'No odds ratio data available', ha='center', va='center', transform=axes[0].transAxes)
        else:
            axes[0].text(0.5, 0.5, 'No effects data in summaries', ha='center', va='center', transform=axes[0].transAxes)
        if not is_fixed and len(axes) > 1:
            if 'hierarchical' in model_name and 'trace' in result and result['trace'] is not None and 'sigma_group' in result['trace'].posterior:
                trace = result['trace']
                sigma_group = trace.posterior['sigma_group'].values.reshape(-1, trace.posterior['sigma_group'].shape[-1])
                pi2_3 = (np.pi**2)/3
                icc_text = ''
                for k in range(sigma_group.shape[1]):
                    sigma2 = sigma_group[:, k]**2
                    icc = sigma2 / (sigma2 + pi2_3)
                    mean_icc = np.mean(icc)
                    hdi_icc = az.hdi(icc, hdi_prob=0.95)
                    label = ['invasive','threatened'][k]
                    axes[1].hist(sigma_group[:, k], bins=40, alpha=0.5, edgecolor='black', density=True, label=label)
                    icc_text += f'ICC {label}: {mean_icc:.2g}\n' #  [{hdi_icc[0]:.2g}, {hdi_icc[1]:.2g}]
                axes[1].set_xlabel('Group SD')
                axes[1].set_ylabel('Density')
                axes[1].legend()
                axes[1].text(0.9, 0.7, icc_text.rstrip(), transform=axes[1].transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                axes[1].text(0.5, 0.5, 'No hierarchical parameters', ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'No bayesian summaries', ha='center', va='center', transform=axes[0].transAxes)
        if len(axes) > 1:
            axes[1].text(0.5, 0.5, 'No bayesian summaries', ha='center', va='center', transform=axes[1].transAxes)
    plt.tight_layout()
    plt.savefig(f'img/inference_{model_name}_{outcome_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def create_multinomial_posterior_plot(bayesian_results, outcome_name, model_name):
    set_plot_style()
    result = bayesian_results[(bayesian_results['outcome'] == outcome_name) & (bayesian_results['model'] == model_name)]
    if len(result) == 0:
        print("No result found for this outcome/model combination")
        return
    result = result.iloc[0]
    if 'error' in result and result['error'] is not None:
        print(f"Skipping due to error: {result['error']}")
        return
    if 'trace' not in result or result['trace'] is None:
        print("No trace available")
        return
    trace = result['trace']; X_columns = result.get('X_columns', [])
    tipping_params = [col for col in X_columns if 'tipping_point_type' in col]
    if not tipping_params:
        print("No tipping point parameters found")
        return
    n_plots = len(tipping_params)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    plot_idx = 0
    posterior = trace.posterior
    beta_vals = posterior['beta'].values
    n_cat = beta_vals.shape[2]
    cat_names = ['invasive', 'threatened']
    for param_name in tipping_params:
        param_idx = list(X_columns).index(param_name)
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]
        for k in range(n_cat):
            samples = beta_vals[:, :, k, param_idx].reshape(-1)
            ax.hist(samples, bins=50, alpha=0.5, density=True, label=cat_names[k], edgecolor='black')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Log-Odds Coefficient')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left', fontsize=10)
        plot_idx += 1
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'img/posterior_distributions_{outcome_name}_{model_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def create_predicted_probability_plot(inv_df, iucn_df, outcome_name, threshold):
    set_plot_style()
    full_bif = pd.read_csv('csv/bifurcation_results.csv')
    full_bif = full_bif[full_bif['transition_year'].notna()].copy()
    full_bif['species_short'] = full_bif['species'].apply(preprocess_species_name)
    full_bif['tipping_point_type'] = full_bif.apply(lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no bifurcation', axis=1)
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation', 'no bifurcation']
    full_bif = full_bif[full_bif['tipping_point_type'].isin(valid_types)]
    if outcome_name == 'invasive':
        status_df = inv_df[['species_short', 'is_invasive']].copy()
        outcome_col = 'is_invasive'
        title_suffix = 'Invasive'
    else:
        status_df = iucn_df[['species_short', 'is_threatened']].copy()
        outcome_col = 'is_threatened'
        title_suffix = 'Threatened'
    merged = pd.merge(full_bif[['species_short', 'tipping_point_type']], status_df, on='species_short', how='inner')
    merged = merged.dropna(subset=[outcome_col, 'tipping_point_type']).copy()
    obs_data = merged.groupby('tipping_point_type').agg({outcome_col: ['mean', 'sum', 'count']}).round(3)
    obs_data.columns = ['mean', 'positive', 'total']
    tipping_order = ['transcritical bifurcation', 'fold bifurcation', 'Hopf bifurcation']
    tipping_colors = ['#FF8C00', '#1E90FF', '#9370DB']
    tipping_abbr = ['Transc. bif.', 'Fold bif.', 'Hopf bif.']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    means = [obs_data['mean'].get(tip, 0) for tip in tipping_order]
    positives = [int(obs_data['positive'].get(tip, 0)) for tip in tipping_order]
    totals = [int(obs_data['total'].get(tip, 0)) for tip in tipping_order]
    bars = ax.bar(tipping_abbr, means, color=tipping_colors, alpha=0.7)
    for i, (bar, positive, total) in enumerate(zip(bars, positives, totals)):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height/2, f'{positive}/{total}', ha='center', va='center', fontweight='bold', color='black')
    ax.set_ylabel(f'Proportion {title_suffix}')
    ax.set_ylim(0, max(means) * 1.3 if means else 1)
    total_positive = merged[outcome_col].sum()
    total_obs = len(merged)
    ax.text(0.02, 0.98, f'Total: {total_positive}/{total_obs}', transform=ax.transAxes, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'img/predicted_probabilities_{outcome_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def run_analysis(probability_threshold):
    bif_df = load_and_preprocess_bifurcation(threshold=probability_threshold)
    inv_df = load_and_preprocess_invasive()
    iucn_df = load_and_preprocess_iucn()
    biome_df = load_and_preprocess_biome()
    invasive_data, threatened_data = merge_datasets(bif_df, inv_df, iucn_df)
    create_predicted_probability_plot(inv_df, iucn_df, 'invasive', probability_threshold)
    create_predicted_probability_plot(inv_df, iucn_df, 'threatened', probability_threshold)
    combined_data = create_combined_dataset(invasive_data, threatened_data, biome_df)
    bayesian_results = []
    data_comb, feat_comb = prepare_multinomial_data(combined_data, biome_df, 'category')
    res_fixed = run_multinomial_model_fixed(data_comb, feat_comb, 'category', model_name='fixed_multinomial')
    bayesian_results.append(res_fixed)
    if 'taxonomic_group' in data_comb.columns:
        res_hier_tax = run_multinomial_model_hierarchical(data_comb, feat_comb, 'category', group_col='taxonomic_group', model_name='hierarchical_taxonomic_multinomial')
        bayesian_results.append(res_hier_tax)
    if 'biome' in data_comb.columns:
        res_hier_bio = run_multinomial_model_hierarchical(data_comb, feat_comb, 'category', group_col='biome', model_name='hierarchical_biome_multinomial')
        bayesian_results.append(res_hier_bio)
    bayesian_df = pd.DataFrame(bayesian_results)
    if bayesian_df.empty:
        print("No Bayesian results to plot.")
    for idx, row in bayesian_df.iterrows():
        create_multinomial_inference_figure(bayesian_df, row['outcome'], row['model'])
        create_multinomial_posterior_plot(bayesian_df, row['outcome'], row['model'])

def main():
    probability_thresholds = [0.5]
    create_directories()
    for thresh in probability_thresholds:
        os.makedirs(f'img', exist_ok=True)
        run_analysis(thresh)

if __name__ == "__main__":
    main()
