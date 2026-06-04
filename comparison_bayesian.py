import os, sys
import itertools
from itertools import combinations
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
#import pytensor.tensor as pt
seed = 1234
import pytensor
pytensor.config.rng_seed = seed
np.random.seed(seed)

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 18, 
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16, 
        'legend.fontsize': 14, 
        'axes.titlesize': 18, 
        'font.size': 14, 
        'lines.linewidth': 1.5, 
        'figure.facecolor': 'white', 
        'axes.facecolor': 'white', 
        'axes.edgecolor': 'black', 
        'axes.linewidth': 1.5
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
        lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no-bifurcation', axis=1
    )
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation', 'no-bifurcation']
    df = df[df['tipping_point_type'].isin(valid_types)]
    print(f"[bifurcation] Applied probability threshold {threshold} and kept species with transition_year:")
    print(f"- Total species: {len(df)}")
    print(f"- Tipping point distribution:")
    for tip_type in valid_types:
        count = (df['tipping_point_type'] == tip_type).sum()
        print(f"{tip_type}: {count} species")
    if 'taxonomic_group' not in df.columns:
        print("ERROR: 'taxonomic_group' column missing from bifurcation data. Cannot proceed with random effects analysis.")
        sys.exit(1)
    if df['taxonomic_group'].isna().all():
        print("ERROR: 'taxonomic_group' column contains only NaN values. Incorrectly loaded.")
        sys.exit(1)
    print(f"- Taxonomic groups available: {df['taxonomic_group'].nunique()}")
    return df

def collapse_taxonomic_group(series):
    mapping = {
        'Elasmobranchii': 'Pisces',
        'Malacostraca': 'Crustacea', 'Maxillopoda': 'Crustacea', 'Copepoda': 'Crustacea', 'Thecostraca': 'Crustacea',
        'Chloroplastida': 'Plantae', 'Tracheophyta': 'Plantae',
        'Myzozoa': 'Algae', 'Ochrophyta': 'Algae', 'Rhodophyta': 'Algae',
        'Bacteria': 'Microbiota', 'Fungi': 'Microbiota',
        'Brachiopoda': 'OtherInvertebrates', 'Bryozoa': 'OtherInvertebrates', 'Chelicerata': 'OtherInvertebrates',
        'Rotifera': 'OtherInvertebrates', 'Tunicata': 'OtherInvertebrates'
    }
    return series.replace(mapping)

def collapse_biome(series):
    mapping = {
        'Large Lake Ecosystems': 'LakeEcosystems', 'Small Lake Ecosystems': 'LakeEcosystems',
        'Large River Ecosystems': 'RiverEcosystems', 'Small River Ecosystems': 'RiverEcosystems',
        'Temperate Broadleaf & Mixed Forests': 'Unknown/Unclassified',
        'Temperate Coniferous Forests': 'Unknown/Unclassified',
        'Temperate Grasslands, Savannas & Shrublands': 'Unknown/Unclassified',
        'Tropical Coral Reefs': 'Temperate Shelf & Seas',
        'Mixed/Generalist Habitats': 'Unknown/Unclassified'
    }
    return series.replace(mapping)

def collapse_koppen(series):
    mapping = {
        'Am': 'Af',
        'Cwa': 'Cfa',
        'Cfc': 'Cfb',
        'Dfc': 'Dfb',
        'BSh': 'B', 'BSk': 'B', 'BWh': 'B',
        'EF': 'E', 'ET': 'E'
    }
    return series.replace(mapping)

def collapse_meow(series):
    mapping = {
        'Southern Ocean': 'OtherMeow',
        'Western Indo-Pacific': 'OtherMeow',
        'Temperate South America': 'OtherMeow'
    }
    return series.replace(mapping)

def apply_all_collapses(data):
    data['taxonomic_group'] = collapse_taxonomic_group(data['taxonomic_group'])
    data['biome'] = collapse_biome(data['biome'])
    data['koppen_zone'] = collapse_koppen(data['koppen_zone'])
    data['meow_zone'] = collapse_meow(data['meow_zone'])
    return data

def prepare_multicategory_data(bif_df, biome_df, koppen_df, meow_df):
    data = bif_df[['species_short', 'tipping_point_type', 'taxonomic_group']].copy()
    if biome_df is not None and not biome_df.empty:
        data = pd.merge(data, biome_df, on='species_short', how='left')
    if koppen_df is not None and not koppen_df.empty:
        data = pd.merge(data, koppen_df, on='species_short', how='left')
    if meow_df is not None and not meow_df.empty:
        data = pd.merge(data, meow_df, on='species_short', how='left')
    data = data.dropna(subset=['tipping_point_type', 'taxonomic_group', 'biome', 'koppen_zone', 'meow_zone'])
    data = apply_all_collapses(data)
    return data

def abbreviate_category_name(col_name):
    mapping = {
        'tipping_point_type_fold bifurcation': 'fold bif.',
        'tipping_point_type_transcritical bifurcation': 'transc. bif.',
        'tipping_point_type_Hopf bifurcation': 'Hopf bif.',
        'tipping_point_type_no-bifurcation': 'no-bif.',
    }
    if col_name in mapping:
        return mapping[col_name]
    if '_x_' in col_name:
        parts = col_name.split('_x_')
        abbr_parts = [abbreviate_category_name(p) for p in parts]
        return ' x '.join(abbr_parts)
    if col_name.startswith('tip_'):
        col_name = col_name[4:]
    elif col_name.startswith('taxonomic_group_'):
        col_name = col_name[16:]
    elif col_name.startswith('biome_'):
        col_name = col_name[6:]
    elif col_name.startswith('koppen_zone_'):
        col_name = col_name[12:]
    elif col_name.startswith('meow_zone_'):
        col_name = col_name[10:]
    biome_map = {
        'Large Lake Ecosystems': 'Large Lake',
        'Large River Ecosystems': 'Large River',
        'Mixed/Generalist Habitats': 'Mixed/Gen.',
        'Polar Ecoregions': 'Polar',
        'Small Lake Ecosystems': 'Small Lake',
        'Small River Ecosystems': 'Small River',
        'Temperate Broadleaf & Mixed Forests': 'Temp. Broadleaf',
        'Temperate Coniferous Forests': 'Temp. Conifer',
        'Temperate Grasslands, Savannas & Shrublands': 'Temp. Grassland',
        'Temperate Shelf & Seas': 'Temp. Shelf',
        'Tropical & Subtropical Grasslands, Savannas & Shrublands': 'Trop. Grassland',
        'Tropical Coral Reefs': 'Trop. Coral',
        'Unknown/Unclassified': 'Unknown',
    }
    if col_name in biome_map:
        return biome_map[col_name]
    meow_map = {
        'Arctic': 'Arctic',
        'Central Indo-Pacific': 'C. Indo-Pac',
        'Southern Ocean': 'S. Ocean',
        'Temperate Australasia': 'Temp. Aus.',
        'Temperate Northern Atlantic': 'Temp. N. Atl.',
        'Temperate Northern Pacific': 'Temp. N. Pac.',
        'Temperate South America': 'Temp. S. Am.',
        'Tropical Atlantic': 'Trop. Atl.',
        'Western Indo-Pacific': 'W. Indo-Pac',
    }
    if col_name in meow_map:
        return meow_map[col_name]
    if col_name == 'fold bifurcation':
        return 'fold bif.'
    elif col_name == 'transcritical bifurcation':
        return 'transc. bif.'
    elif col_name == 'Hopf bifurcation':
        return 'Hopf bif.'
    elif col_name == 'no-bifurcation':
        return 'no-bif.'
    return col_name

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

def merge_datasets(bifurcation_df, iucn_df):
    if 'taxonomic_group' in bifurcation_df.columns:
        per_species_tipping = bifurcation_df[['species_short', 'tipping_point_type', 'taxonomic_group']].copy()
    else:
        per_species_tipping = bifurcation_df[['species_short', 'tipping_point_type']].copy()
    per_species_tipping = per_species_tipping.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    iucn_unique = iucn_df[['species_short', 'iucn_status', 'is_threatened']].copy()
    iucn_unique = iucn_unique.drop_duplicates(subset=['species_short']).reset_index(drop=True)
    threatened_for_test = pd.merge(per_species_tipping, iucn_unique, on='species_short', how='inner')
    threatened_for_test['iucn_status_known'] = True
    print(f"Merged datasets - species with bifurcation: {len(per_species_tipping)}; threatened analysis: {len(threatened_for_test)}")
    if 'taxonomic_group' in threatened_for_test.columns:
        print(f"Taxonomic groups in threatened analysis: {threatened_for_test['taxonomic_group'].nunique()}")
    # Print threatened and total counts per tipping point type
    type_counts = threatened_for_test.groupby('tipping_point_type')['is_threatened'].agg(['sum', 'count'])
    type_counts.columns = ['threatened', 'total']
    print("\nTipping point type counts for threatened analysis:")
    order = ['no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    for tip in order:
        if tip in type_counts.index:
            thr = int(type_counts.loc[tip, 'threatened'])
            tot = int(type_counts.loc[tip, 'total'])
            print(f"{tip:<30} {thr}/{tot}")
    return threatened_for_test

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

KOPPEN_CODE_TO_ABBR = {
    11: 'Af', 12: 'Am', 13: 'As', 14: 'Aw',
    21: 'BWk', 22: 'BWh', 26: 'BSk', 27: 'BSh',
    31: 'Cfa', 32: 'Cfb', 33: 'Cfc', 34: 'Csa', 35: 'Csb', 36: 'Csc', 37: 'Cwa', 38: 'Cwb', 39: 'Cwc',
    41: 'Dfa', 42: 'Dfb', 43: 'Dfc', 44: 'Dfd', 45: 'Dsa', 46: 'Dsb', 47: 'Dsc', 48: 'Dsd', 49: 'Dwa',
    50: 'Dwb', 51: 'Dwc', 52: 'Dwd', 61: 'EF', 62: 'ET'
}

def convert_koppen_code(code):
    if pd.isna(code):
        return np.nan
    try:
        int_code = int(float(code))
        return KOPPEN_CODE_TO_ABBR.get(int_code, code)
    except (ValueError, TypeError):
        return code

def load_and_preprocess_koppen():
    df = pd.read_csv('csv/biomes.csv')
    df['species_short'] = df['species'].apply(preprocess_species_name)
    if 'koppen_zone' not in df.columns:
        print("WARNING: 'koppen_zone' column not found in biomes.csv. Skipping Koppen analysis.")
        return pd.DataFrame(columns=['species_short', 'koppen_zone'])
    dedup = df[['species_short', 'koppen_zone']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    if dedup.empty:
        print("WARNING: No Koppen zone data available.")
        return pd.DataFrame(columns=['species_short', 'koppen_zone'])
    dedup['koppen_zone'] = dedup['koppen_zone'].apply(convert_koppen_code)
    if dedup['koppen_zone'].isna().all():
        print("WARNING: 'koppen_zone' column contains only NaN values.")
        return pd.DataFrame(columns=['species_short', 'koppen_zone'])
    print(f"[koppen] Loaded {len(dedup)} species with Koppen zone information.")
    return dedup

def load_and_preprocess_meow():
    df = pd.read_csv('csv/biomes.csv')
    df['species_short'] = df['species'].apply(preprocess_species_name)
    if 'meow_zone' not in df.columns:
        print("WARNING: 'meow_zone' column not found in biomes.csv. Skipping MEOW analysis.")
        return pd.DataFrame(columns=['species_short', 'meow_zone'])
    dedup = df[['species_short', 'meow_zone']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    if dedup.empty:
        print("WARNING: No MEOW zone data available.")
        return pd.DataFrame(columns=['species_short', 'meow_zone'])
    if dedup['meow_zone'].isna().all():
        print("WARNING: 'meow_zone' column contains only NaN values.")
        return pd.DataFrame(columns=['species_short', 'meow_zone'])
    print(f"[meow] Loaded {len(dedup)} species with MEOW zone information.")
    return dedup

def observed_proportions_plot(iucn_df, outcome_name, threshold):
    set_plot_style()
    full_bif = pd.read_csv('csv/bifurcation_results_short.csv')
    full_bif = full_bif[full_bif['transition_year'].notna()].copy()
    full_bif['species_short'] = full_bif['species'].apply(preprocess_species_name)
    full_bif['tipping_point_type'] = full_bif.apply(lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no-bifurcation', axis=1)
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation', 'no-bifurcation']
    full_bif = full_bif[full_bif['tipping_point_type'].isin(valid_types)]
    status_df = iucn_df[['species_short', 'is_threatened']].copy()
    outcome_col = 'is_threatened'
    title_suffix = 'Threatened'
    merged = pd.merge(full_bif[['species_short', 'tipping_point_type']], status_df, on='species_short', how='inner')
    merged = merged.dropna(subset=[outcome_col, 'tipping_point_type']).copy()
    obs_data = merged.groupby('tipping_point_type').agg({outcome_col: ['mean', 'sum', 'count']}).round(3)
    obs_data.columns = ['mean', 'positive', 'total']
    tipping_order = ['transcritical bifurcation', 'fold bifurcation', 'Hopf bifurcation', 'no-bifurcation']
    tipping_colors = {
        'fold bifurcation': '#1f77b4',
        'transcritical bifurcation': '#ff7f0e',
        'Hopf bifurcation': '#9467bd',
        'no-bifurcation': '#999999',
    }
    tipping_abbr = [abbreviate_category_name(t) for t in tipping_order]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    means = [obs_data['mean'].get(tip, 0) for tip in tipping_order]
    positives = [int(obs_data['positive'].get(tip, 0)) for tip in tipping_order]
    totals = [int(obs_data['total'].get(tip, 0)) for tip in tipping_order]
    bars = ax.bar(tipping_abbr, means, color=[tipping_colors[tip] for tip in tipping_order], alpha=0.7)
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
    plt.savefig(f'img/observed_proportions_{outcome_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

##### Bayesian logistic regression (Bernoulli likelihood with logit link)
def prepare_binary_data(data_df, biome_df, outcome_col):
    data = data_df.copy()
    if biome_df is not None and 'biome' not in data.columns:
        data = pd.merge(data, biome_df, on='species_short', how='left')
    cat = data['tipping_point_type'].astype('category')
    all_types = ['no-bifurcation', 'fold bifurcation', 'Hopf bifurcation', 'transcritical bifurcation']
    present_types = [t for t in all_types if t in cat.cat.categories]
    cat = cat.cat.reorder_categories(present_types, ordered=True)
    data['tipping_point_type'] = cat
    print(f"[prepare_binary_data] Tipping point counts for {outcome_col}:")
    print(data['tipping_point_type'].value_counts())
    dummies = pd.get_dummies(data['tipping_point_type'], prefix='tipping_point_type', drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    feature_cols = list(dummies.columns)
    return data, feature_cols

def compute_binary_summaries(trace, X_columns):
    summaries = {'effects': {}}
    posterior = trace.posterior
    beta_samples = posterior['beta'].values
    n_predictors = beta_samples.shape[-1]
    for i, col in enumerate(X_columns):
        key = abbreviate_category_name(col)
        samples_beta = beta_samples[:, :, i].reshape(-1)
        samples_or = np.exp(samples_beta)
        mean_or = np.mean(samples_or)
        hdi_or = az.hdi(samples_or, hdi_prob=0.95)
        prob_dir = np.mean(samples_beta > 0)
        summaries['effects'][key] = {
            'odds_ratio': {'mean': mean_or, 'hdi_lower': hdi_or[0], 'hdi_upper': hdi_or[1]},
            'bayesian_evidence': {'probability_direction': prob_dir}
        }
    if 'sigma_group' in posterior:
        sigma_group = posterior['sigma_group'].values
        sigma_samples = sigma_group.reshape(-1)
        summaries['group_sd'] = {'mean': np.mean(sigma_samples), 'hdi': az.hdi(sigma_samples, hdi_prob=0.95).tolist()}
    return summaries

def run_binary_model_fixed(data, feature_cols, outcome_col, model_name='fixed'):
    y = data[outcome_col].astype(int).values
    X = data[feature_cols].values.astype(float)
    n_obs = len(data)
    n_predictors = X.shape[1]
    try:
        with pm.Model() as model:
            intercept = pm.Normal('intercept', mu=0, sigma=5)
            beta = pm.Normal('beta', mu=0, sigma=5, shape=n_predictors)
            logit_p = intercept + pm.math.dot(X, beta)
            y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)
            trace = pm.sample(draws=2000, tune=2000, cores=4, target_accept=0.95, return_inferencedata=True, random_seed=seed)
        check_convergence(trace, model_name)
        summaries = compute_binary_summaries(trace, feature_cols)
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

def run_binary_model_hierarchical(data, feature_cols, outcome_col, group_col, model_name='hierarchical'):
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
    sigma_beta = 5
    sigma_group_prior = 2
    target_accept = 0.95
    draws = 2000
    tune = 2000
    try:
        with pm.Model() as model:
            sigma_group = pm.HalfNormal('sigma_group', sigma=sigma_group_prior)
            z_group = pm.Normal('z_group', mu=0, sigma=1, shape=n_groups)
            group_offset = z_group * sigma_group
            intercept = pm.Normal('intercept', mu=0, sigma=5)
            beta = pm.Normal('beta', mu=0, sigma=sigma_beta, shape=n_predictors)
            logit_p = intercept + pm.math.dot(X, beta) + group_offset[group_idx]
            y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)
            step = pm.NUTS(target_accept=target_accept, max_treedepth=100)
            trace = pm.sample(draws=draws, tune=tune, cores=4, step=step, return_inferencedata=True, random_seed=seed)
        check_convergence(trace, model_name)
        summaries = compute_binary_summaries(trace, feature_cols)
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

def create_binary_inference_figure(bayesian_results, outcome_name, model_name):
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if 'bayesian_summaries' in result and result['bayesian_summaries'] is not None:
        summaries = result['bayesian_summaries']
        if 'effects' in summaries and summaries['effects']:
            effect_data = []
            coef_names = []
            for coef_key, vals in summaries['effects'].items():
                coef_names.append(coef_key)
                or_mean = vals.get('odds_ratio', {}).get('mean', 1.0)
                or_lower = vals.get('odds_ratio', {}).get('hdi_lower', 1.0)
                or_upper = vals.get('odds_ratio', {}).get('hdi_upper', 1.0)
                effect_data.append((or_mean, or_lower, or_upper))
            if effect_data:
                y_pos = np.arange(len(coef_names))
                for i, (mean, low, high) in enumerate(effect_data):
                    axes[0].plot(mean, y_pos[i], 'o', color='#1f77b4', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
                    axes[0].hlines(y_pos[i], low, high, color='black', linewidth=1.5)
                axes[0].axvline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                axes[0].set_yticks(y_pos)
                axes[0].set_yticklabels(coef_names)
                axes[0].set_xlabel('Odds Ratio')
                axes[0].set_xlim(0, 5)
                axes[0].set_ylim(-0.5, len(coef_names) - 0.5)
            else:
                axes[0].text(0.5, 0.5, 'No odds ratio data available', ha='center', va='center', transform=axes[0].transAxes)
        else:
            axes[0].text(0.5, 0.5, 'No effects data in summaries', ha='center', va='center', transform=axes[0].transAxes)
        if not is_fixed and len(axes) > 1:
            if 'hierarchical' in model_name and 'trace' in result and result['trace'] is not None and 'sigma_group' in result['trace'].posterior:
                trace = result['trace']
                sigma_group = trace.posterior['sigma_group'].values.reshape(-1)
                axes[1].hist(sigma_group, bins=40, alpha=0.5, edgecolor='black', density=True)
                axes[1].set_xlabel('Group SD')
                axes[1].set_ylabel('Density')
                pi2_3 = (np.pi**2)/3
                sigma2 = sigma_group**2
                icc = sigma2 / (sigma2 + pi2_3)
                mean_icc = np.mean(icc)
                hdi_icc = az.hdi(icc, hdi_prob=0.95)
                axes[1].text(0.9, 0.7, f'ICC mean: {mean_icc:.2g} [{hdi_icc[0]:.2g}, {hdi_icc[1]:.2g}]', transform=axes[1].transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)) # , fontsize=10
            else:
                axes[1].text(0.5, 0.5, 'No hierarchical parameters', ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'No bayesian summaries', ha='center', va='center', transform=axes[0].transAxes)
        if len(axes) > 1:
            axes[1].text(0.5, 0.5, 'No bayesian summaries', ha='center', va='center', transform=axes[1].transAxes)
    plt.tight_layout()
    plt.savefig(f'img/inference_{model_name}_{outcome_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def create_binary_posterior_plot(bayesian_results, outcome_name, model_name):
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
    for param_name in tipping_params:
        param_idx = list(X_columns).index(param_name)
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]
        samples = beta_vals[:, :, param_idx].reshape(-1)
        ax.hist(samples, bins=50, alpha=0.5, density=True, edgecolor='black')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Log-Odds Coefficient')
        ax.set_ylabel('Density')
        abbr = abbreviate_category_name(param_name)
        ax.set_title(f'{abbr} posterior')
        plot_idx += 1
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'img/posterior_distributions_{outcome_name}_{model_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_binary_summary(result):
    if result['error'] is not None:
        print(f"Model {result['model']} failed: {result['error']}")
        return
    print(f"\n---------- {result['model']} (outcome: {result['outcome']}) ----------")
    summaries = result['bayesian_summaries']
    if summaries is None or 'effects' not in summaries:
        print("No effect summaries available.")
        return
    for coef, vals in summaries['effects'].items():
        or_mean = vals['odds_ratio']['mean']
        or_low = vals['odds_ratio']['hdi_lower']
        or_up = vals['odds_ratio']['hdi_upper']
        prob_dir = vals['bayesian_evidence']['probability_direction']
        coef_name = coef.replace('tipping_point_type_', '').replace('_', ' ')
        coef_name = coef_name.replace('bifurcation', '').replace('transcritical', 'Transc.').replace('Hopf', 'Hopf').replace('fold', 'Fold').strip()
        print(f"{coef_name}: OR = {or_mean:.2g} [{or_low:.2g}, {or_up:.2g}], P(beta>0) = {prob_dir:.2g}")
    if 'group_sd' in summaries:
        gsd = summaries['group_sd']
        print(f"Group SD: mean = {gsd['mean']:.2g}, 95% HDI [{gsd['hdi'][0]:.2g}, {gsd['hdi'][1]:.2g}]")
    print("-----------------------------------------\n")

# ==================== MULTINOMIAL LOGISTIC ====================

def format_num(x, sig=2):
    s = f"{x:.{sig}g}"
    if 'e' in s or 'E' in s:
        s = f"{x:.{sig}f}"
        s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s

def check_convergence(trace, model_name, rhat_threshold=1.05):
    rhat_ds = az.rhat(trace, method='rank')
    rhat_da = rhat_ds.to_array()
    max_rhat = float(rhat_da.max().values)
    ess_ds = az.ess(trace)
    ess_da = ess_ds.to_array()
    min_ess = float(ess_da.min().values)
    print(f"[convergence] {model_name}: max R-hat = {format_num(max_rhat, 2)}, min ESS = {format_num(min_ess, 2)}")
    if max_rhat > rhat_threshold:
        raise RuntimeError(f"Convergence failed: max R-hat = {format_num(max_rhat, 2)} > {rhat_threshold}")
    if min_ess < 400:
        raise RuntimeError(f"Convergence failed: min ESS = {format_num(min_ess, 2)} < 400")

def prepare_multinomial_data(data, outcome_col, predictor_cols):
    data = data.dropna(subset=[outcome_col] + predictor_cols).copy()
    if 'taxonomic_group' in predictor_cols:
        tax_vals = data['taxonomic_group'].astype(str)
        if 'Other' in tax_vals.values:
            other_cats = sorted([c for c in tax_vals.unique() if c != 'Other'])
            tax_cats = ['Other'] + other_cats
        else:
            tax_cats = sorted(tax_vals.unique())
        data['taxonomic_group'] = pd.Categorical(tax_vals, categories=tax_cats, ordered=True)
    if 'biome' in predictor_cols:
        biome_vals = data['biome'].astype(str)
        uniq = biome_vals.unique()
        if 'Unknown/Unclassified' in uniq:
            other_biome = sorted([c for c in uniq if c != 'Unknown/Unclassified'])
            biome_cats = ['Unknown/Unclassified'] + other_biome
        else:
            biome_cats = sorted(uniq)
        data['biome'] = pd.Categorical(biome_vals, categories=biome_cats, ordered=True)
    if 'koppen_zone' in predictor_cols:
        mode_cat = data['koppen_zone'].value_counts().idxmax()
        other_cats = sorted([c for c in data['koppen_zone'].unique() if c != mode_cat])
        koppen_cats = [mode_cat] + other_cats
        data['koppen_zone'] = pd.Categorical(data['koppen_zone'], categories=koppen_cats, ordered=True)
    if 'meow_zone' in predictor_cols:
        meow_vals = data['meow_zone'].astype(str)
        if 'OtherMeow' in meow_vals.values:
            other_meow = sorted([c for c in meow_vals.unique() if c != 'OtherMeow'])
            meow_cats = ['OtherMeow'] + other_meow
        else:
            meow_cats = sorted(meow_vals.unique())
        data['meow_zone'] = pd.Categorical(meow_vals, categories=meow_cats, ordered=True)
    outcome_series = data[outcome_col].astype('category')
    tip_order = ['no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    outcome_series = outcome_series.cat.reorder_categories(tip_order, ordered=True)
    outcome_codes = outcome_series.cat.codes.values
    n_categories = len(tip_order)
    reference_outcome = tip_order[0]
    dummies_list = []
    for col in predictor_cols:
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
        dummies_list.append(dummies)
    X = pd.concat(dummies_list, axis=1).astype(float)
    return X.values, outcome_codes, n_categories, reference_outcome, tip_order, X.columns.tolist()

def compute_multinomial_summaries(trace, predictor_names, outcome_categories, reference_outcome):
    summaries = {'effects': {}}
    beta_samples = trace.posterior['beta'].values  # (chain, draw, n_pred, n_cat-1)
    intercept_samples = trace.posterior['intercept'].values
    n_pred = beta_samples.shape[2]
    n_cat_minus_1 = beta_samples.shape[3]
    for i in range(n_pred):
        for j in range(n_cat_minus_1):
            outcome_level = outcome_categories[j+1]
            coef_name = f'{predictor_names[i]} vs {reference_outcome} for {outcome_level}'
            samples_beta = beta_samples[:, :, i, j].reshape(-1)
            samples_or = np.exp(samples_beta)
            mean_or = np.mean(samples_or)
            hdi_or = az.hdi(samples_or, hdi_prob=0.95)
            prob_dir = np.mean(samples_beta > 0)
            summaries['effects'][coef_name] = {
                'odds_ratio': {'mean': mean_or, 'hdi_lower': hdi_or[0], 'hdi_upper': hdi_or[1]},
                'bayesian_evidence': {'probability_direction': prob_dir}
            }
    return summaries

def print_multinomial_summary(result):
    if result is None:
        return
    print(f"\n---------- MULTINOMIAL FULL MODEL (outcome: tipping_point_type) ----------")
    summaries = result['summaries']
    ref_outcome = result['ref_outcome']
    if summaries is None or 'effects' not in summaries:
        print("No effect summaries available.")
        return
    for coef, vals in summaries['effects'].items():
        parts = coef.split(' for ')
        if len(parts) != 2:
            continue
        pred_part = parts[0]
        outcome_level = parts[1]
        pred_abbr = pred_part.split(' vs ')[0]
        pred_abbr = abbreviate_category_name(pred_abbr)
        outcome_abbr = abbreviate_category_name(outcome_level)
        or_mean = vals['odds_ratio']['mean']
        or_low = vals['odds_ratio']['hdi_lower']
        or_up = vals['odds_ratio']['hdi_upper']
        prob_dir = vals['bayesian_evidence']['probability_direction']
        print(f"{pred_abbr} -> {outcome_abbr} (vs {ref_outcome}): OR = {or_mean:.2g} [{or_low:.2g}, {or_up:.2g}], P(beta>0) = {prob_dir:.2g}")
    print("-----------------------------------------\n")

def run_multinomial_analysis(data, outcome_col, all_predictors):
    # subsets = []
    # for r in range(1, len(all_predictors)+1):
    #     for combo in combinations(all_predictors, r):
    #         subsets.append(list(combo))
    # results = []
    # for preds in subsets:
    #     X, y, n_cat, ref_outcome, outcome_cats, pred_names = prepare_multinomial_data(data, outcome_col, preds)
    #     trace, _ = fit_multinomial_model(X, y, n_cat, pred_names, model_name='_'.join(preds))
    #     waic_obj = az.waic(trace)
    #     loo_obj = az.loo(trace)
    #     waic_val = -2 * waic_obj.elpd_waic
    #     waic_se = 2 * waic_obj.se
    #     loo_val = -2 * loo_obj.elpd_loo
    #     loo_se = 2 * loo_obj.se
    #     summaries = compute_multinomial_summaries(trace, pred_names, outcome_cats, ref_outcome)
    #     results.append({
    #         'vars': preds,
    #         'waic': waic_val,
    #         'waic_se': waic_se,
    #         'loo': loo_val,
    #         'loo_se': loo_se,
    #         'trace': trace,
    #         'summaries': summaries,
    #         'outcome_cats': outcome_cats,
    #         'ref_outcome': ref_outcome
    #     })
    # results_sorted = sorted(results, key=lambda x: x['waic'])
    # print("\n=== MULTINOMIAL MODELS RANKED BY WAIC ===")
    # for rank, res in enumerate(results_sorted, 1):
    #     vars_str = ' + '.join(res['vars'])
    #     print(f"{rank}. {vars_str}: WAIC = {format_num(res['waic'])} (se={format_num(res['waic_se'])}), LOO = {format_num(res['loo'])} (se={format_num(res['loo_se'])})")
    # df_compare = pd.DataFrame([{
    #     'vars': '_'.join(r['vars']),
    #     'WAIC': r['waic'],
    #     'WAIC_se': r['waic_se'],
    #     'LOO': r['loo'],
    #     'LOO_se': r['loo_se']
    # } for r in results_sorted])
    # df_compare.to_csv('csv/multinomial_model_comparison.csv', index=False, float_format='%.2g')
    # return results_sorted # the commented snippet did the comparison among the sub-models
    preds = all_predictors
    X, y, n_cat, ref_outcome, outcome_cats, pred_names = prepare_multinomial_data(data, outcome_col, preds)
    trace, _ = fit_multinomial_model(X, y, n_cat, pred_names, model_name='_'.join(preds))
    summaries = compute_multinomial_summaries(trace, pred_names, outcome_cats, ref_outcome)
    result = {
        'vars': preds,
        'trace': trace,
        'summaries': summaries,
        'outcome_cats': outcome_cats,
        'ref_outcome': ref_outcome
    }
    print_multinomial_summary(result)
    return [result]

def plot_multinomial_odds_ratios(results, outcome_name):
    set_plot_style()
    best = results[0]
    summaries = best['summaries']
    if not summaries['effects']:
        return
    groups = {'taxonomic_group': [], 'biome': [], 'koppen_zone': [], 'meow_zone': []}
    for coef_key, vals in summaries['effects'].items():
        parts = coef_key.split(' for ')
        if len(parts) != 2:
            continue
        pred_part = parts[0]
        out_part = parts[1]
        if pred_part.startswith('taxonomic_group_'):
            group = 'taxonomic_group'
            pred_name = pred_part.replace('taxonomic_group_', '').split(' vs ')[0]
        elif pred_part.startswith('biome_'):
            group = 'biome'
            pred_name = pred_part.replace('biome_', '').split(' vs ')[0]
        elif pred_part.startswith('koppen_zone_'):
            group = 'koppen_zone'
            pred_name = pred_part.replace('koppen_zone_', '').split(' vs ')[0]
        elif pred_part.startswith('meow_zone_'):
            group = 'meow_zone'
            pred_name = pred_part.replace('meow_zone_', '').split(' vs ')[0]
        else:
            continue
        out_abbr = abbreviate_category_name(out_part)
        pred_abbr = abbreviate_category_name(pred_name)
        coef_label = f'{pred_abbr} -> {out_abbr}'
        or_mean = vals['odds_ratio']['mean']
        or_lower = vals['odds_ratio']['hdi_lower']
        or_upper = vals['odds_ratio']['hdi_upper']
        groups[group].append((coef_label, or_mean, or_lower, or_upper))
    for group, items in groups.items():
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: x[0])
        coef_names = [item[0] for item in items_sorted]
        effect_data = [(item[1], item[2], item[3]) for item in items_sorted]
        y_pos = np.arange(len(coef_names)-1, -1, -1)
        fig, ax = plt.subplots(1, 1, figsize=(2, 0.4*len(coef_names)+2))
        for i, (mean, low, high) in enumerate(effect_data):
            ax.plot(mean, y_pos[i], 'o', color='#1f77b4', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
            ax.hlines(y_pos[i], low, high, color='black', linewidth=1.5)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_names)
        ax.set_xlabel('Odds Ratio')
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.5, len(coef_names)-0.5)
        plt.tight_layout()
        plt.savefig(f'img/multinomial_odds_ratios_{outcome_name}_{group}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

def fit_multinomial_model(X, y, n_categories, predictor_names, model_name='multinomial'):
    n_obs, n_pred = X.shape
    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0, sigma=5, shape=n_categories-1)
        beta = pm.Normal('beta', mu=0, sigma=5, shape=(n_pred, n_categories-1))
        logits_nonref = pm.math.dot(X, beta) + intercept
        zeros = pytensor.tensor.zeros((n_obs, 1))
        logits_full = pytensor.tensor.concatenate([zeros, logits_nonref], axis=1)
        y_obs = pm.Categorical('y_obs', logit_p=logits_full, observed=y)
        trace = pm.sample(draws=1500, tune=1500, cores=4, target_accept=0.95, return_inferencedata=True, random_seed=seed)
        pm.compute_log_likelihood(trace, model=model)
    check_convergence(trace, model_name)
    return trace, predictor_names

def plot_multinomial_posterior(trace, predictor_names, outcome_categories, reference_outcome):
    set_plot_style()
    beta_samples = trace.posterior['beta'].values
    n_pred = beta_samples.shape[2]
    n_cat_minus_1 = beta_samples.shape[3]
    n_plots = n_pred * n_cat_minus_1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    plot_idx = 0
    for i in range(n_pred):
        for j in range(n_cat_minus_1):
            if plot_idx >= len(axes): break
            ax = axes[plot_idx]
            samples = beta_samples[:, :, i, j].reshape(-1)
            ax.hist(samples, bins=50, alpha=0.5, density=True, edgecolor='black')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel('Log-Odds')
            ax.set_ylabel('Density')
            plot_idx += 1
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig('img/multinomial_posterior.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def print_reference_categories(data):
    tip_cats = ['no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    tip_ref = tip_cats[0]
    tax_cats = data['taxonomic_group'].dropna().unique()
    if 'Other' in tax_cats:
        tax_ref = 'Other'
    else:
        tax_ref = sorted(tax_cats)[0]
    biome_vals = data['biome'].astype(str).replace('Mixed/Generalist Habitats', 'Unknown/Unclassified')
    biome_cats = sorted(biome_vals.unique())
    if 'Unknown/Unclassified' in biome_cats:
        biome_ref = 'Unknown/Unclassified'
    else:
        biome_ref = biome_cats[0]
    koppen_vals = data['koppen_zone'].apply(convert_koppen_code)
    koppen_cats = sorted(koppen_vals.dropna().unique())
    koppen_ref = koppen_cats[0] if koppen_cats else 'Af'
    meow_cats = sorted(data['meow_zone'].dropna().unique())
    meow_ref = meow_cats[0] if meow_cats else 'Arctic'
    print(f"Reference (dropped) categories:\n  - tipping point type: {tip_ref}\n  - taxonomic group: {tax_ref}\n  - biome: {biome_ref}\n  - koppen zone: {koppen_ref}\n  - meow zone: {meow_ref}")

def print_contingency_tables(data):
    other_vars = ['taxonomic_group', 'biome', 'koppen_zone', 'meow_zone']
    tip_order = ['no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    for var in other_vars:
        if var not in data.columns:
            continue
        cross = pd.crosstab(data[var], data['tipping_point_type'], margins=False)
        cross = cross[tip_order]
        print(f"\nContingency table: {var} vs. tipping point type")
        print(cross.to_string())
        print("-" * 80)

def run_analysis(probability_threshold):
    bif_df = load_and_preprocess_bifurcation(threshold=probability_threshold)
    iucn_df = load_and_preprocess_iucn()
    biome_df = load_and_preprocess_biome()
    koppen_df = load_and_preprocess_koppen()
    meow_df = load_and_preprocess_meow()
    print("\n----------MULTICATEGORY MULTINOMIAL LOGISTIC REGRESSION----------\n")
    multi_data = prepare_multicategory_data(bif_df, biome_df, koppen_df, meow_df)
    if multi_data.empty:
        print("Not enough data for multicategory analysis.")
    else:
        print_reference_categories(multi_data)
        print_contingency_tables(multi_data)
        outcome_col = 'tipping_point_type'
        all_predictors = ['taxonomic_group', 'biome', 'koppen_zone', 'meow_zone']
        results_sorted = run_multinomial_analysis(multi_data, outcome_col, all_predictors)
        if results_sorted:
            full_result = None
            for res in results_sorted:
                if set(res['vars']) == set(all_predictors):
                    full_result = res
                    break
            if full_result is not None:
                plot_multinomial_odds_ratios([full_result], 'bifurcation_type')
                plot_multinomial_posterior(full_result['trace'], full_result['vars'], full_result['outcome_cats'], full_result['ref_outcome'])
    threatened_data = merge_datasets(bif_df, iucn_df)
    observed_proportions_plot(iucn_df, 'threatened', probability_threshold)
    combined_data = threatened_data.copy()
    if not koppen_df.empty:
        combined_data = pd.merge(combined_data, koppen_df, on='species_short', how='left')
    if not meow_df.empty:
        combined_data = pd.merge(combined_data, meow_df, on='species_short', how='left')
    bayesian_results = []
    data_bin, feat_bin = prepare_binary_data(combined_data, biome_df, 'is_threatened')
    print("\n----------LOGIT----------\n")
    res_fixed = run_binary_model_fixed(data_bin, feat_bin, 'is_threatened', model_name='fixed_logistic')
    bayesian_results.append(res_fixed)
    print_binary_summary(res_fixed)
    if 'taxonomic_group' in data_bin.columns:
        res_hier_tax = run_binary_model_hierarchical(data_bin, feat_bin, 'is_threatened', group_col='taxonomic_group', model_name='hierarchical_taxonomic_logistic')
        bayesian_results.append(res_hier_tax)
        print_binary_summary(res_hier_tax)
    if 'biome' in data_bin.columns:
        res_hier_bio = run_binary_model_hierarchical(data_bin, feat_bin, 'is_threatened', group_col='biome', model_name='hierarchical_biome_logistic')
        bayesian_results.append(res_hier_bio)
        print_binary_summary(res_hier_bio)
    if 'koppen_zone' in data_bin.columns and data_bin['koppen_zone'].notna().any():
        res_hier_kop = run_binary_model_hierarchical(data_bin, feat_bin, 'is_threatened', group_col='koppen_zone', model_name='hierarchical_koppen_logistic')
        bayesian_results.append(res_hier_kop)
        print_binary_summary(res_hier_kop)
    if 'meow_zone' in data_bin.columns and data_bin['meow_zone'].notna().any():
        res_hier_meow = run_binary_model_hierarchical(data_bin, feat_bin, 'is_threatened', group_col='meow_zone', model_name='hierarchical_meow_logistic')
        bayesian_results.append(res_hier_meow)
        print_binary_summary(res_hier_meow)
    bayesian_df = pd.DataFrame(bayesian_results)
    if bayesian_df.empty:
        print("No Bayesian results to plot.")
    for idx, row in bayesian_df.iterrows():
        create_binary_inference_figure(bayesian_df, row['outcome'], row['model'])
        create_binary_posterior_plot(bayesian_df, row['outcome'], row['model'])

def main():
    probability_threshold = [0.5]
    create_directories()
    for thresh in probability_threshold:
        os.makedirs(f'img', exist_ok=True)
        run_analysis(thresh)

if __name__ == "__main__":
    main()
