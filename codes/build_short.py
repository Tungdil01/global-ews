import os
import shutil
import numpy as np
import pandas as pd
from main_orchestrator import setup_data, DATASET_PRIORITY, probability_threshold, late_transition_window, create_summary_figure, BANDING_CODE_TO_NAME, EXCLUDED_GROUPS
NAME_TO_BANDING_CODE = {v: k for k, v in BANDING_CODE_TO_NAME.items()}

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    parts = str(species_name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return species_name

def copy_file(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)
    print(f"File copied: {src_path} -> {dst_path}")

def read_data(dl_path, dev_path):
    if not os.path.exists(dl_path):
        print(f"Error: {dl_path} missing.")
        return None, None
    df_dl = pd.read_csv(dl_path)
    if os.path.exists(dev_path):
        df_dev = pd.read_csv(dev_path)
    else:
        print(f"Note: {dev_path} missing; proceeding with DL only.")
        df_dev = pd.DataFrame(columns=['species', 'dataset'])
    return df_dl, df_dev

def filter_gbif(df_dl, df_dev, filter_type):
    if filter_type == 'non_gbif':
        df_dl = df_dl[df_dl['dataset'] != 'GBIF']
        if not df_dev.empty:
            df_dev = df_dev[df_dev['dataset'] != 'GBIF']
    elif filter_type == 'gbif':
        df_dl = df_dl[df_dl['dataset'] == 'GBIF']
        if not df_dev.empty:
            df_dev = df_dev[df_dev['dataset'] == 'GBIF']
    elif filter_type == 'all':
        pass
    else:
        raise ValueError("filter_type must be 'non_gbif', 'gbif', or 'all'")
    return df_dl, df_dev

def merge_dfs(df_dl, df_dev):
    if df_dev.empty:
        return df_dl.copy()
    dl_has_ts = 'timeseries_id' in df_dl.columns
    dev_has_ts = 'timeseries_id' in df_dev.columns
    if dl_has_ts and dev_has_ts:
        return pd.merge(df_dl, df_dev, on=['species', 'dataset', 'timeseries_id'], how='outer', suffixes=('_dl', '_dev'))
    return pd.merge(df_dl, df_dev, on=['species', 'dataset'], how='outer', suffixes=('_dl', '_dev'))

def combine_taxonomic(merged):
    if 'taxonomic_group_dl' in merged.columns and 'taxonomic_group_dev' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dl'].fillna(merged['taxonomic_group_dev'])
    elif 'taxonomic_group_dl' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dl']
    elif 'taxonomic_group_dev' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dev']
    return merged

def convert_insecta_to_hexapoda(merged):
    if 'taxonomic_group' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group'].replace('Insecta', 'Hexapoda')
    return merged

def remove_empty_transition_year(merged):
    if 'transition_year' in merged.columns:
        merged = merged.dropna(subset=['transition_year'])
    return merged

def remove_other_taxonomic(merged):
    if 'taxonomic_group' in merged.columns:
        merged = merged[merged['taxonomic_group'] != 'Other']
        merged = merged[~merged['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    return merged

def print_time_series_stats(merged, short_file):
    if 'timeseries_id' not in merged.columns:
        return
    short_df = pd.read_csv(short_file)
    short_species = set(short_df['species'].apply(preprocess_species_name).dropna())
    df = merged.copy()
    df['_species_short'] = df['species'].apply(preprocess_species_name)
    df = df[df['_species_short'].isin(short_species)]
    counts = df.groupby('_species_short')['timeseries_id'].nunique()
    multi = counts[counts > 1]
    print("\n=== TIME SERIES STATISTICS PER SPECIES ===")
    print(f"Total species: {len(counts)}")
    print(f"Species with >1 time series: {len(multi)}")
    if len(multi) > 0:
        print(f"Max time series for a species: {int(multi.max())}")
        print(f"Mean time series for species with >1: {multi.mean():.2g}")
        print(f"SD time series for species with >1: {multi.std():.2g}")
        print(f"Median time series for species with >1: {multi.median():.2g}")
    print("Selection rule: 1) Max unique years, 2) Max total data points, 3) Dataset priority.")
    print("==========================================\n")

def print_longest_time_series(datasets_dict):
    frames = []
    for ds_name, df in datasets_dict.items():
        if str(ds_name).lower() == 'gbif' or len(df) == 0:
            continue
        df_local = df.copy()
        df_local['TIME'] = pd.to_numeric(df_local['TIME'], errors='coerce')
        df_local = df_local.dropna(subset=['TIME'])
        df_local['TIMESERIES_ID'] = df_local['TIMESERIES_ID'].astype(str).str.strip()
        df_local['_year'] = np.floor(df_local['TIME']).astype(int)
        quality = df_local.groupby(['SPECIES_ID', 'TIMESERIES_ID']).agg(min_year=('_year', 'min'), max_year=('_year', 'max'), n_points=('ABUNDANCE', 'size')).reset_index()
        quality['duration'] = quality['max_year'] - quality['min_year'] + 1
        quality['dataset'] = ds_name
        frames.append(quality)
    if not frames:
        return
    quality_df = pd.concat(frames, ignore_index=True)
    longest_years = quality_df.sort_values('duration', ascending=False).iloc[0]
    longest_points = quality_df.sort_values('n_points', ascending=False).iloc[0]
    print("\n=== LONGEST TIME SERIES STATISTICS ===")
    print(f"Longest by years: {longest_years['SPECIES_ID']} (Dataset: {longest_years['dataset']}) with {int(longest_years['duration'])} years and {int(longest_years['n_points'])} points.")
    print(f"Longest by points: {longest_points['SPECIES_ID']} (Dataset: {longest_points['dataset']}) with {int(longest_points['duration'])} years and {int(longest_points['n_points'])} points.")
    print("=====================================\n")

def _canonical_species(s):
    s = str(s).strip()
    return NAME_TO_BANDING_CODE.get(s, s)

def compute_data_quality_scores(datasets_dict):
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df_local = df.copy()
        df_local['TIME'] = pd.to_numeric(df_local['TIME'], errors='coerce')
        df_local = df_local.dropna(subset=['TIME'])
        df_local['TIMESERIES_ID'] = df_local['TIMESERIES_ID'].astype(str).str.strip()
        df_local['_year'] = np.floor(df_local['TIME']).astype(int)
        df_local['_canonical'] = df_local['SPECIES_ID'].astype(str).str.strip().map(_canonical_species)
        quality = df_local.groupby(['_canonical', 'TIMESERIES_ID']).agg(n_unique_years=('_year', 'nunique'), n_points=('ABUNDANCE', 'size')).reset_index()
        quality = quality.rename(columns={'TIMESERIES_ID': 'timeseries_id'})
        quality['timeseries_id'] = quality['timeseries_id'].astype(str).str.strip()
        quality['dataset'] = ds_name
        frames.append(quality)
    if not frames:
        return pd.DataFrame(columns=['_canonical', 'dataset', 'timeseries_id', 'n_unique_years', 'n_points'])
    return pd.concat(frames, ignore_index=True)

def select_best_dataset_per_species(merged, datasets_dict, mode, threshold=probability_threshold):
    if merged.empty:
        return merged
    merged = merged.copy()
    merged['species'] = merged['species'].astype(str).str.strip()
    merged['_canonical'] = merged['species'].map(_canonical_species)
    merged['dataset'] = merged['dataset'].astype(str).str.strip()
    if 'timeseries_id' in merged.columns:
        merged['timeseries_id'] = merged['timeseries_id'].astype(str).str.strip()
    merged['_priority'] = merged['dataset'].map(DATASET_PRIORITY).fillna(999)
    quality = compute_data_quality_scores(datasets_dict)
    if 'timeseries_id' in merged.columns:
        merged = merged.merge(quality, on=['_canonical', 'dataset', 'timeseries_id'], how='left')
    else:
        quality_dataset = quality.groupby(['_canonical', 'dataset']).agg(n_unique_years=('n_unique_years', 'sum'), n_points=('n_points', 'sum')).reset_index()
        merged = merged.merge(quality_dataset, on=['_canonical', 'dataset'], how='left')
    merged['n_unique_years'] = merged['n_unique_years'].fillna(0)
    merged['n_points'] = merged['n_points'].fillna(0)
    merged['best_probability'] = pd.to_numeric(merged['best_probability'], errors='coerce').fillna(0.0)
    merged = merged.sort_values(['_canonical', 'n_unique_years', 'n_points', '_priority'], ascending=[True, False, False, True])
    if mode == 'best':
        merged['_meets_threshold'] = merged['best_probability'] >= threshold
        merged['_has_any_meets'] = merged.groupby('_canonical')['_meets_threshold'].transform('any')
        merged = merged[merged['_meets_threshold'] | ~merged['_has_any_meets']]
        merged = merged.drop_duplicates(subset=['_canonical'], keep='first')
        merged = merged.drop(columns=['_meets_threshold', '_has_any_meets'])
    elif mode == 'threshold':
        merged = merged[merged['best_probability'] >= threshold].drop_duplicates(subset=['_canonical'], keep='first')
    else:
        raise ValueError(f"Unknown mode: {mode}")
    merged = merged.drop(columns=['_priority', 'n_unique_years', 'n_points', '_canonical'])
    return merged

def remove_duplicates(merged):
    return merged.drop_duplicates(subset=['species'], keep='first')

def remove_low_best_probability(merged, min_prob=probability_threshold):
    return merged[merged['best_probability'] >= min_prob]

def remove_late_transitions(merged, datasets_dict, window_years=1):
    if 'transition_year' not in merged.columns:
        return merged
    merged = merged.copy()
    merged['_canonical'] = merged['species'].astype(str).str.strip().map(_canonical_species)
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df_local = df.copy()
        df_local['TIME'] = pd.to_numeric(df_local['TIME'], errors='coerce')
        df_local['TIMESERIES_ID'] = df_local['TIMESERIES_ID'].astype(str).str.strip()
        df_local['_canonical'] = df_local['SPECIES_ID'].astype(str).str.strip().map(_canonical_species)
        g = df_local.groupby(['_canonical', 'TIMESERIES_ID'])['TIME'].max().reset_index()
        g.columns = ['_canonical', 'timeseries_id', 'max_year']
        g['dataset'] = ds_name
        frames.append(g)
    if not frames:
        return merged.drop(columns=['_canonical'])
    max_year_df = pd.concat(frames, ignore_index=True)
    if 'timeseries_id' in merged.columns:
        merged = merged.merge(max_year_df, on=['_canonical', 'dataset', 'timeseries_id'], how='left')
    else:
        max_year_agg = max_year_df.groupby(['_canonical', 'dataset'])['max_year'].max().reset_index()
        merged = merged.merge(max_year_agg, on=['_canonical', 'dataset'], how='left')
    mask = (merged['transition_year'].notna()) & (merged['transition_year'] >= merged['max_year'] - window_years)
    merged = merged[~mask].copy()
    merged = merged.drop(columns=['max_year', '_canonical'])
    return merged

def select_columns(merged, include_dev=False, column_order=None):
    shared_cols = ['species', 'dataset', 'timeseries_id', 'taxonomic_group']
    dl_cols = ['tipping_point_type', 'raw_tipping_point_type', 'ews_metrics', 'transition_year', 'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
    dev_cols = ['early_warning', 'final_re', 'final_im', 'final_abs', 'bifurcation_type', 'time_to_transition']
    if column_order is not None:
        existing_cols = [col for col in column_order if col in merged.columns]
        return merged[existing_cols]
    all_cols = shared_cols + dl_cols + (dev_cols if include_dev else [])
    existing_cols = [col for col in all_cols if col in merged.columns]
    return merged[existing_cols]

def write_output(merged, out_path, filter_type):
    merged.to_csv(out_path, index=False)
    label = 'non-GBIF' if filter_type == 'non_gbif' else 'GBIF' if filter_type == 'gbif' else 'all (GBIF + non-GBIF)'
    print(f"File created: {out_path} with {len(merged)} rows ({label}).")
    print(f"Number of species: {merged['species'].nunique()}")

def build_output(merged, datasets_dict, out_file, filter_type, include_dev=False, remove_late=False, window_years=1, min_best_probability=None, mode=None, column_order=None):
    merged = remove_empty_transition_year(merged)
    merged = remove_other_taxonomic(merged)
    if mode is None:
        mode = 'threshold' if min_best_probability is not None else 'best'
    threshold = min_best_probability if min_best_probability is not None else probability_threshold
    merged = select_best_dataset_per_species(merged, datasets_dict, mode, threshold)
    if remove_late:
        merged = remove_late_transitions(merged, datasets_dict, window_years)
    merged = merged.copy()
    merged['_species_short'] = merged['species'].apply(preprocess_species_name)
    merged = merged.drop_duplicates(subset=['_species_short'], keep='first').drop(columns=['_species_short'])
    merged = select_columns(merged, include_dev=include_dev, column_order=column_order)
    write_output(merged, out_file, filter_type)

def plot_large_summary(large_file):
    df = pd.read_csv(large_file)
    create_summary_figure(df, probability_threshold)

def print_short_tipping_type_counts(short_file):
    df = pd.read_csv(short_file)
    if 'tipping_point_type' not in df.columns:
        print(f"{short_file} has no 'tipping_point_type' column.")
        return
    counts = df['tipping_point_type'].value_counts()
    print(f"\n=== SHORT FILE TIPPING TYPE COUNTS ({len(df)} rows) ===")
    print(f"fold:         {int(counts.get('fold bifurcation', 0))}")
    print(f"transc:       {int(counts.get('transcritical bifurcation', 0))}")
    print(f"Hopf:         {int(counts.get('Hopf bifurcation', 0))}")
    print(f"no-bif:       {int(counts.get('no-bifurcation', 0))}")
    print("==========================================\n")

def prepare_merged_data(DL_FILE, DEV_FILE, COPY_FILE, DATASET_FILTER):
    df_dl, df_dev = read_data(DL_FILE, DEV_FILE)
    if df_dl is None:
        return None, None
    copy_file(DL_FILE, COPY_FILE)
    column_order = df_dl.columns.tolist()
    df_dl, df_dev = filter_gbif(df_dl, df_dev, DATASET_FILTER)
    merged = merge_dfs(df_dl, df_dev)
    merged = combine_taxonomic(merged)
    merged = convert_insecta_to_hexapoda(merged)
    return merged, column_order

def run_pipeline(DL_FILE, DEV_FILE, COPY_FILE, LARGE_FILE, SHORT_FILE, DATASET_FILTER, MIN_RECORDS):
    merged, column_order = prepare_merged_data(DL_FILE, DEV_FILE, COPY_FILE, DATASET_FILTER)
    if merged is None:
        return
    datasets_dict, _, _, _ = setup_data(min_records=MIN_RECORDS)
    print_longest_time_series(datasets_dict)
    build_output(merged, datasets_dict, LARGE_FILE, DATASET_FILTER, include_dev=False, remove_late=False, mode='best', column_order=column_order)
    build_output(merged, datasets_dict, SHORT_FILE, DATASET_FILTER, include_dev=True, remove_late=True, window_years=late_transition_window, min_best_probability=probability_threshold, mode='threshold', column_order=column_order)
    print_time_series_stats(merged, SHORT_FILE)
    print_short_tipping_type_counts(SHORT_FILE)
    plot_large_summary(LARGE_FILE)

def main():
    DL_FILE = 'csv/bifurcation_results_dl_ews.csv'
    DEV_FILE = 'csv/bifurcation_results_dev.csv'
    COPY_FILE = 'csv/bifurcation_results.csv'
    LARGE_FILE = 'csv/bifurcation_results_large.csv'
    SHORT_FILE = 'csv/bifurcation_results_short.csv'
    DATASET_FILTER = 'non_gbif'
    MIN_RECORDS = 20
    run_pipeline(DL_FILE, DEV_FILE, COPY_FILE, LARGE_FILE, SHORT_FILE, DATASET_FILTER, MIN_RECORDS)

if __name__ == '__main__':
    main()
