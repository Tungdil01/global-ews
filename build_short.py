#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
from main_orchestrator import setup_data

def read_data(dl_path, dev_path):
    if not os.path.exists(dl_path) or not os.path.exists(dev_path):
        print("Error: One or both input files are missing.")
        return None, None
    return pd.read_csv(dl_path), pd.read_csv(dev_path)

def filter_gbif(df_dl, df_dev, filter_type):
    if filter_type == 'non_gbif':
        df_dl = df_dl[df_dl['dataset'] != 'GBIF']
        df_dev = df_dev[df_dev['dataset'] != 'GBIF']
    elif filter_type == 'gbif':
        df_dl = df_dl[df_dl['dataset'] == 'GBIF']
        df_dev = df_dev[df_dev['dataset'] == 'GBIF']
    elif filter_type == 'all':
        pass
    else:
        raise ValueError("filter_type must be 'non_gbif', 'gbif', or 'all'")
    return df_dl, df_dev

def merge_dfs(df_dl, df_dev):
    return pd.merge(df_dl, df_dev, on=['species', 'dataset'], how='outer', suffixes=('_dl', '_dev'))

def combine_taxonomic(merged):
    if 'taxonomic_group_dl' in merged.columns and 'taxonomic_group_dev' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dl'].fillna(merged['taxonomic_group_dev'])
    elif 'taxonomic_group_dl' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dl']
    elif 'taxonomic_group_dev' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group_dev']
    else:
        merged['taxonomic_group'] = None
    return merged

def convert_insecta_to_hexapoda(merged):
    if 'taxonomic_group' in merged.columns:
        merged['taxonomic_group'] = merged['taxonomic_group'].replace('Insecta', 'Hexapoda')
    return merged

def remove_late_transitions(merged, full_data, window_years=1):
    if 'transition_year' not in merged.columns:
        return merged
    full_data['TIME'] = pd.to_numeric(full_data['TIME'], errors='coerce')
    max_year_df = full_data.groupby(['SPECIES_ID', 'dataset'])['TIME'].max().reset_index()
    max_year_df.columns = ['species', 'dataset', 'max_year']
    merged = merged.merge(max_year_df, on=['species', 'dataset'], how='left')
    mask = (merged['transition_year'].notna()) & (merged['transition_year'] >= merged['max_year'] - window_years)
    merged = merged[~mask].copy()
    merged = merged.drop(columns=['max_year'])
    return merged

def select_columns(merged):
    shared_cols = ['species', 'dataset', 'taxonomic_group']
    dl_cols = ['tipping_point_type', 'raw_tipping_point_type', 'ews_metrics', 'transition_year', 'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
    dev_cols = ['early_warning', 'final_re', 'final_im', 'final_abs', 'bifurcation_type', 'time_to_transition']
    all_cols = shared_cols + dl_cols + dev_cols
    existing_cols = [col for col in all_cols if col in merged.columns]
    return merged[existing_cols]

def write_output(merged, out_path, filter_type):
    merged.to_csv(out_path, index=False)
    label = 'non-GBIF' if filter_type == 'non_gbif' else 'GBIF' if filter_type == 'gbif' else 'all (GBIF + non-GBIF)'
    print(f"Short file created: {out_path} with {len(merged)} rows ({label}).")

def main():
    DL_FILE = 'csv/bifurcation_results_dl_ews.csv'
    DEV_FILE = 'csv/bifurcation_results_dev.csv'
    OUT_FILE = 'csv/bifurcation_results_short.csv'
    DATASET_FILTER = 'non_gbif'
    transition_year = 3
    df_dl, df_dev = read_data(DL_FILE, DEV_FILE)
    if df_dl is None or df_dev is None:
        return
    df_dl, df_dev = filter_gbif(df_dl, df_dev, DATASET_FILTER)
    merged = merge_dfs(df_dl, df_dev)
    merged = combine_taxonomic(merged)
    merged = convert_insecta_to_hexapoda(merged)
    full_data, _, _, _, _ = setup_data(min_records=20)
    merged = remove_late_transitions(merged, full_data, transition_year)
    merged = select_columns(merged)
    write_output(merged, OUT_FILE, DATASET_FILTER)

if __name__ == '__main__':
    main()
