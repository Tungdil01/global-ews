#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil
import numpy as np
import pandas as pd
from main_dl import load_combined_data
from main_orchestrator import check_annual_series, DL_FILE, FAIL_FILE, min_records

def load_full_data(min_records):
    print("Loading combined data (raw, without prefiltering)...")
    datasets_dict, species_list_raw, data_taxonomy, genus_taxonomy = load_combined_data(min_records=min_records, prefilter=False, min_time=1900, whitelist={})
    return datasets_dict, species_list_raw

def _annualize(grp):
    annual = grp.groupby(np.floor(grp['TIME']).astype(int))['ABUNDANCE'].mean().reset_index()
    annual.columns = ['YEAR', 'ABUNDANCE']
    annual['ABUNDANCE'] = annual['ABUNDANCE'].astype(float)
    return annual.dropna(subset=['ABUNDANCE'])

def _bounds(grp):
    return int(np.floor(grp['TIME'].min())), int(np.floor(grp['TIME'].max()))

def filter_species_by_time(datasets_dict):
    species_info = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df = df.copy()
        if 'dataset' not in df.columns:
            df['dataset'] = ds_name
        for (sp, ds), ds_grp in df.groupby(['SPECIES_ID', 'dataset']):
            pooled_annual = _annualize(ds_grp)
            passed_pooled, reason_pooled = check_annual_series(pooled_annual)
            if passed_pooled:
                for ts, ts_grp in ds_grp.groupby('TIMESERIES_ID'):
                    first_year, last_year = _bounds(ts_grp)
                    species_info.append({'species': sp, 'dataset': ds, 'timeseries_id': ts, 'first_year': first_year, 'last_year': last_year, 'n_records': len(ts_grp), 'keep': True, 'reason': 'pooled_pass'})
            else:
                for ts, ts_grp in ds_grp.groupby('TIMESERIES_ID'):
                    first_year, last_year = _bounds(ts_grp)
                    annual = _annualize(ts_grp)
                    passed, reason = check_annual_series(annual)
                    species_info.append({'species': sp, 'dataset': ds, 'timeseries_id': ts, 'first_year': first_year, 'last_year': last_year, 'n_records': len(ts_grp), 'keep': passed, 'reason': 'segment_pass' if passed else reason})
    df_info = pd.DataFrame(species_info)
    kept = df_info[df_info['keep'] == True]
    removed = df_info[df_info['keep'] == False]
    print(f"Total (species, dataset, timeseries) triplets: {len(df_info)}")
    print(f"Kept: {len(kept)}")
    print(f"Removed: {len(removed)}")
    return kept[['species', 'dataset', 'timeseries_id', 'first_year', 'last_year']].copy(), df_info

def backup_file(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path}")
        return True
    return False

def clean_result_files(keep_pairs):
    keep_pairs = keep_pairs.copy()
    keep_pairs['species'] = keep_pairs['species'].astype(str).str.strip()
    keep_pairs['dataset'] = keep_pairs['dataset'].astype(str).str.strip()
    keep_set = keep_pairs[['species', 'dataset']].drop_duplicates()
    for f in [DL_FILE, FAIL_FILE]:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        backup_file(f)
        df = pd.read_csv(f)
        if 'species' not in df.columns or 'dataset' not in df.columns:
            continue
        before = len(df)
        df['species'] = df['species'].astype(str).str.strip()
        df['dataset'] = df['dataset'].astype(str).str.strip()
        df = df.merge(keep_set, on=['species', 'dataset'], how='inner')
        removed = before - len(df)
        if removed > 0:
            df.to_csv(f, index=False)
            print(f"Removed {removed} rows from {f} (species+dataset not in whitelist).")
        else:
            print(f"No rows removed from {f}.")

def run_pipeline(min_records):
    datasets_dict, species_list_raw = load_full_data(min_records)
    total = sum(len(df) for df in datasets_dict.values())
    print(f"Raw data loaded: {total} records, {len(species_list_raw)} species across {len(datasets_dict)} datasets.")
    kept_pairs, info_df = filter_species_by_time(datasets_dict)
    os.makedirs('csv', exist_ok=True)
    if not os.path.exists('csv/species_time_info.csv'):
        info_df.to_csv('csv/species_time_info.csv', index=False)
        print("Time info saved to csv/species_time_info.csv.")
    else:
        print("csv/species_time_info.csv already exists, skipping write.")
    if not os.path.exists('csv/species_post1900.csv'):
        kept_pairs.to_csv('csv/species_post1900.csv', index=False)
        print(f"Kept (species, dataset) pairs saved to csv/species_post1900.csv ({len(kept_pairs)} pairs).")
    else:
        print("csv/species_post1900.csv already exists, skipping write.")
    clean_result_files(kept_pairs)
    print("Result files cleaned (species+dataset failing orchestrator filter removed).")
    return kept_pairs

def main():
    run_pipeline(min_records)

if __name__ == "__main__":
    main()
