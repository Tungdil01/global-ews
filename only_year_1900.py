#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import numpy as np
import pandas as pd
from main_dl import load_combined_data
from main_orchestrator import check_annual_series, DL_FILE, DEV_FILE, FAIL_FILE

def filter_species_by_time(full_data, min_year=1900):
    species_info = []
    for sp, grp in full_data.groupby('SPECIES_ID'):
        min_time = grp['TIME'].min()
        max_time = grp['TIME'].max()
        n_records = len(grp)
        keep = min_time >= min_year
        species_info.append({'species': sp, 'min_time': min_time, 'max_time': max_time, 'n_records': n_records, 'keep': keep})
    df_info = pd.DataFrame(species_info)
    kept = df_info[df_info['keep'] == True]
    removed = df_info[df_info['keep'] == False]
    print(f"Total species: {len(df_info)}")
    print(f"Kept (all records >= {min_year}): {len(kept)}")
    print(f"Removed (any record < {min_year}): {len(removed)}")
    return kept['species'].tolist(), df_info

def backup_file(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path}")
        return True
    return False

def clean_result_files(keep_species):
    keep_set = set(keep_species)
    for f in [DL_FILE, DEV_FILE, FAIL_FILE]:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        backup_file(f)
        df = pd.read_csv(f)
        if 'species' not in df.columns:
            continue
        before = len(df)
        df = df[df['species'].isin(keep_set)]
        removed = before - len(df)
        if removed > 0:
            df.to_csv(f, index=False)
            print(f"Removed {removed} rows from {f} (species with data before 1900).")
        else:
            print(f"No rows removed from {f}.")

def main():
    min_year = 1900
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    print("Loading combined data (raw, without prefiltering)...")
    full_data, species_list_raw, data_taxonomy, genus_taxonomy = load_combined_data(
        biotime_path=os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv'),
        rivfishtime_path=os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv'),
        cedar_creek_paths={
            'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
            'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
            'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')
        },
        gbif_path=os.path.join(data_dir, 'gbif', 'output.csv'),
        gpdd_path=os.path.join(data_dir, 'gpdd', 'gpdd_final.csv'),
        min_records=20,
        use_existing_biotime=True,
        prefilter=False
    )
    print(f"Raw data loaded: {len(full_data)} records, {len(species_list_raw)} species.")
    kept_species, info_df = filter_species_by_time(full_data, min_year)
    os.makedirs('csv', exist_ok=True)
    info_df.to_csv('csv/species_time_info.csv', index=False)
    pd.DataFrame({'species': kept_species}).to_csv('csv/species_post1900.csv', index=False)
    print(f"Kept species list saved to csv/species_post1900.csv ({len(kept_species)} species).")
    print("Time info saved to csv/species_time_info.csv.")
    clean_result_files(kept_species)
    print("Result files cleaned (species with pre-1900 data removed).")
    return kept_species

if __name__ == "__main__":
    main()
