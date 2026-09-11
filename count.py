#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
from main_dl import load_combined_data, build_species_whitelist
from main_orchestrator import check_annual_series, EXCLUDED_GROUPS

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
rivfishtime_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
cedar_creek_paths = {
    'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
    'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
    'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')
}
gpdd_path = os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
FAIL_FILE = 'csv/failed_species.csv'
DL_FILE = 'csv/bifurcation_results_dl_ews.csv'
SHORT_FILE = 'csv/bifurcation_results_short.csv'
POST1900_FILE = 'csv/species_post1900.csv'

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    parts = str(species_name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return species_name

def load_iucn_species(iucn_path='csv/iucn_species.csv'):
    df = pd.read_csv(iucn_path, dtype=str, low_memory=False)
    df['species_short'] = df['species'].apply(preprocess_species_name)
    df['IUCN_Status_raw'] = df['iucn_status'].astype(str).fillna('').str.strip()
    valid_statuses = ['LEAST_CONCERN', 'NEAR_THREATENED', 'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED']
    iucn_present = df[df['IUCN_Status_raw'].str.upper().str.strip().isin(valid_statuses)].copy()
    dedup = iucn_present[['species_short', 'IUCN_Status_raw']].drop_duplicates(subset=['species_short']).reset_index(drop=True)
    status_dict = dict(zip(dedup['species_short'], dedup['IUCN_Status_raw']))
    print(f"[IUCN] Loaded {len(status_dict)} species with valid IUCN status (LC, NT, VU, EN, CR)")
    return status_dict

def find_column_in_chunk(chunk, synonyms):
    cols_lower = {col.strip().lower(): col for col in chunk.columns}
    for syn in synonyms:
        syn_lower = syn.strip().lower()
        if syn_lower in cols_lower:
            return cols_lower[syn_lower]
    return None

def count_records_in_file(filepath, sep, year_synonyms, species_config, skiprows=None, chunksize=100000):
    counts = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return counts
    try:
        header_df = pd.read_csv(filepath, sep=sep, skiprows=skiprows, nrows=0)
    except Exception as e:
        print(f"Could not read header from {filepath}: {e}")
        return counts
    print(f"Columns in {os.path.basename(filepath)}: {list(header_df.columns)}")
    for chunk in pd.read_csv(filepath, sep=sep, skiprows=skiprows, chunksize=chunksize,
                             quotechar='"', on_bad_lines='warn', engine='python'):
        year_col = find_column_in_chunk(chunk, year_synonyms)
        if year_col is None:
            continue
        if isinstance(species_config, str):
            species_col = find_column_in_chunk(chunk, [species_config])
            if species_col is None:
                continue
            species_series = chunk[species_col].astype(str).str.strip()
        else:
            genus_col = find_column_in_chunk(chunk, species_config[0])
            epithet_col = find_column_in_chunk(chunk, species_config[1])
            if genus_col is None or epithet_col is None:
                continue
            genus_series = chunk[genus_col].astype(str).str.strip()
            epithet_series = chunk[epithet_col].astype(str).str.strip()
            species_series = genus_series + ' ' + epithet_series
            species_series = species_series.replace({'undet*': 'Other'})
        valid = chunk[year_col].notna() & species_series.notna()
        years = pd.to_numeric(chunk.loc[valid, year_col], errors='coerce')
        valid = valid & years.notna()
        if not valid.any():
            continue
        sp_list = species_series[valid].tolist()
        for sp in sp_list:
            if sp and sp != 'nan':
                counts[sp] = counts.get(sp, 0) + 1
    return counts

def count_raw_species_with_min_records(min_records):
    all_counts = {}
    counts = count_records_in_file(biotime_path, ',', ['year'], 'valid_name')
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    counts = count_records_in_file(rivfishtime_path, ',', ['year'], 'Species')
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    counts = count_records_in_file(cedar_creek_paths['e120_plant'], ',', ['year'], 'Species')
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    counts = count_records_in_file(cedar_creek_paths['e120_arthropod'], '\t', ['year'], (['genus'], ['specific.epithet', 'specificepithet']))
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    counts = count_records_in_file(cedar_creek_paths['e244'], ',', ['year'], 'Species')
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    counts = count_records_in_file(gpdd_path, ',', ['year'], 'Species')
    for sp, cnt in counts.items():
        all_counts[sp] = all_counts.get(sp, 0) + cnt
    return {sp for sp, cnt in all_counts.items() if cnt >= min_records}

def get_failed_tasks_by_reason(reason_substring):
    if not os.path.exists(FAIL_FILE) or os.path.getsize(FAIL_FILE) == 0:
        return set()
    df = pd.read_csv(FAIL_FILE)
    if 'reason' not in df.columns:
        return set()
    mask = df['reason'].str.contains(reason_substring, case=False, na=False)
    return set(df.loc[mask, 'species'].str.strip())

def count_from_result_file(filepath, non_gbif_only=True):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return set()
    df = pd.read_csv(filepath, low_memory=False)
    if 'species' not in df.columns:
        return set()
    if non_gbif_only and 'dataset' in df.columns:
        df = df[df['dataset'].str.upper() != 'GBIF']
    return set(df['species'].str.strip())

def get_dl_dataframe():
    if not os.path.exists(DL_FILE) or os.path.getsize(DL_FILE) == 0:
        return pd.DataFrame()
    df = pd.read_csv(DL_FILE, low_memory=False)
    if 'dataset' in df.columns:
        df = df[df['dataset'].str.upper() != 'GBIF']
    if 'transition_year' in df.columns:
        df = df[df['transition_year'].notna() & (df['transition_year'] != '')]
    return df[['species', 'dataset', 'transition_year', 'best_probability']].copy()

def get_whitelist():
    if os.path.exists(POST1900_FILE) and os.path.getsize(POST1900_FILE) > 0:
        df = pd.read_csv(POST1900_FILE)
        if 'species' in df.columns:
            return set(df['species'].tolist())
    print("Building whitelist from BioTIME...")
    whitelist = build_species_whitelist(biotime_path, min_year=1900, min_records=20)
    os.makedirs('csv', exist_ok=True)
    pd.DataFrame({'species': list(whitelist)}).to_csv(POST1900_FILE, index=False)
    print(f"Built and saved whitelist of {len(whitelist)} species.")
    return whitelist

def get_full_data():
    whitelist = get_whitelist()
    full_data, _, _, _ = load_combined_data(
        biotime_path, rivfishtime_path, cedar_creek_paths, None, gpdd_path,
        min_records=20, use_existing_biotime=True, prefilter=False,
        min_time=1900, whitelist=whitelist
    )
    if 'DATASET' in full_data.columns and 'dataset' not in full_data.columns:
        full_data = full_data.rename(columns={'DATASET': 'dataset'})
    return full_data

def apply_late_removal(dl_df, full_data, window_years=3):
    full_data['TIME'] = pd.to_numeric(full_data['TIME'], errors='coerce')
    max_year_df = full_data.groupby(['SPECIES_ID', 'dataset'])['TIME'].max().reset_index()
    max_year_df.columns = ['species', 'dataset', 'max_year']
    merged = pd.merge(dl_df, max_year_df, on=['species', 'dataset'], how='left')
    mask = (merged['transition_year'].notna()) & (merged['transition_year'] >= merged['max_year'] - window_years)
    merged = merged[~mask].copy()
    return merged.drop(columns=['max_year'])

def main():
    min_records = 20
    prob_threshold = 0.7
    window_years = 3
    print("Counting unique species across all datasets except GBIF (BioTIME, RivFishTIME, Cedar Creek, GPDD)")
    species_raw = count_raw_species_with_min_records(min_records)
    print(f"Step 0 - min_records >= {min_records}: {len(species_raw)} unique species")
    tax_excl_species = get_failed_tasks_by_reason('taxonomic')
    tax_excl_count = len(tax_excl_species)
    print(f"Taxonomic exclusion removed: {tax_excl_count} species")
    dl_df = get_dl_dataframe()
    dl_species = set(dl_df['species'].str.strip())
    dl_tax = dl_species - tax_excl_species
    print(f"Step 1 - species with DL transition (non-GBIF, taxonomically included): {len(dl_tax)}")
    full_data = get_full_data()
    dl_after_late = apply_late_removal(dl_df, full_data, window_years)
    dl_after_late_species = set(dl_after_late['species'].str.strip()) - tax_excl_species
    print(f"Step 2 - species after late transition removal (window_years={window_years}): {len(dl_after_late_species)}")
    prob_species = set(dl_after_late[dl_after_late['best_probability'] >= prob_threshold]['species'].str.strip()) - tax_excl_species
    print(f"Step 3 - species with best_probability >= {prob_threshold}: {len(prob_species)}")
    iucn_dict = load_iucn_species('csv/iucn_species.csv')
    iucn_set = set(iucn_dict.keys())
    final_species = prob_species.intersection(iucn_set)
    print(f"Step 4 - with IUCN status: {len(final_species)}")
    threatened_categories = {'NEAR_THREATENED', 'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED'}
    threat_count = 0
    nonthreat_count = 0
    for sp in prob_species:
        if sp in iucn_dict:
            status = iucn_dict[sp].upper().strip()
            if status in threatened_categories:
                threat_count += 1
            elif status == 'LEAST_CONCERN':
                nonthreat_count += 1
    unknown_count = len(prob_species) - len(prob_species.intersection(iucn_set))
    print(f"Breakdown for prob>={prob_threshold}:")
    print(f"  Threatened (NT, VU, EN, CR): {threat_count}")
    print(f"  Non-threatened (LC): {nonthreat_count}")
    print(f"  Unknown (no IUCN): {unknown_count}")

if __name__ == "__main__":
    main()
