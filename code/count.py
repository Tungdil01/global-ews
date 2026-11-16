#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from main_dl import load_combined_data, get_cached_taxonomic_group
from main_orchestrator import check_annual_series, EXCLUDED_GROUPS, min_records, probability_threshold, late_transition_window
FAIL_FILE = 'csv/failed_species.csv'
DL_FILE = 'csv/bifurcation_results.csv'
SHORT_FILE = 'csv/bifurcation_results_short.csv'
LARGE_FILE = 'csv/bifurcation_results_large.csv'
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

def get_failed_tasks_by_reason(reason_substring):
    if not os.path.exists(FAIL_FILE) or os.path.getsize(FAIL_FILE) == 0:
        return set()
    df = pd.read_csv(FAIL_FILE)
    if 'reason' not in df.columns:
        return set()
    mask = df['reason'].str.contains(reason_substring, case=False, na=False)
    return set(df.loc[mask, 'species'].str.strip())

def get_dl_dataframe():
    if not os.path.exists(DL_FILE) or os.path.getsize(DL_FILE) == 0:
        return pd.DataFrame()
    df = pd.read_csv(DL_FILE, low_memory=False)
    if 'dataset' in df.columns:
        df = df[df['dataset'].str.upper() != 'GBIF']
    if 'transition_year' in df.columns:
        df = df[df['transition_year'].notna() & (df['transition_year'] != '')]
    keep_cols = ['species', 'dataset', 'transition_year', 'best_probability']
    if 'timeseries_id' in df.columns:
        keep_cols = ['species', 'dataset', 'timeseries_id', 'transition_year', 'best_probability']
    return df[keep_cols].copy()

def get_short_dataframe():
    if not os.path.exists(SHORT_FILE) or os.path.getsize(SHORT_FILE) == 0:
        return pd.DataFrame()
    return pd.read_csv(SHORT_FILE, low_memory=False)

def get_full_data():
    datasets_dict, _, data_taxonomy, genus_taxonomy = load_combined_data(min_records=min_records, prefilter=False, min_time=1900)
    for ds_name in list(datasets_dict.keys()):
        df = datasets_dict[ds_name]
        if 'DATASET' in df.columns and 'dataset' not in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
        datasets_dict[ds_name] = df
    return datasets_dict, data_taxonomy, genus_taxonomy

def apply_late_removal(dl_df, datasets_dict, window_years):
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df_local = df.copy()
        df_local['TIME'] = pd.to_numeric(df_local['TIME'], errors='coerce')
        df_local['TIMESERIES_ID'] = df_local['TIMESERIES_ID'].astype(str).str.strip()
        g = df_local.groupby(['SPECIES_ID', 'TIMESERIES_ID'])['TIME'].max().reset_index()
        g.columns = ['species', 'timeseries_id', 'max_year']
        g['species'] = g['species'].astype(str).str.strip()
        g['dataset'] = ds_name
        frames.append(g)
    if not frames:
        return dl_df.iloc[0:0].copy()
    max_year_df = pd.concat(frames, ignore_index=True)
    if 'timeseries_id' in dl_df.columns:
        merged = pd.merge(dl_df, max_year_df, on=['species', 'dataset', 'timeseries_id'], how='left')
    else:
        max_year_agg = max_year_df.groupby(['species', 'dataset'])['max_year'].max().reset_index()
        merged = pd.merge(dl_df, max_year_agg, on=['species', 'dataset'], how='left')
    mask = (merged['transition_year'].notna()) & (merged['transition_year'] >= merged['max_year'] - window_years)
    merged = merged[~mask].copy()
    return merged.drop(columns=['max_year'])

def report_table_s1_counts():
    info_file = 'csv/species_time_info.csv'
    if not os.path.exists(info_file):
        print(f"\n[Table S1] {info_file} not found. Run only_year_1900.py to generate it.")
        return
    print(f"\n--- Table S1 Reconstruction from {info_file} ---")
    df = pd.read_csv(info_file)
    total_pops = len(df)
    total_sp = df['species'].nunique()
    df_20 = df[df['n_records'] >= 20]
    pops_20 = len(df_20)
    sp_20 = df_20['species'].nunique()
    unique_sp = df_20['species'].dropna().astype(str).str.strip().unique()
    sp_to_tax = {sp: get_cached_taxonomic_group(sp) for sp in unique_sp}
    df_20['tax_group'] = df_20['species'].astype(str).str.strip().map(sp_to_tax)
    df_20_tax = df_20[~df_20['tax_group'].isin(EXCLUDED_GROUPS)]
    pops_tax = len(df_20_tax)
    sp_tax = df_20_tax['species'].nunique()
    df_annual = df_20_tax[df_20_tax['keep'] == True]
    pops_annual = len(df_annual)
    sp_annual = df_annual['species'].nunique()
    print(f"Step 1 (raw, min_time=1900):   {total_pops} populations, {total_sp} species")
    print(f"Step 1b (min_records>=20):      {pops_20} populations, {sp_20} species")
    print(f"Step 2 (taxonomic exclusion):  {pops_tax} populations, {sp_tax} species")
    print(f"Step 3-6 (annual filters):     {pops_annual} populations, {sp_annual} species")
    print(f"----------------------------------------------------------------------")

def count_species_steps(dl_df, short_df):
    species_raw = get_failed_tasks_by_reason('taxonomic')
    print(f"Taxonomic exclusion removed (failed tasks log): {len(species_raw)} species")
    dl_species = set(dl_df['species'].str.strip())
    dl_tax = dl_species - species_raw
    print(f"Step 1 - species with DL transition (non-GBIF, taxonomically included): {len(dl_tax)}")
    dl_after_late = apply_late_removal(dl_df, get_full_data()[0], late_transition_window)
    dl_after_late_species = set(dl_after_late['species'].str.strip()) - species_raw
    print(f"Step 2 - species after late transition removal (window_years={late_transition_window}): {len(dl_after_late_species)}")
    dl_prob_species = set(dl_after_late[dl_after_late['best_probability'] >= probability_threshold]['species'].str.strip()) - species_raw
    print(f"Step 3 - Level 3 species with best_probability >= {probability_threshold} in DL_FILE: {len(dl_prob_species)}")
    short_species = set(short_df['species'].astype(str).str.strip())
    print(f"Step 3b - species in SHORT_FILE (final filtered set, prob >= {probability_threshold}): {len(short_species)}")
    return dl_prob_species

def report_iucn_breakdown(short_df):
    iucn_df = pd.read_csv('csv/iucn_species.csv', dtype=str, low_memory=False)
    iucn_df['species_short'] = iucn_df['species'].apply(preprocess_species_name)
    iucn_df['IUCN_Status_raw'] = iucn_df['iucn_status'].astype(str).fillna('').str.strip().str.upper()
    full_status_df = iucn_df.drop_duplicates(subset=['species_short']).dropna(subset=['species_short'])
    full_status_dict = dict(zip(full_status_df['species_short'], full_status_df['IUCN_Status_raw']))
    short_species = set(short_df['species'].astype(str).str.strip())
    short_species_binomials = {preprocess_species_name(sp) for sp in short_species}
    short_species_binomials = {sp for sp in short_species_binomials if sp is not None}
    threatened_categories = {'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED'}
    nonthreat_categories = {'LEAST_CONCERN', 'NEAR_THREATENED'}
    extinct_categories = {'EXTINCT', 'EXTINCT_IN_THE_WILD', 'EX', 'EW'}
    threat_count = sum(1 for sp in short_species_binomials if full_status_dict.get(sp, '') in threatened_categories)
    nonthreat_count = sum(1 for sp in short_species_binomials if full_status_dict.get(sp, '') in nonthreat_categories)
    extinct_species = [sp for sp in short_species_binomials if full_status_dict.get(sp, '') in extinct_categories]
    extinct_count = len(extinct_species)
    unknown_count = len(short_species_binomials) - threat_count - nonthreat_count - extinct_count
    print(f"Level 4 breakdown for short file (prob>={probability_threshold}):")
    print(f"  Threatened (VU, EN, CR): {threat_count}")
    print(f"  Non-threatened (LC, NT): {nonthreat_count}")
    print(f"  Unknown (NA/DD/absent): {unknown_count}")
    print(f"  Excluded (EX/EW): {extinct_count}")
    if not short_df.empty and 'taxonomic_group' in short_df.columns:
        species_to_group = dict(zip(short_df['species'].apply(preprocess_species_name), short_df['taxonomic_group']))
        for sp in extinct_species:
            group = species_to_group.get(sp, 'Unknown')
            print(f"    Extinct species: {sp} (Taxonomic group: {group})")
    print(f"  Total: {threat_count + nonthreat_count + unknown_count + extinct_count}")

def report_dataset_proportions(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return
    df = pd.read_csv(filepath, low_memory=False)
    counts = df['dataset'].value_counts()
    props = df['dataset'].value_counts(normalize=True) * 100
    print(f"Dataset proportions for {len(df)} species:")
    for dataset, count in counts.items():
        print(f"  {dataset}: {count} ({props[dataset]:.6f}%)")

def print_validation_counts():
    large_df = pd.read_csv(LARGE_FILE, low_memory=False)
    short_df = pd.read_csv(SHORT_FILE, low_memory=False)
    fail_df = pd.read_csv(FAIL_FILE, low_memory=False)
    large_species = set(large_df['species'].astype(str).str.strip())
    short_species = set(short_df['species'].astype(str).str.strip())
    fail_species = set(fail_df['species'].astype(str).str.strip())
    n_large = len(large_species)
    n_short = len(short_species)
    n_fail = len(fail_species)
    n_low = n_large - n_short
    n_overlap = len(large_species & fail_species)
    n_fail_only = n_fail - n_overlap
    n_total = n_large + n_fail_only
    print(f"[validation] Level 2 large={n_large} short={n_short} low_conf={n_low} fail_unique={n_fail} overlap={n_overlap} fail_only={n_fail_only} Level 1 total={n_total}")
    taxon_counts = large_df.drop_duplicates(subset=['species'])['taxonomic_group'].value_counts().sort_index()
    print(f"[validation] taxon_counts: " + " ".join(f"{k}={v}" for k, v in taxon_counts.items()) + f" sum={taxon_counts.sum()}")
    iucn_df = pd.read_csv('csv/iucn_species.csv', dtype=str, low_memory=False)
    iucn_df['species_short'] = iucn_df['species'].apply(preprocess_species_name)
    iucn_df['IUCN_Status_raw'] = iucn_df['iucn_status'].astype(str).fillna('').str.strip().str.upper()
    threatened_categories = {'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED'}
    nonthreat_categories = {'LEAST_CONCERN', 'NEAR_THREATENED'}
    extinct_categories = {'EXTINCT', 'EXTINCT_IN_THE_WILD', 'EX', 'EW'}
    full_status_df = iucn_df.drop_duplicates(subset=['species_short']).dropna(subset=['species_short'])
    full_status_dict = dict(zip(full_status_df['species_short'], full_status_df['IUCN_Status_raw']))
    short_species_binomials = {preprocess_species_name(sp) for sp in short_species}
    short_species_binomials = {sp for sp in short_species_binomials if sp is not None}
    n_threat = sum(1 for sp in short_species_binomials if full_status_dict.get(sp, '') in threatened_categories)
    n_nonthreat = sum(1 for sp in short_species_binomials if full_status_dict.get(sp, '') in nonthreat_categories)
    n_extinct = sum(1 for sp in short_species_binomials if full_status_dict.get(sp, '') in extinct_categories)
    n_unknown = len(short_species_binomials) - n_threat - n_nonthreat - n_extinct
    n_analysis = n_threat + n_nonthreat + n_unknown
    print(f"[validation] short unique binomials: {len(short_species_binomials)}")
    print(f"[validation] iucn_short: threatened={n_threat} non_threatened={n_nonthreat} unknown={n_unknown} excluded={n_extinct} n={n_analysis}")

def main():
    print("Counting unique species across all datasets except GBIF (BioTIME, RivFishTIME, Cedar Creek, GPDD, NPS, LPD, GLBA, VRData)")
    dl_df = get_dl_dataframe()
    short_df = get_short_dataframe()
    report_table_s1_counts()
    count_species_steps(dl_df, short_df)
    report_iucn_breakdown(short_df)
    report_dataset_proportions(SHORT_FILE)
    print_validation_counts()

if __name__ == "__main__":
    main()
