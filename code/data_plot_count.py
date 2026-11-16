#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from main_dl import load_combined_data, get_cached_taxonomic_group
from main_orchestrator import probability_threshold

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    parts = str(species_name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return species_name

def load_combined_data_for_stats():
    datasets_dict, _, _, _ = load_combined_data(min_records=1, prefilter=False, min_time=1900)
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df = df.copy()
        if 'DATASET' in df.columns and 'dataset' not in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
        if 'dataset' not in df.columns:
            df['dataset'] = ds_name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['SPECIES_ID', 'dataset', 'TIME', 'ABUNDANCE'])
    return pd.concat(frames, ignore_index=True)

def report_taxonomic_counts(df_species):
    if 'taxonomic_group' not in df_species.columns:
        print("No 'taxonomic_group' column found.")
        return
    tax_counts = df_species['taxonomic_group'].fillna('Unknown').value_counts()
    print("Taxonomic group counts (unique species):")
    for group, cnt in tax_counts.items():
        print(f"  {group}: {cnt}")
    plant_groups = {'Tracheophyta', 'Bryophyta', 'Chlorophyta'}
    vertebrate_groups = {'Aves', 'Mammalia', 'Lissamphibia', 'Actinopterygii', 'Chondrichthyes', 'Agnatha', 'Squamata', 'Crocodylia', 'Testudines', 'Sphenodontia'}
    plant_count = 0
    vertebrate_count = 0
    invertebrate_count = 0
    for group, cnt in tax_counts.items():
        if group in plant_groups:
            plant_count += cnt
        elif group in vertebrate_groups:
            vertebrate_count += cnt
        else:
            invertebrate_count += cnt
    print(f"Plant species: {plant_count}")
    print(f"Invertebrate species: {invertebrate_count}")
    print(f"Vertebrate species: {vertebrate_count}")
    print(f"Sum: {plant_count + invertebrate_count + vertebrate_count}")

def report_dataset_counts(used_df, failed_df):
    used_pop_counts = used_df['dataset'].value_counts()
    used_sp_counts = used_df.groupby('dataset')['species'].nunique()
    failed_sp_counts = failed_df.groupby('dataset')['species'].nunique()
    print(f"\n=== Per-dataset breakdown ===")
    print(f"{'Dataset':<14} {'USED(pop)':>10} {'USED(sp)':>9} {'FAIL(sp)':>9}")
    print("-" * 50)
    all_ds = set(used_pop_counts.index) | set(used_sp_counts.index) | set(failed_sp_counts.index)
    for ds in sorted(all_ds):
        u_pop = int(used_pop_counts.get(ds, 0))
        u_sp = int(used_sp_counts.get(ds, 0))
        f_sp = int(failed_sp_counts.get(ds, 0))
        print(f"{ds:<14} {u_pop:>10} {u_sp:>9} {f_sp:>9}")
    print("-" * 50)
    print(f"{'SUM':<14} {int(used_pop_counts.sum()):>10} {int(used_sp_counts.sum()):>9} {int(failed_sp_counts.sum()):>9}")

def report_years_distribution(all_data, species_dataset_pairs, label):
    all_data['SPECIES_ID'] = all_data['SPECIES_ID'].astype(str).str.strip()
    all_data['dataset'] = all_data['dataset'].astype(str).str.strip()
    mask = all_data.apply(lambda row: (row['SPECIES_ID'], row['dataset']) in species_dataset_pairs, axis=1)
    all_data_filtered = all_data[mask]
    if all_data_filtered.empty:
        print(f"[{label}] No data found for the selected species.")
        return
    all_data_filtered = all_data_filtered.copy()
    all_data_filtered['YEAR_INT'] = np.floor(all_data_filtered['TIME']).astype(int)
    annual = all_data_filtered.groupby(['SPECIES_ID', 'dataset', 'YEAR_INT'])['ABUNDANCE'].mean().reset_index()
    year_counts = annual.groupby(['SPECIES_ID', 'dataset']).size().reset_index(name='num_years')
    n_pairs = len(year_counts)
    min_years = year_counts['num_years'].min()
    max_years = year_counts['num_years'].max()
    print(f"[{label}] Number of species-dataset pairs: {n_pairs}")
    print(f"[{label}] Minimum number of years with data: {min_years}")
    print(f"[{label}] Maximum number of years with data: {max_years}")

def report_validation_counts(large_df, short_df, failed_df):
    large_species = set(large_df['species'].astype(str).str.strip())
    short_species = set(short_df['species'].astype(str).str.strip())
    fail_species = set(failed_df['species'].astype(str).str.strip())
    n_large = len(large_species)
    n_short = len(short_species)
    n_fail = len(fail_species)
    n_low = n_large - n_short
    n_overlap = len(large_species & fail_species)
    n_fail_only = n_fail - n_overlap
    n_total = n_large + n_fail_only
    print(f"[validation] large={n_large} short={n_short} low_conf={n_low} fail_unique={n_fail} overlap={n_overlap} fail_only={n_fail_only} total={n_total}")

def compute_species_statistics():
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_df['dataset'] = large_df['dataset'].astype(str).str.strip()
    large_df = large_df[large_df['taxonomic_group'] != 'Other'].copy()
    large_species_set = set(large_df['species'])
    short_df = pd.read_csv('csv/bifurcation_results_short.csv')
    short_df['species'] = short_df['species'].astype(str).str.strip()
    short_df['dataset'] = short_df['dataset'].astype(str).str.strip()
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    failed_df['dataset'] = failed_df['dataset'].astype(str).str.strip()
    failed_df['taxonomic_group'] = failed_df['species'].apply(get_cached_taxonomic_group)
    failed_only_df = failed_df[~failed_df['species'].isin(large_species_set)].copy()
    print(f"=== Core counts ===")
    print(f"USED populations (rows):     {len(large_df)}")
    print(f"USED unique species:         {large_df['species'].nunique()}")
    print(f"SHORT populations (rows):    {len(short_df)}")
    print(f"SHORT unique species:        {short_df['species'].nunique()}")
    print(f"FAILED populations (rows):   {len(failed_only_df)}")
    print(f"FAILED unique species:       {failed_only_df['species'].nunique()}")
    short_unique_species = short_df[['species', 'taxonomic_group']].drop_duplicates(subset=['species']).copy()
    short_unique_species = short_unique_species[short_unique_species['taxonomic_group'] != 'Other'].copy()
    report_taxonomic_counts(short_unique_species)
    report_dataset_counts(short_df, failed_only_df)
    report_validation_counts(large_df, short_df, failed_df)
    all_data = load_combined_data_for_stats()
    species_dataset_pairs_large = set(zip(large_df['species'], large_df['dataset']))
    species_dataset_pairs_short = set(zip(short_df['species'], short_df['dataset']))
    report_years_distribution(all_data, species_dataset_pairs_large, label=f"large n={large_df['species'].nunique()}")
    report_years_distribution(all_data, species_dataset_pairs_short, label=f"short n={short_df['species'].nunique()}")

def main():
    compute_species_statistics()

if __name__ == "__main__":
    main()
