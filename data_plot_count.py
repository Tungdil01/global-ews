#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd

def parse_quarter(q):
    if pd.isna(q):
        return None
    try:
        return int(q)
    except ValueError:
        parts = str(q).split('/')
        if len(parts) == 2:
            try:
                q1, q2 = map(int, parts)
                avg = (q1 + q2) / 2.0
                return int(np.clip(np.rint(avg), 1, 4))
            except ValueError:
                return None
        return None

def find_column(df, synonyms):
    for col in df.columns:
        col_clean = col.strip().lower()
        for syn in synonyms:
            if col_clean == syn.strip().lower():
                return col
    return None

def load_biotime_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    required = ["YEAR", "ABUNDANCE", "VALID_NAME"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=["YEAR", "ABUNDANCE", "VALID_NAME"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    time_val = df["YEAR"].astype(float).copy()
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        mask_month = df["MONTH"].notna()
        month_adjust = (df.loc[mask_month, "MONTH"].astype(int).values - 1) / 12.0
        time_val.loc[mask_month] += month_adjust
        if "DAY" in df.columns:
            df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce")
            mask_day = df["DAY"].notna()
            mask = mask_month & mask_day
            day_adjust = (df.loc[mask, "DAY"].astype(int).values - 1) / 365.0
            time_val.loc[mask] += day_adjust
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["VALID_NAME"]
    df["DATASET"] = "BioTIME"
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]]

def load_rivfishtime_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    required = ["YEAR", "SPECIES", "ABUNDANCE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing RivFishTIME columns: {missing}")
    df = df.dropna(subset=required)
    time_val = df["YEAR"].copy()
    if "QUARTER" in df.columns:
        parsed_q = df["QUARTER"].apply(parse_quarter).fillna(2.5)
        time_val += (parsed_q - 1) / 4.0
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["SPECIES"]
    df["ABUNDANCE"] = pd.to_numeric(df["ABUNDANCE"], errors='coerce')
    df["DATASET"] = "RivFishTIME"
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]]

def load_cedar_creek_data(e120_plant_path, e120_arthropod_path, e244_path):
    all_dfs = []
    e120_plant_df = pd.read_csv(e120_plant_path, low_memory=False)
    year_col = find_column(e120_plant_df, ['year', 'YEAR', 'Year'])
    biomass_col = find_column(e120_plant_df, ['biomass (g/m2)', 'BIOMASS (G/M2)', 'Biomass (g/m2)', 'biomass', 'BIOMASS', 'Biomass', 'mass', 'MASS', 'Mass'])
    species_col = find_column(e120_plant_df, ['species', 'SPECIES', 'Species'])
    if year_col is None or biomass_col is None or species_col is None:
        raise ValueError("Missing columns in E120 plant data")
    e120_plant_df.rename(columns={year_col: 'YEAR', biomass_col: 'BIOMASS', species_col: 'SPECIES'}, inplace=True)
    e120_plant_df = e120_plant_df.dropna(subset=["YEAR", "BIOMASS"])
    e120_plant_df["YEAR"] = pd.to_numeric(e120_plant_df["YEAR"], errors="coerce")
    e120_plant_df["SPECIES"] = e120_plant_df["SPECIES"].str.strip()
    time_val = e120_plant_df["YEAR"].copy()
    if "MONTH" in e120_plant_df.columns:
        e120_plant_df["MONTH"] = pd.to_numeric(e120_plant_df["MONTH"], errors="coerce")
        mask_month = e120_plant_df["MONTH"].notna()
        time_val.loc[mask_month] = e120_plant_df.loc[mask_month, "YEAR"] + (e120_plant_df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
    e120_plant_df["TIME"] = time_val
    e120_plant_df["SPECIES_ID"] = e120_plant_df["SPECIES"]
    e120_plant_df["ABUNDANCE"] = pd.to_numeric(e120_plant_df["BIOMASS"], errors='coerce')
    e120_plant_df["DATASET"] = "CedarCreek"
    all_dfs.append(e120_plant_df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]])
    e120_arthropod_df = pd.read_csv(e120_arthropod_path, sep='\t', low_memory=False)
    year_col = find_column(e120_arthropod_df, ['year', 'YEAR', 'Year'])
    count_col = find_column(e120_arthropod_df, ['count', 'COUNT', 'Count'])
    genus_col = find_column(e120_arthropod_df, ['genus', 'GENUS', 'Genus'])
    epithet_col = find_column(e120_arthropod_df, ['specific.epithet', 'SPECIFIC.EPITHET', 'Specific.epithet', 'specificepithet', 'specific epithet'])
    if year_col is None or count_col is None or genus_col is None or epithet_col is None:
        raise ValueError("Missing columns in E120 arthropod data")
    e120_arthropod_df.rename(columns={year_col: 'YEAR', count_col: 'COUNT', genus_col: 'GENUS', epithet_col: 'SPECIFIC.EPITHET'}, inplace=True)
    e120_arthropod_df = e120_arthropod_df.dropna(subset=["YEAR", "COUNT"])
    e120_arthropod_df["YEAR"] = pd.to_numeric(e120_arthropod_df["YEAR"], errors="coerce")
    e120_arthropod_df["SPECIES"] = e120_arthropod_df["GENUS"] + " " + e120_arthropod_df["SPECIFIC.EPITHET"]
    e120_arthropod_df["SPECIES"] = e120_arthropod_df["SPECIES"].replace({"undet*": "Other"})
    time_val = e120_arthropod_df["YEAR"].copy()
    if "MONTH" in e120_arthropod_df.columns:
        e120_arthropod_df["MONTH"] = pd.to_numeric(e120_arthropod_df["MONTH"], errors="coerce")
        mask_month = e120_arthropod_df["MONTH"].notna()
        time_val.loc[mask_month] = e120_arthropod_df.loc[mask_month, "YEAR"] + (e120_arthropod_df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
    e120_arthropod_df["TIME"] = time_val
    e120_arthropod_df["SPECIES_ID"] = e120_arthropod_df["SPECIES"]
    e120_arthropod_df["ABUNDANCE"] = pd.to_numeric(e120_arthropod_df["COUNT"], errors='coerce')
    e120_arthropod_df["DATASET"] = "CedarCreek"
    all_dfs.append(e120_arthropod_df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]])
    e244_df = pd.read_csv(e244_path, low_memory=False)
    year_col = find_column(e244_df, ['year', 'YEAR', 'Year'])
    mass_col = find_column(e244_df, ['Mass g/m^2)', 'mass g/m^2)'])
    species_col = find_column(e244_df, ['species', 'SPECIES', 'Species'])
    if year_col is None or mass_col is None or species_col is None:
        raise ValueError("Missing columns in E244 data")
    e244_df.rename(columns={year_col: 'YEAR', mass_col: 'MASS', species_col: 'SPECIES'}, inplace=True)
    e244_df = e244_df.dropna(subset=["YEAR", "MASS"])
    e244_df["YEAR"] = pd.to_numeric(e244_df["YEAR"], errors="coerce")
    e244_df["SPECIES"] = e244_df["SPECIES"].str.strip()
    time_val = e244_df["YEAR"].copy()
    if "DATE" in e244_df.columns:
        try:
            dates = pd.to_datetime(e244_df["DATE"])
            time_val = dates.dt.year + (dates.dt.month - 1) / 12.0 + (dates.dt.day - 1) / 365.0
        except:
            pass
    e244_df["TIME"] = time_val
    e244_df["SPECIES_ID"] = e244_df["SPECIES"]
    e244_df["ABUNDANCE"] = pd.to_numeric(e244_df["MASS"], errors='coerce')
    e244_df["DATASET"] = "CedarCreek"
    all_dfs.append(e244_df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]])
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def load_gpdd_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    year_col = find_column(df, ['YEAR', 'YEAR'])
    species_col = find_column(df, ['SPECIES', 'SPECIES', 'SPECIES_ID'])
    abund_col = find_column(df, ['ABUNDANCE', 'ABUNDANCE'])
    if year_col is None or species_col is None or abund_col is None:
        raise ValueError("Missing required columns in GPDD data")
    df.rename(columns={year_col: 'YEAR', species_col: 'SPECIES_ID', abund_col: 'ABUNDANCE'}, inplace=True)
    df = df.dropna(subset=["YEAR", "SPECIES_ID", "ABUNDANCE"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["ABUNDANCE"] = pd.to_numeric(df["ABUNDANCE"], errors="coerce")
    df["TIME"] = df["YEAR"].astype(float)
    df["DATASET"] = "GPDD"
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "DATASET"]]

def compute_species_statistics():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    df_short = pd.read_csv('csv/bifurcation_results_short.csv')
    df_filtered = df_short[df_short['best_probability'] >= 0.7]
    if 'taxonomic_group' in df_filtered.columns:
        tax_counts = df_filtered['taxonomic_group'].fillna('Unknown').value_counts()
        print("Taxonomic group counts:")
        for group, cnt in tax_counts.items():
            print(f"  {group}: {cnt}")
        # Aggregate into plants, invertebrates, vertebrates
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
                # assume invertebrate (all animal groups not in vertebrates, plus non-animal groups are excluded by filters)
                invertebrate_count += cnt
        print(f"Plant species: {plant_count}")
        print(f"Invertebrate species: {invertebrate_count}")
        print(f"Vertebrate species: {vertebrate_count}")
    else:
        print("No 'taxonomic_group' column found in bifurcation_results_short.csv")
    dataset_counts = df_filtered['dataset'].value_counts()
    total_species = len(df_filtered)
    print("Species per dataset (probability >= 0.7):")
    for ds, cnt in dataset_counts.items():
        pct = (cnt / total_species) * 100
        print(f"  {ds}: {cnt} ({pct:.2g}%)")
    species_dataset_pairs = set(zip(df_filtered['species'].str.strip(), df_filtered['dataset'].str.strip()))
    biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
    bio_df = load_biotime_data(biotime_path)
    riv_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
    riv_df = load_rivfishtime_data(riv_path)
    cedar_paths = {
        'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
        'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
        'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')
    }
    cedar_df = load_cedar_creek_data(cedar_paths['e120_plant'], cedar_paths['e120_arthropod'], cedar_paths['e244'])
    gpdd_path = os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
    gpdd_df = load_gpdd_data(gpdd_path)
    all_data = pd.concat([bio_df, riv_df, cedar_df, gpdd_df], ignore_index=True)
    all_data['SPECIES_ID'] = all_data['SPECIES_ID'].str.strip()
    all_data['DATASET'] = all_data['DATASET'].str.strip()
    mask = all_data.apply(lambda row: (row['SPECIES_ID'], row['DATASET']) in species_dataset_pairs, axis=1)
    all_data_filtered = all_data[mask]
    if all_data_filtered.empty:
        print("No data found for the selected species.")
        return
    all_data_filtered['YEAR_INT'] = np.floor(all_data_filtered['TIME']).astype(int)
    annual = all_data_filtered.groupby(['SPECIES_ID', 'DATASET', 'YEAR_INT'])['ABUNDANCE'].mean().reset_index()
    year_counts = annual.groupby(['SPECIES_ID', 'DATASET']).size().reset_index(name='num_years')
    n_species = len(year_counts)
    if n_species == 0:
        min_years = 0
        max_years = 0
    else:
        min_years = year_counts['num_years'].min()
        max_years = year_counts['num_years'].max()
    print(f"Number of species: {n_species}")
    print(f"Minimum number of years: {min_years}")
    print(f"Maximum number of years: {max_years}")
    dist = year_counts['num_years'].value_counts().sort_index()
    print("\nDistribution of species by number of years:")
    for span, count in dist.items():
        print(f"  {span} years: {count} species")

def main():
    compute_species_statistics()

if __name__ == "__main__":
    main()
