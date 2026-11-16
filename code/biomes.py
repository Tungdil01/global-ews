import os
import re
import difflib
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

CANONICAL_BIOMES = [
    'Temperate Broadleaf & Mixed Forests', 'Temperate Coniferous Forests', 'Boreal Forests/Taiga', 'Deserts & Xeric Shrublands', 'Tropical & Subtropical Moist Broadleaf Forests', 'Tropical & Subtropical Dry Broadleaf Forests', 'Tropical & Subtropical Grasslands, Savannas & Shrublands', 'Temperate Grasslands, Savannas & Shrublands', 'Mediterranean Forests, Woodlands & Scrub', 'Montane Grasslands & Shrublands', 'Flooded Grasslands & Savannas', 'Mangroves', 'Tundra', 'Polar Ecoregions', 'Temperate Shelf & Seas', 'Tropical Seas (Non-coral)', 'Temperate Upwelling', 'Small River Ecosystems', 'Small Lake Ecosystems', 'Large River Ecosystems', 'Large River Delta Ecosystems', 'Large Lake Ecosystems', 'Tropical Coral Reefs', 'Mixed/Generalist Habitats', 'Unknown/Unclassified'
]

def preprocess_biome_string(s):
    if pd.isna(s):
        return ''
    s = str(s).lower()
    s = s.replace('&', 'and')
    s = re.sub(r'\s+', ' ', s.strip())
    return s

def get_canonical_biome(biome_map, habitat):
    if pd.isna(biome_map) or biome_map == '' or biome_map == 'NA':
        return 'Unknown/Unclassified'
    if habitat is not None and pd.notna(habitat) and 'reef' in str(habitat).lower():
        if 'tropical seas' in str(biome_map).lower():
            return 'Tropical Coral Reefs'
    if 'coral' in str(biome_map).lower():
        return 'Tropical Coral Reefs'
    raw_clean = preprocess_biome_string(biome_map)
    canonical_clean = [preprocess_biome_string(b) for b in CANONICAL_BIOMES]
    matches = difflib.get_close_matches(raw_clean, canonical_clean, n=1, cutoff=0.8)
    if matches:
        idx = canonical_clean.index(matches[0])
        return CANONICAL_BIOMES[idx]
    manual_map = {
        'multiple ecoregions': 'Mixed/Generalist Habitats', 'mixed': 'Mixed/Generalist Habitats', 'various': 'Mixed/Generalist Habitats', }
    if raw_clean in manual_map:
        return manual_map[raw_clean]
    return 'Unknown/Unclassified'

def mode_or_first(x):
    return x.mode()[0] if not x.mode().empty else x.iloc[0]

def load_and_merge(input_bifurcation, input_rawdata, input_metadata):
    df_bif = pd.read_csv(input_bifurcation)
    if 'species' not in df_bif.columns:
        raise KeyError("Bifurcation file missing 'species' column.")
    print(f"Bifurcation: {len(df_bif)} rows, {df_bif['species'].nunique()} unique species")
    try:
        df_raw = pd.read_csv(input_rawdata, encoding='latin-1', usecols=['valid_name', 'STUDY_ID', 'LATITUDE', 'LONGITUDE'])
    except UnicodeDecodeError:
        try:
            df_raw = pd.read_csv(input_rawdata, encoding='ISO-8859-1', usecols=['valid_name', 'STUDY_ID', 'LATITUDE', 'LONGITUDE'])
        except:
            df_raw = pd.read_csv(input_rawdata, usecols=['valid_name', 'STUDY_ID', 'LATITUDE', 'LONGITUDE'])
    df_raw = df_raw.dropna(subset=['LATITUDE', 'LONGITUDE', 'STUDY_ID'])
    df_raw['STUDY_ID'] = pd.to_numeric(df_raw['STUDY_ID'], errors='coerce').astype('Int64')
    df_raw = df_raw.dropna(subset=['STUDY_ID'])
    print(f"Rawdata: {len(df_raw)} rows, {df_raw['valid_name'].nunique()} unique species")
    df_raw_grouped = df_raw.groupby('valid_name').agg({'STUDY_ID': mode_or_first, 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
    df = pd.merge(df_bif, df_raw_grouped, left_on='species', right_on='valid_name', how='left')
    matched_species = df['valid_name'].notna().sum()
    print(f"Species matched with rawdata: {matched_species} / {len(df)}")
    try:
        df_meta = pd.read_csv(input_metadata, encoding='latin-1', usecols=['STUDY_ID', 'HABITAT', 'BIOME_MAP'])
    except UnicodeDecodeError:
        try:
            df_meta = pd.read_csv(input_metadata, encoding='ISO-8859-1', usecols=['STUDY_ID', 'HABITAT', 'BIOME_MAP'])
        except:
            df_meta = pd.read_csv(input_metadata, usecols=['STUDY_ID', 'HABITAT', 'BIOME_MAP'])
    df_meta['STUDY_ID'] = pd.to_numeric(df_meta['STUDY_ID'], errors='coerce').astype('Int64')
    df_meta = df_meta.dropna(subset=['STUDY_ID'])
    df = pd.merge(df, df_meta, on='STUDY_ID', how='left', suffixes=('', '_meta'))
    print(f"Rows with HABITAT after merge: {df['HABITAT'].notna().sum()} / {len(df)}")
    print(f"Rows with BIOME_MAP after merge: {df['BIOME_MAP'].notna().sum()} / {len(df)}")
    return df

def assign_koppen_zone(lat, lon, gdf_koppen):
    if pd.isna(lat) or pd.isna(lon):
        return None
    point = Point(lon, lat)
    matches = gdf_koppen[gdf_koppen.intersects(point)]
    if not matches.empty:
        return matches.iloc[0]['GRIDCODE']
    return None

def assign_meow_zone(lat, lon, gdf_meow):
    if pd.isna(lat) or pd.isna(lon):
        return None
    point = Point(lon, lat)
    matches = gdf_meow[gdf_meow.intersects(point)]
    if not matches.empty:
        row = matches.iloc[0]
        if 'REALM' in row.index:
            return row['REALM']
        elif 'ECOREGION' in row.index:
            return row['ECOREGION']
        else:
            return 'Unknown'
    return None

def apply_cleaning(df):
    df['biome'] = df.apply(lambda row: get_canonical_biome(row['BIOME_MAP'], row['HABITAT']), axis=1)
    return df

def save_output(df, output_path):
    cols = ['species', 'LATITUDE', 'LONGITUDE', 'HABITAT', 'BIOME_MAP', 'biome', 'koppen_zone', 'meow_zone']
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df[cols].to_csv(output_path, index=False)

def print_validation_counts(output_csv):
    print("\n" + "="*60)
    print("BIOMES VALIDATION COUNTS")
    print("="*60)
    df = pd.read_csv(output_csv)
    n_coords = int(df[['LATITUDE', 'LONGITUDE']].notna().all(axis=1).sum())
    n_koppen = int(df['koppen_zone'].notna().sum())
    n_meow = int(df['meow_zone'].notna().sum())
    print(f"[val] biomes.csv rows: {int(len(df))}")
    print(f"[val] unique species: {int(df['species'].nunique())}")
    print(f"[val] species with both coords: {n_coords}")
    print(f"[val] LATITUDE notnull: {int(df['LATITUDE'].notna().sum())}")
    print(f"[val] LONGITUDE notnull: {int(df['LONGITUDE'].notna().sum())}")
    print(f"[val] HABITAT notnull: {int(df['HABITAT'].notna().sum())}")
    print(f"[val] BIOME_MAP notnull: {int(df['BIOME_MAP'].notna().sum())}")
    print(f"[val] biome notnull: {int(df['biome'].notna().sum())}")
    print(f"[val] koppen_zone notnull: {n_koppen} unique={int(df['koppen_zone'].nunique())} missing_with_coords={n_coords - n_koppen}")
    print(f"[val] meow_zone notnull: {n_meow} unique={int(df['meow_zone'].nunique())} missing_with_coords={n_coords - n_meow}")
    print("[val] biome distribution:")
    for b, c in df['biome'].value_counts().items():
        print(f"[val]   {b}: {int(c)}")
    top_koppen = df['koppen_zone'].value_counts().head(3)
    print(f"[val] top koppen: {top_koppen.to_dict()}")
    top_meow = df['meow_zone'].value_counts().head(3)
    print(f"[val] top meow: {top_meow.to_dict()}")
    print("="*60 + "\n")

def main():
    input_bifurcation = 'csv/bifurcation_results_short.csv'
    input_rawdata = 'data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv'
    input_metadata = 'data/biotime_v2_full_2025/biotime_v2_metadata_2025.csv'
    output_csv = 'csv/biomes.csv'
    koppen_shapefile = 'data/koppen/1976-2000_GIS/1976-2000.shp'
    meow_shapefile = 'data/meow/Marine_Ecoregions_of_the_World/data/commondata/data0/meow_ecos_expl_clipped_expl.shp'
    df = load_and_merge(input_bifurcation, input_rawdata, input_metadata)
    df = apply_cleaning(df)
    if os.path.exists(koppen_shapefile):
        gdf_koppen = gpd.read_file(koppen_shapefile)
        if gdf_koppen.crs is None:
            gdf_koppen = gdf_koppen.set_crs('EPSG:4326')
        elif str(gdf_koppen.crs) != 'EPSG:4326':
            gdf_koppen = gdf_koppen.to_crs('EPSG:4326')
        tqdm.pandas(desc="Assigning Koppen zones")
        df['koppen_zone'] = df.progress_apply(lambda row: assign_koppen_zone(row['LATITUDE'], row['LONGITUDE'], gdf_koppen), axis=1)
    else:
        print(f"Koppen shapefile not found at {koppen_shapefile}, skipping.")
        df['koppen_zone'] = None
    if os.path.exists(meow_shapefile):
        gdf_meow = gpd.read_file(meow_shapefile)
        if gdf_meow.crs is None:
            gdf_meow = gdf_meow.set_crs('EPSG:4326')
        elif str(gdf_meow.crs) != 'EPSG:4326':
            gdf_meow = gdf_meow.to_crs('EPSG:4326')
        tqdm.pandas(desc="Assigning MEOW zones")
        df['meow_zone'] = df.progress_apply(lambda row: assign_meow_zone(row['LATITUDE'], row['LONGITUDE'], gdf_meow), axis=1)
    else:
        print(f"MEOW shapefile not found at {meow_shapefile}, skipping.")
        df['meow_zone'] = None
    save_output(df, output_csv)
    print("Output saved to", output_csv)
    print_validation_counts(output_csv)

if __name__ == '__main__':
    main()
