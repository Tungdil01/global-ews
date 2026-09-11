import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import traceback

plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'axes.titlesize': 18,
    'font.size': 18,
    'lines.linewidth': 1.5
})

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

def apply_tipping_point_probability_threshold(df, probability_threshold=0.3):
    if 'raw_tipping_point_type' in df.columns and 'best_probability' in df.columns:
        df['tipping_point_type'] = df.apply(
            lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= probability_threshold else 'no-bifurcation',
            axis=1
        )
    return df

def get_top_taxonomic_groups(probability_threshold):
    freq_df = pd.read_csv('csv/bifurcation_results_short.csv')
    freq_df = apply_tipping_point_probability_threshold(freq_df, probability_threshold)
    taxon_counts = freq_df['taxonomic_group'].value_counts()
    if 'Other' in taxon_counts.index:
        taxon_counts = taxon_counts.drop('Other')
    top_taxa = taxon_counts.head(20).index.tolist()
    distinct_colors = [
        '#4363d8', '#e6194b', '#ffe119', '#911eb4', '#3cb44b', '#f58231', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#000075', '#ffd8b1', '#808080'
    ]
    specified_colors = {
        'Hexapoda': '#e6194b',
        'Tracheophyta': '#3cb44b',
        'Bryophyta': '#808000',
        'Actinopterygii': '#4363d8',
        'Chlorophyta': '#bcf60c'
    }
    available = distinct_colors[:]
    used_specified = set()
    for group in top_taxa:
        if group in specified_colors:
            color = specified_colors[group]
            if color in available:
                available.remove(color)
                used_specified.add(color)
    taxonomic_colors = {}
    for group in top_taxa:
        if group in specified_colors:
            taxonomic_colors[group] = specified_colors[group]
        else:
            if available:
                taxonomic_colors[group] = available.pop(0)
            else:
                taxonomic_colors[group] = '#cccccc'
    return top_taxa, taxonomic_colors

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

def load_gpdd_locations(gpdd_data_path, gpdd_main_path, gpdd_taxon_path, gpdd_location_path):
    data_df = pd.read_csv(gpdd_data_path, low_memory=False)
    data_df = data_df[['MainID']].drop_duplicates()
    main_df = pd.read_csv(gpdd_main_path, low_memory=False)
    main_df = main_df[['MainID', 'TaxonID', 'LocationID']].drop_duplicates()
    taxon_df = pd.read_csv(gpdd_taxon_path, low_memory=False)
    taxon_df = taxon_df[['TaxonID', 'TaxonName']].drop_duplicates()
    loc_df = pd.read_csv(gpdd_location_path, low_memory=False)
    loc_df = loc_df[['LocationID', 'LongDD', 'LatDD']].dropna(subset=['LongDD', 'LatDD'])
    merged = data_df.merge(main_df, on='MainID', how='inner')
    merged = merged.merge(taxon_df, on='TaxonID', how='inner')
    merged = merged.merge(loc_df, on='LocationID', how='inner')
    merged['valid_name'] = merged['TaxonName'].str.strip()
    merged['LATITUDE'] = merged['LatDD']
    merged['LONGITUDE'] = merged['LongDD']
    return merged[['valid_name', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

def load_combined_data_for_plots():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
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
    combined = pd.concat([bio_df, riv_df, cedar_df, gpdd_df], ignore_index=True)
    combined = combined[combined['TIME'] >= 1900]
    return combined

def load_biotime_locations(filepath):
    df = pd.read_csv(filepath, usecols=['valid_name', 'LATITUDE', 'LONGITUDE'], encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    return df

def load_rivfishtime_locations(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns or 'SPECIES' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'SPECIES'])
    df['SPECIES'] = df['SPECIES'].str.strip()
    df['valid_name'] = df['SPECIES']
    return df[['valid_name', 'LATITUDE', 'LONGITUDE']]

def plot_time_series_ranges(top_taxa, taxonomic_colors):
    freq_df = pd.read_csv('csv/bifurcation_results_short.csv')
    species_to_taxon = dict(zip(freq_df['species'], freq_df['taxonomic_group']))
    species_list = set(freq_df['species'].unique())
    all_data = load_combined_data_for_plots()
    all_data['SPECIES_ID'] = all_data['SPECIES_ID'].str.strip()
    all_data = all_data[all_data['SPECIES_ID'].isin(species_list)]
    time_ranges = []
    for species, group in all_data.groupby('SPECIES_ID'):
        if not group.empty:
            min_time = group['TIME'].min()
            max_time = group['TIME'].max()
            taxon = species_to_taxon.get(species, 'Other')
            time_ranges.append({
                'species': species, 'start': min_time, 'end': max_time,
                'duration': max_time - min_time,
                'taxonomic_group': taxon
            })
    time_df = pd.DataFrame(time_ranges)
    time_df_top = time_df[time_df['taxonomic_group'].isin(top_taxa)].copy()
    taxon_order = {group: i for i, group in enumerate(top_taxa)}
    time_df_top['taxon_order'] = time_df_top['taxonomic_group'].map(taxon_order)
    time_df_top = time_df_top.sort_values(['taxon_order', 'duration']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in time_df_top.iterrows():
        color = taxonomic_colors.get(row['taxonomic_group'], 'lightgray')
        ax.plot([row['start'], row['end']], [i, i], color=color, linewidth=1.5, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlim(1900,2026)
    ax.set_xlabel('Year')
    ax.set_ylabel('Species Index')
    plt.tight_layout()
    plt.savefig('img/time_series_ranges.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_species_locations(top_taxa, taxonomic_colors):
    processed_df = pd.read_csv('csv/bifurcation_results_short.csv')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
    bio_locs = load_biotime_locations(biotime_path)
    riv_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
    riv_locs = load_rivfishtime_locations(riv_path)
    gpdd_data_path = os.path.join(data_dir, 'gpdd', 'gpdd_data.csv')
    gpdd_main_path = os.path.join(data_dir, 'gpdd', 'gpdd_main.csv')
    gpdd_taxon_path = os.path.join(data_dir, 'gpdd', 'gpdd_taxon.csv')
    gpdd_location_path = os.path.join(data_dir, 'gpdd', 'gpdd_location.csv')
    gpdd_locs = load_gpdd_locations(gpdd_data_path, gpdd_main_path, gpdd_taxon_path, gpdd_location_path)
    all_locs = pd.concat([bio_locs, riv_locs, gpdd_locs], ignore_index=True)
    all_locs['valid_name'] = all_locs['valid_name'].str.strip()
    processed_df['species'] = processed_df['species'].str.strip()
    merged_data = all_locs.merge(processed_df[['species', 'taxonomic_group']], left_on='valid_name', right_on='species', how='inner')
    merged_data = merged_data[merged_data['taxonomic_group'].isin(top_taxa)]
    if merged_data.empty:
        print("No location data for top taxa.")
        return
    location_groups = merged_data.groupby(['LATITUDE', 'LONGITUDE'])['taxonomic_group'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else 'Other'
    ).reset_index()
    plt.figure(figsize=(18, 10))
    m = Basemap(projection='robin', llcrnrlat=-60, urcrnrlat=90, lon_0=0, resolution='c')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='#dff2f9')
    m.fillcontinents(color='#ffeecc', lake_color='#dff2f9')
    parallels = np.arange(-60, 91, 30)
    m.drawparallels(parallels, labels=[0,0,1,0], fontsize=14, linewidth=0.5, color='gray')
    meridians = np.arange(-180, 181, 60)
    m.drawmeridians(meridians, labels=[1,0,0,0], fontsize=14, linewidth=0.5, color='gray')
    x, y = m(location_groups['LONGITUDE'].values, location_groups['LATITUDE'].values)
    point_colors = [taxonomic_colors.get(group, 'lightgray') for group in location_groups['taxonomic_group']]
    plt.scatter(x, y, s=40, c=point_colors, alpha=1.0, edgecolors='none', rasterized=True)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=taxonomic_colors[group], markersize=10, label=group) for group in top_taxa]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/species_locations_map.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created species location map with {len(location_groups)} unique locations")

def main():
    os.makedirs('img', exist_ok=True)
    probability_threshold=0.7
    top_taxa, taxonomic_colors = get_top_taxonomic_groups(probability_threshold)
    print(f"Top 20 taxonomic groups: {top_taxa}")
    print("Generating time series ranges figure...")
    plot_time_series_ranges(top_taxa, taxonomic_colors)
    print("Generating species locations map...")
    plot_species_locations(top_taxa, taxonomic_colors)
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
