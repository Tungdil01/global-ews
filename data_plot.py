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

def apply_tipping_point_threshold(df, threshold=0.3):
    if 'raw_tipping_point_type' in df.columns and 'best_probability' in df.columns:
        df['tipping_point_type'] = df.apply(
            lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no-bifurcation',
            axis=1
        )
    return df

def get_top_taxonomic_groups(threshold):
    freq_df = pd.read_csv('csv/bifurcation_results_short.csv')
    freq_df = apply_tipping_point_threshold(freq_df, threshold)
    # Count species per taxonomic group and get top 20
    taxon_counts = freq_df['taxonomic_group'].value_counts()
    if 'Other' in taxon_counts.index:
        taxon_counts = taxon_counts.drop('Other')
    top_taxa = taxon_counts.head(20).index.tolist()
    # Create a discrete colormap with 20 distinct colors
    distinct_colors = [
        '#4363d8', '#e6194b', '#3cb44b', '#ffe119', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#000075', '#ffd8b1', '#808080'
    ]
    # Map colors to taxonomic groups
    taxonomic_colors = {group: distinct_colors[i] for i, group in enumerate(top_taxa)}
    return top_taxa, taxonomic_colors

def load_biotime_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
        except:
            df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    required = ["YEAR", "ABUNDANCE", "VALID_NAME", "TAXON"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=required)
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
    return df

def plot_time_series_ranges(top_taxa, taxonomic_colors):
    freq_df = pd.read_csv('csv/bifurcation_results_short.csv')
    raw_data = load_biotime_data('data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv')
    # Create a mapping from species to taxonomic group from freq.csv
    species_to_taxon = dict(zip(freq_df['species'], freq_df['taxonomic_group']))
    # Get time ranges and taxonomic groups for each species
    time_ranges = []
    for species in freq_df['species'].unique():
        species_data = raw_data[raw_data['SPECIES_ID'] == species]
        if not species_data.empty:
            min_time = species_data['TIME'].min()
            max_time = species_data['TIME'].max()
            taxon = species_to_taxon.get(species, 'Other')
            time_ranges.append({
                'species': species, 'start': min_time, 'end': max_time,
                'duration': max_time - min_time,
                'taxonomic_group': taxon
            })
    time_df = pd.DataFrame(time_ranges)
    # Filter to only include top taxonomic groups
    time_df_top = time_df[time_df['taxonomic_group'].isin(top_taxa)].copy()
    # Create order mapping for top taxa
    taxon_order = {group: i for i, group in enumerate(top_taxa)}
    time_df_top['taxon_order'] = time_df_top['taxonomic_group'].map(taxon_order)
    # Sort by taxonomic group and then by duration (smallest first)
    time_df_top = time_df_top.sort_values(['taxon_order', 'duration']).reset_index(drop=True)
    # Create plot with taxonomic group colors
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in time_df_top.iterrows():
        color = taxonomic_colors.get(row['taxonomic_group'], 'lightgray')
        ax.plot([row['start'], row['end']], [i, i], color=color, linewidth=1.5, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Year')
    ax.set_ylabel('Species Index')
#    # Create legend for top taxonomic groups
#    legend_elements = []
#    for group in top_taxa:
#        legend_elements.append(plt.Line2D([0], [0], color=taxonomic_colors[group], label=group, linewidth=3))
#    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('img/time_series_ranges.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def load_biotime_locations(filepath):
    try:
        df = pd.read_csv(filepath, usecols=['valid_name', 'LATITUDE', 'LONGITUDE'], encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, usecols=['valid_name', 'LATITUDE', 'LONGITUDE'], encoding='ISO-8859-1')
        except:
            df = pd.read_csv(filepath, usecols=['valid_name', 'LATITUDE', 'LONGITUDE'])
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    return df

def plot_species_locations(top_taxa, taxonomic_colors):
    processed_df = pd.read_csv('csv/bifurcation_results_short.csv')
    try:
        raw_data = load_biotime_locations('data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv')
        merged_data = raw_data.merge(processed_df[['species', 'taxonomic_group']], left_on='valid_name', right_on='species', how='inner')
        # Filter to only include top taxonomic groups
        merged_data = merged_data[merged_data['taxonomic_group'].isin(top_taxa)]
        # Get unique locations with their taxonomic groups
        location_groups = merged_data.groupby(['LATITUDE', 'LONGITUDE'])['taxonomic_group'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'Other'
        ).reset_index()
        plt.figure(figsize=(18, 10))
        m = Basemap(projection='robin', llcrnrlat=-60, urcrnrlat=90, lon_0=0, resolution='c')
        m.drawcoastlines()
        m.drawmapboundary(fill_color='#ffffff')
        m.fillcontinents(color='#ffeecc', lake_color='#ffffff')
        x, y = m(location_groups['LONGITUDE'].values, location_groups['LATITUDE'].values)
        point_colors = [taxonomic_colors.get(group, 'lightgray') for group in location_groups['taxonomic_group']]
        plt.scatter(x, y, s=40, c=point_colors, alpha=1.0, edgecolors='none', rasterized=True)
        # Create legend for top taxonomic groups
        legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=taxonomic_colors[group], markersize=10, label=group) for group in top_taxa]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        plt.tight_layout()
        plt.savefig('img/species_locations_map.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created species location map with {len(location_groups)} unique locations")
    except Exception as e:
        print(f"Error creating species location map: {str(e)}")
        traceback.print_exc()

def main():
    os.makedirs('img', exist_ok=True)
    threshold=0.5
    # Get top taxonomic groups and their colors
    top_taxa, taxonomic_colors = get_top_taxonomic_groups(threshold)
    print(f"Top 20 taxonomic groups: {top_taxa}")
    print("Generating time series ranges figure...")
    plot_time_series_ranges(top_taxa, taxonomic_colors)
    print("Generating species locations map...")
    plot_species_locations(top_taxa, taxonomic_colors)
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
