#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from main_dl import load_combined_data, EXCLUDED_GROUPS, EXCLUDED_SPECIES_LABELS
from main_orchestrator import min_records, probability_threshold, late_transition_window, check_annual_series
from build_short import select_best_dataset_per_species
plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14, 'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 1.5})

def apply_tipping_point_probability_threshold(df, probability_threshold=probability_threshold):
    if 'raw_tipping_point_type' in df.columns and 'best_probability' in df.columns:
        df['tipping_point_type'] = df.apply(lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= probability_threshold else 'no-bifurcation', axis=1)
    return df

def get_top_taxonomic_groups(probability_threshold):
    freq_df = pd.read_csv('csv/bifurcation_results_large.csv')
    freq_df = apply_tipping_point_probability_threshold(freq_df, probability_threshold)
    taxon_counts = freq_df['taxonomic_group'].value_counts()
    if 'Other' in taxon_counts.index:
        taxon_counts = taxon_counts.drop('Other')
    top_taxa = taxon_counts.head(20).index.tolist()
    distinct_colors = ['#4363d8', '#e6194b', '#ffe119', '#911eb4', '#3cb44b', '#f58231', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#000075', '#ffd8b1', '#808080']
    specified_colors = {'Hexapoda': '#e6194b', 'Tracheophyta': '#3cb44b', 'Bryophyta': '#808000', 'Actinopterygii': '#4363d8', 'Chlorophyta': '#bcf60c'}
    available = distinct_colors[:]
    for group in top_taxa:
        if group in specified_colors:
            color = specified_colors[group]
            if color in available:
                available.remove(color)
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

def get_analyzed_species_map():
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_df['dataset'] = large_df['dataset'].astype(str).str.strip()
    large_map = dict(zip(large_df['species'], large_df['dataset']))
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    failed_df['dataset'] = failed_df['dataset'].astype(str).str.strip()
    if 'reason' in failed_df.columns:
        failed_df = failed_df[failed_df['reason'].astype(str).str.strip() != 'taxonomic_excluded']
    species_to_taxon = get_species_to_taxon()
    failed_df['taxonomic_group'] = failed_df['species'].map(species_to_taxon).fillna('Other')
    failed_df = failed_df[failed_df['taxonomic_group'] != 'Other']
    failed_df = failed_df[~failed_df['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    failed_df = failed_df[~failed_df['species'].isin(EXCLUDED_SPECIES_LABELS)]
    failed_df = failed_df.drop_duplicates(subset=['species'], keep='first')
    failed_map = dict(zip(failed_df['species'], failed_df['dataset']))
    return {**failed_map, **large_map}

def get_species_to_taxon():
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_taxon = dict(zip(large_df['species'], large_df['taxonomic_group']))
    taxon_cache = pd.read_csv('csv/taxonomic_cache.csv')
    taxon_cache['species'] = taxon_cache['species'].astype(str).str.strip()
    cache_taxon = dict(zip(taxon_cache['species'], taxon_cache['taxonomic_group']))
    return {**cache_taxon, **large_taxon}

def get_all_species_df():
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_df['dataset'] = large_df['dataset'].astype(str).str.strip()
    large_species_set = set(large_df['species'])
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    failed_df['dataset'] = failed_df['dataset'].astype(str).str.strip()
    if 'reason' in failed_df.columns:
        failed_df = failed_df[failed_df['reason'].astype(str).str.strip() != 'taxonomic_excluded']
    failed_only = failed_df[~failed_df['species'].isin(large_species_set)]
    return pd.concat([large_df[['species', 'dataset']], failed_only[['species', 'dataset']]], ignore_index=True).drop_duplicates()

def load_datasets_dict_for_plots():
    datasets_dict, _, _, _ = load_combined_data(min_records=1, prefilter=False, min_time=1900)
    for ds_name in list(datasets_dict.keys()):
        df = datasets_dict[ds_name]
        if len(df) == 0:
            continue
        df = df.copy()
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
        df = df[df['TIME'] >= 1900].copy()
        if 'DATASET' in df.columns and 'dataset' not in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
        if 'dataset' not in df.columns:
            df['dataset'] = ds_name
        datasets_dict[ds_name] = df
    return datasets_dict

def load_combined_data_for_plots(datasets_dict):
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['SPECIES_ID', 'dataset', 'TIME', 'ABUNDANCE', 'LATITUDE', 'LONGITUDE'])
    return pd.concat(frames, ignore_index=True)

def get_best_timeseries_selection(datasets_dict, species_to_taxon):
    large_df = pd.read_csv('csv/bifurcation_results_large.csv')
    large_df['species'] = large_df['species'].astype(str).str.strip()
    large_df['dataset'] = large_df['dataset'].astype(str).str.strip()
    large_df['timeseries_id'] = large_df['timeseries_id'].astype(str).str.strip()
    large_sel = large_df[['species', 'dataset', 'timeseries_id', 'best_probability']].copy()
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    failed_df['dataset'] = failed_df['dataset'].astype(str).str.strip()
    if 'reason' in failed_df.columns:
        failed_df = failed_df[failed_df['reason'].astype(str).str.strip() != 'taxonomic_excluded']
    failed_df['taxonomic_group'] = failed_df['species'].map(species_to_taxon)
    failed_df = failed_df[failed_df['taxonomic_group'].notna()]
    failed_df = failed_df[failed_df['taxonomic_group'] != 'Other']
    failed_df = failed_df[~failed_df['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    failed_df = failed_df[~failed_df['species'].isin(EXCLUDED_SPECIES_LABELS)]
    large_species_set = set(large_df['species'])
    failed_only = failed_df[~failed_df['species'].isin(large_species_set)].copy()
    failed_rows = []
    for _, row in failed_only.iterrows():
        sp = row['species']
        ds = row['dataset']
        df_ds = datasets_dict[ds]
        df_ds_local = df_ds.copy()
        df_ds_local['SPECIES_ID'] = df_ds_local['SPECIES_ID'].astype(str).str.strip()
        df_ds_local['TIMESERIES_ID'] = df_ds_local['TIMESERIES_ID'].astype(str).str.strip()
        sp_ts = df_ds_local[df_ds_local['SPECIES_ID'] == sp]['TIMESERIES_ID'].unique()
        for ts_id in sp_ts:
            failed_rows.append({'species': sp, 'dataset': ds, 'timeseries_id': str(ts_id), 'best_probability': 0.0})
    if failed_rows:
        failed_only_df = pd.DataFrame(failed_rows)
        merged = pd.concat([large_sel, failed_only_df], ignore_index=True)
    else:
        merged = large_sel
    best = select_best_dataset_per_species(merged, datasets_dict, mode='best', threshold=probability_threshold)
    return best[['species', 'dataset', 'timeseries_id', 'best_probability']].copy()

def filter_selection_by_annual_check(best_sel, all_data):
    all_data = all_data.copy()
    all_data['SPECIES_ID'] = all_data['SPECIES_ID'].astype(str).str.strip()
    all_data['dataset'] = all_data['dataset'].astype(str).str.strip()
    all_data['TIMESERIES_ID'] = all_data['TIMESERIES_ID'].astype(str).str.strip()
    sel = best_sel.copy()
    sel['species'] = sel['species'].astype(str).str.strip()
    sel['dataset'] = sel['dataset'].astype(str).str.strip()
    sel['timeseries_id'] = sel['timeseries_id'].astype(str).str.strip()
    all_data['_key'] = all_data['SPECIES_ID'] + '|' + all_data['dataset'] + '|' + all_data['TIMESERIES_ID']
    sel_keys = (sel['species'] + '|' + sel['dataset'] + '|' + sel['timeseries_id']).unique().tolist()
    filtered = all_data[all_data['_key'].isin(sel_keys)].copy()
    filtered['YEAR_INT'] = np.floor(filtered['TIME']).astype(int)
    annual = filtered.groupby(['SPECIES_ID', 'dataset', 'YEAR_INT'])['ABUNDANCE'].mean().reset_index()
    passing = set()
    for (species, dataset), group in annual.groupby(['SPECIES_ID', 'dataset']):
        passed, reason = check_annual_series(group.rename(columns={'YEAR_INT': 'YEAR'}))
        if passed:
            passing.add((species, dataset))
    keep_sel_mask = np.array([(s, d) in passing for s, d in zip(sel['species'].values, sel['dataset'].values)])
    return sel[keep_sel_mask].copy()

def load_combined_locations_for_plots(full_data):
    loc_df = full_data[['SPECIES_ID', 'dataset', 'LATITUDE', 'LONGITUDE']].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
    loc_df = loc_df.rename(columns={'SPECIES_ID': 'valid_name'})
    loc_df['valid_name'] = loc_df['valid_name'].str.strip()
    loc_df = loc_df.drop_duplicates()
    return loc_df

def pick_point(points_df, reference_df):
    counts = points_df.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='count')
    max_count = counts['count'].max()
    top = counts[counts['count'] == max_count]
    if len(top) == 1:
        return top.iloc[0]['LATITUDE'], top.iloc[0]['LONGITUDE']
    centroid_lat = reference_df['LATITUDE'].mean()
    centroid_lon = reference_df['LONGITUDE'].mean()
    top = top.copy()
    top['dist'] = np.sqrt((top['LATITUDE'] - centroid_lat)**2 + (top['LONGITUDE'] - centroid_lon)**2)
    best = top.loc[top['dist'].idxmin()]
    return best['LATITUDE'], best['LONGITUDE']

def pick_single_point_per_species(df):
    chosen_per_dataset = []
    for (species, dataset), group in df.groupby(['species', 'dataset']):
        lat, lon = pick_point(group[['LATITUDE', 'LONGITUDE']], group[['LATITUDE', 'LONGITUDE']])
        chosen_per_dataset.append({'species': species, 'dataset': dataset, 'LATITUDE': lat, 'LONGITUDE': lon})
    chosen_df = pd.DataFrame(chosen_per_dataset)
    final_points = []
    for species, group in chosen_df.groupby('species'):
        original_group = df[df['species'] == species]
        lat, lon = pick_point(group[['LATITUDE', 'LONGITUDE']], original_group[['LATITUDE', 'LONGITUDE']])
        final_points.append({'species': species, 'LATITUDE': lat, 'LONGITUDE': lon})
    return pd.DataFrame(final_points)

def build_or_load_locations(species_to_taxon):
    cache_path = 'csv/locations.csv'
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    datasets_dict = load_datasets_dict_for_plots()
    full_data = load_combined_data_for_plots(datasets_dict)
    loc_df = load_combined_locations_for_plots(full_data)
    all_species_df = get_all_species_df()
    target_species = set(all_species_df['species'].unique())
    filtered_loc = loc_df.rename(columns={'valid_name': 'species'}).copy()
    filtered_loc['species'] = filtered_loc['species'].astype(str).str.strip()
    filtered_loc['dataset'] = filtered_loc['dataset'].astype(str).str.strip()
    filtered_loc = filtered_loc[filtered_loc['species'].isin(target_species)]
    points_df = pick_single_point_per_species(filtered_loc[['species', 'dataset', 'LATITUDE', 'LONGITUDE']])
    points_df['taxonomic_group'] = points_df['species'].map(species_to_taxon)
    if points_df['taxonomic_group'].isna().any():
        missing = points_df.loc[points_df['taxonomic_group'].isna(), 'species'].tolist()
        raise ValueError(f"Missing taxonomic_group for species: {missing}")
    short_df = pd.read_csv('csv/bifurcation_results_short.csv')
    short_species = set(short_df['species'].astype(str).str.strip())
    points_df['status'] = np.where(points_df['species'].isin(short_species), 'transitions', 'low_confidence')
    points_df.to_csv(cache_path, index=False)
    return points_df

def build_or_load_time_series(species_to_taxon):
    cache_path = 'csv/time_series.csv'
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    datasets_dict = load_datasets_dict_for_plots()
    full_data = load_combined_data_for_plots(datasets_dict)
    all_species_df = get_all_species_df()
    target_species = set(all_species_df['species'].unique())
    full_data = full_data.copy()
    full_data['SPECIES_ID'] = full_data['SPECIES_ID'].astype(str).str.strip()
    full_data['dataset'] = full_data['dataset'].astype(str).str.strip()
    filtered = full_data[full_data['SPECIES_ID'].isin(target_species)].copy()
    filtered['YEAR_INT'] = np.floor(filtered['TIME']).astype(int)
    annual = filtered.groupby(['SPECIES_ID', 'dataset', 'YEAR_INT'])['ABUNDANCE'].mean().reset_index()
    time_ranges = []
    for (species, dataset), group in annual.groupby(['SPECIES_ID', 'dataset']):
        min_time = int(group['YEAR_INT'].min())
        max_time = int(group['YEAR_INT'].max())
        taxon = species_to_taxon.get(species, 'Other')
        time_ranges.append({'species': species, 'dataset': dataset, 'start': min_time, 'end': max_time, 'duration': max_time - min_time, 'taxonomic_group': taxon})
    time_df = pd.DataFrame(time_ranges)
    time_df.to_csv(cache_path, index=False)
    return time_df

def plot_time_series_ranges(top_taxa, taxonomic_colors, time_df):
    long_df = time_df[time_df['duration'] > 95].sort_values('duration', ascending=False)
    print("\n=== SPECIES WITH > 95 YEARS ===")
    print(f"{'Species':<40} {'Start':>6} {'End':>6} {'Duration':>9} {'Taxon'}")
    print("-" * 100)
    for _, row in long_df.iterrows():
        print(f"{row['species']:<40} {int(row['start']):>6} {int(row['end']):>6} {int(row['duration']):>9} {row['taxonomic_group']}")
    print("-" * 100)
    print(f"COUNT > 95 years: {len(long_df)}")
    print(f"GLOBAL MIN START: {int(time_df['start'].min())}")
    print(f"GLOBAL MAX END:   {int(time_df['end'].max())}")
    print(f"MIN DURATION:     {int(time_df['duration'].min())}")
    print(f"MAX DURATION:     {int(time_df['duration'].max())}")
    print("=== END ===\n")
    time_df_top = time_df[time_df['taxonomic_group'].isin(top_taxa)].copy()
    taxon_order = {group: i for i, group in enumerate(top_taxa)}
    time_df_top['taxon_order'] = time_df_top['taxonomic_group'].map(taxon_order)
    time_df_top = time_df_top.sort_values(['taxon_order', 'duration']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in time_df_top.iterrows():
        color = taxonomic_colors.get(row['taxonomic_group'], 'lightgray')
        ax.plot([row['start'], row['end']], [i, i], color=color, linewidth=1.5, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Year')
    ax.set_ylabel('Individual Time Series')
    legend_elements = [plt.Line2D([0], [0], color=taxonomic_colors[group], linewidth=1.5, label=group) for group in top_taxa]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/time_series_ranges.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_species_locations(top_taxa, taxonomic_colors, points_df):
    if points_df.empty:
        print("No location data for any species.")
        return
    plt.figure(figsize=(18, 10))
    m = Basemap(projection='robin', llcrnrlat=-60, urcrnrlat=90, lon_0=0, resolution='c')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='#dff2f9')
    m.fillcontinents(color='#ffeecc', lake_color='#dff2f9')
    parallels = np.arange(-60, 91, 30)
    m.drawparallels(parallels, labels=[0,0,1,0], fontsize=14, linewidth=0.5, color='gray')
    meridians = np.arange(-180, 181, 60)
    m.drawmeridians(meridians, labels=[1,0,0,0], fontsize=14, linewidth=0.5, color='gray')
    x, y = m(points_df['LONGITUDE'].values, points_df['LATITUDE'].values)
    point_colors = [taxonomic_colors.get(group, 'lightgray') for group in points_df['taxonomic_group']]
    plt.scatter(x, y, s=40, c=point_colors, alpha=1.0, edgecolors='none', rasterized=True)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=taxonomic_colors[group], markersize=10, label=group) for group in top_taxa]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/species_locations_map.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created species location map with {len(points_df)} species")

def plot_species_locations_three_colors(probability_threshold, points_df):
    points_df = points_df[points_df['status'].notna()].copy()
    status_colors = {'transitions': 'green', 'low_confidence': 'red'} # '#b8860b
    status_labels = {'transitions': 'transitions', 'low_confidence': 'undetermined'}
    plt.figure(figsize=(18, 10))
    m = Basemap(projection='robin', llcrnrlat=-60, urcrnrlat=90, lon_0=0, resolution='c')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='#dff2f9')
    m.fillcontinents(color='#ffeecc', lake_color='#dff2f9')
    parallels = np.arange(-60, 91, 30)
    m.drawparallels(parallels, labels=[0,0,1,0], fontsize=14, linewidth=0.5, color='gray')
    meridians = np.arange(-180, 181, 60)
    m.drawmeridians(meridians, labels=[1,0,0,0], fontsize=14, linewidth=0.5, color='gray')
    x, y = m(points_df['LONGITUDE'].values, points_df['LATITUDE'].values)
    point_colors = [status_colors[status] for status in points_df['status']]
    plt.scatter(x, y, s=40, c=point_colors, alpha=1.0, edgecolors='none', rasterized=True)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=status_colors[status], markersize=10, label=status_labels[status]) for status in ['transitions', 'low_confidence']]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig('img/species_locations_three_colors_map.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created species location map with {len(points_df)} species")

def generate_all_plots(probability_threshold):
    top_taxa, taxonomic_colors = get_top_taxonomic_groups(probability_threshold)
    print(f"Top 20 taxonomic groups: {top_taxa}")
    species_to_taxon = get_species_to_taxon()
    time_df = build_or_load_time_series(species_to_taxon)
    points_df = build_or_load_locations(species_to_taxon)
    print("Generating time series ranges figure...")
    plot_time_series_ranges(top_taxa, taxonomic_colors, time_df)
    print("Generating species locations map...")
    plot_species_locations(top_taxa, taxonomic_colors, points_df)
    print("Generating three-color species locations map...")
    plot_species_locations_three_colors(probability_threshold, points_df)
    print("All figures generated successfully!")

def main():
    os.makedirs('img', exist_ok=True)
    generate_all_plots(probability_threshold)

if __name__ == "__main__":
    main()
