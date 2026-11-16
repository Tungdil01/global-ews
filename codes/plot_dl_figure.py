#!/usr/bin/env python
# coding: utf-8
import os
import gc
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FixedLocator
import pandas as pd
import ewstools
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import Parallel, delayed
from main_dl import create_summary_figure, TAXONOMIC_ORDER, find_transition, robust_detrending, brook_zero_handling, deseason_residuals, MIN_CP_FRAC, process_species
from main_orchestrator import setup_data, classifier_paths_500, classifier_paths_1500, min_records, late_transition_window, CP_METHODS, DL_FILE, bifurcation_map, probability_threshold
SHORT_FILE = 'csv/bifurcation_results_SHORT.csv'
TARGET_SPECIES = ['Ctenotus ariadnae', 'Microtus californicus', 'Coereba flaveola', 'Cygnus olor'] # , 'Ixoreus naevius', 'Alca torda', 'Sphyrapicus ruber', 'Anthus trivialis', 
X_AXIS_MAX_YEAR ={}

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def report_discarded_taxonomic_groups(df):
    present_groups = set(df['taxonomic_group'].dropna().unique())
    discarded = sorted(present_groups - set(TAXONOMIC_ORDER))
    if discarded:
        total_discarded = int(df['taxonomic_group'].isin(discarded).sum())
        print(f"WARNING: {len(discarded)} taxonomic_group(s) not in TAXONOMIC_ORDER, {total_discarded} rows dropped from summary table:")
        for group in discarded:
            n = int((df['taxonomic_group'] == group).sum())
            print(f"  {group}: {n} rows")

def generate_figure_from_dl_without_prob_filter(dl_file_path):
    if not os.path.exists(dl_file_path) or os.path.getsize(dl_file_path) == 0:
        print("DL file missing or empty.")
        return
    df = pd.read_csv(dl_file_path)
    if 'taxonomic_group' in df.columns:
        df = df[df['taxonomic_group'] != 'Other']
    report_discarded_taxonomic_groups(df)
    print(f"Generating figure from {len(df)} species (non-GBIF, with transition_year, no probability filter).")

def print_target_species_diagnostics(target_species, probability_threshold):
    print("\n=== TARGET SPECIES DIAGNOSTICS ===")
    df = pd.read_csv(DL_FILE)
    df['species'] = df['species'].astype(str).str.strip()
    n_unique = df['species'].nunique()
    print(f"DL_FILE unique species: {n_unique}")
    print(f"DL_FILE total rows (segments): {len(df)}")
    if 'tipping_point_type' in df.columns:
        tc = df['tipping_point_type'].value_counts()
        for t, c in tc.items():
            print(f"  {t}: {c}")
    for sp in target_species:
        sp_key = str(sp).strip()
        sub = df[df['species'] == sp_key]
        if len(sub) == 0:
            print(f"  {sp_key}: NOT FOUND in {DL_FILE}")
            continue
        sub = sub.sort_values('best_probability', ascending=False)
        row = sub.iloc[0]
        ds = str(row.get('dataset', 'N/A'))
        ty = row.get('transition_year', 'N/A')
        ty_str = f"{int(float(ty))}" if pd.notna(ty) else "N/A"
        tt = str(row.get('tipping_point_type', 'N/A'))
        bp = float(row.get('best_probability', 0.0))
        prob_ok = "OK" if bp >= probability_threshold else "LOW"
        print(f"  {sp_key}: dataset={ds} year={ty_str} type={tt} prob={bp:.3g} prob_status={prob_ok}")

def get_already_plotted_species(output_dir):
    plotted = set()
    if not os.path.exists(output_dir):
        return plotted
    for fname in os.listdir(output_dir):
        if fname.endswith('_dl_predictions.pdf'):
            species_name = fname[:-len('_dl_predictions.pdf')]
            plotted.add(species_name)
    return plotted

def filter_unplotted_species(target_species, output_dir):
    plotted = get_already_plotted_species(output_dir)
    remaining = [sp for sp in target_species if sp not in plotted]
    print(f"Already plotted: {len(plotted)}. Remaining to plot: {len(remaining)}.")
    stale = sorted(plotted - set(target_species))
    if stale:
        print(f"WARNING: {len(stale)} stale PDFs in {output_dir} not in TARGET_SPECIES: {stale}")
    return remaining

def get_species_to_process(target_species, output_dir, overwrite):
    if overwrite:
        return target_species
    return filter_unplotted_species(target_species, output_dir)

def load_SHORT_file_lookup(SHORT_file_path=DL_FILE):
    if not os.path.exists(SHORT_file_path) or os.path.getsize(SHORT_file_path) == 0:
        return {}
    df = pd.read_csv(SHORT_file_path)
    required = ['species', 'dataset']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"DL_FILE missing required columns {missing}; cannot build lookup.")
        return {}
    if 'best_probability' in df.columns:
        df = df.sort_values('best_probability', ascending=False)
    if 'timeseries_id' in df.columns:
        df['timeseries_id'] = df['timeseries_id'].apply(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
    lookup = {}
    for _, row in df.iterrows():
        sp = str(row['species']).strip()
        if sp in lookup:
            continue
        ds = str(row['dataset']).strip()
        ts = str(row['timeseries_id']).strip() if 'timeseries_id' in df.columns else ''
        ty = float(row['transition_year']) if 'transition_year' in df.columns and pd.notna(row['transition_year']) else None
        lookup[sp] = (ds, ts, ty)
    return lookup

def _resolve_dataset_key(datasets_dict, dataset_name):
    dataset_name = str(dataset_name).strip()
    if dataset_name in datasets_dict:
        return dataset_name
    for k, df in datasets_dict.items():
        if len(df) == 0:
            continue
        if 'dataset' in df.columns:
            vals = df['dataset'].astype(str).str.strip().unique()
            if len(vals) > 0 and vals[0] == dataset_name:
                return k
        if str(k).strip().lower() == dataset_name.lower():
            return k
    return None

def select_species_unit(datasets_dict, species_name, dataset, timeseries_id):
    key = _resolve_dataset_key(datasets_dict, dataset)
    if key is None:
        return pd.DataFrame()
    df = datasets_dict[key]
    if len(df) == 0:
        return pd.DataFrame()
    mask = df['SPECIES_ID'].astype(str).str.strip() == str(species_name).strip()
    if timeseries_id:
        mask = mask & (df['TIMESERIES_ID'].astype(str).str.strip() == str(timeseries_id).strip())
    return df[mask].copy()

def save_dl_predictions_pdf(ts, species_name, eval_index, output_dir='img'):
    plt.rcParams.update({'font.size': 40, 'axes.labelsize': 40, 'xtick.labelsize': 40, 'ytick.labelsize': 40, 'legend.fontsize': 40, 'axes.linewidth': 2.0, 'lines.linewidth': 5.0, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black'})
    fig, axes = plt.subplots(2, 1, figsize=(1200/72, 1000/72), sharex=True)
    years = np.asarray(ts.state['year'].values)
    eval_year = years[eval_index]
    axes[0].scatter(years, ts.state['state'], color='green', s=200, label=None, zorder=2)
    if 'smoothing' in ts.state.columns:
        axes[0].plot(years, ts.state['smoothing'], color='red', linewidth=5.0, label=None, zorder=3)
    axes[0].axvline(eval_year, color='grey', linestyle='--', linewidth=1.0)
    axes[0].set_ylabel('Abundance')
    axes[0].grid(False)
    for spine in axes[0].spines.values():
        spine.set_linewidth(2.0)
    if len(ts.dl_preds) > 0:
        df = ts.dl_preds.drop(['classifier'], axis=1)
        df_plot = df.groupby(['time']).mean().reset_index()
        df_plot['time'] = years[df_plot['time'].astype(int).values]
        dl_class_map = {0: 'fold bif.', 1: 'Hopf bif.', 2: 'transc. bif.', 3: 'no-bif.'}
        dl_color_map = {0: '#1f77b4', 1: '#9467bd', 2: '#ff7f0e', 3: '#999999'}
        for cls in [0, 1, 2, 3]:
            if cls in df_plot.columns:
                axes[1].plot(df_plot['time'].values, df_plot[cls].values, color=dl_color_map[cls], linewidth=5.0, label=dl_class_map[cls])
        axes[1].axvline(eval_year, color='grey', linestyle='--', linewidth=1.0)
        axes[1].legend(loc='upper right', frameon=True, handlelength=1.0)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Probability')
    if species_name in X_AXIS_MAX_YEAR:
        axes[1].set_xlim(right=X_AXIS_MAX_YEAR[species_name])
    locator = MaxNLocator(nbins=6, integer=True)
    ticks = locator.tick_values(axes[1].get_xlim()[0], axes[1].get_xlim()[1])
    axes[1].xaxis.set_major_locator(FixedLocator(ticks))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[1].grid(False)
    for spine in axes[1].spines.values():
        spine.set_linewidth(2.0)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{species_name}_dl_predictions.pdf"), dpi=72, bbox_inches='tight')
    plt.close(fig)
    print(f"PDF plot saved for {species_name}")

def plot_dl_predictions(species_name, species_data, data_taxonomy, genus_taxonomy):
    result = process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, late_transition_window, species_to_plot=None, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy, output_file=None, cp_method=None)
    if result is None or result.get('error') is not None:
        print(f"No valid result for {species_name}, skipping")
        return
    ts = result.get('ts')
    transition_year = result.get('transition_year')
    if ts is None or transition_year is None:
        print(f"Missing ts or transition_year for {species_name}, skipping")
        return
    years = np.asarray(ts.state['year'].values)
    transition_idx = int(np.where(years == transition_year)[0][0])
    eval_index = int(0.9 * transition_idx)
    if eval_index < 0:
        eval_index = 0
    save_dl_predictions_pdf(ts, species_name, eval_index)

def process_species_pdf(species_name, species_data, data_taxonomy, genus_taxonomy):
    set_seed(1234)
    plot_dl_predictions(species_name, species_data, data_taxonomy, genus_taxonomy)

def _build_species_task_list(target_species, lookup):
    tasks = []
    for species_name in target_species:
        sp_key = str(species_name).strip()
        if sp_key not in lookup:
            print(f"No entry in {DL_FILE} for {sp_key}; skipping.")
            continue
        ds, ts, ty = lookup[sp_key]
        if ty is None:
            print(f"No transition_year for {sp_key} on ({ds}, {ts}); skipping")
            continue
        if not ts or ts.lower() == 'nan':
            print(f"No timeseries_id for {sp_key} on ({ds}, {ts}); skipping")
            continue
        tasks.append((sp_key, ds, ts))
    return tasks

def generate_dl_predictions_pdfs(target_species, datasets_dict, data_taxonomy, genus_taxonomy, batch_size=50):
    lookup = load_SHORT_file_lookup(DL_FILE)
    tasks = _build_species_task_list(target_species, lookup)
    total_batches = (len(tasks) + batch_size - 1) // batch_size if tasks else 0
    print(f"Total species with valid entries: {len(tasks)}. Batches: {total_batches} (batch_size={batch_size}).")
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        batch_num = start // batch_size + 1
        print(f"Batch {batch_num}/{total_batches}: preparing {len(batch)} species...")
        batch_data = []
        for sp_key, ds, ts in batch:
            species_data = select_species_unit(datasets_dict, sp_key, ds, ts)
            if len(species_data) == 0:
                print(f"No data found for {sp_key} on ({ds}, {ts}); skipping")
                continue
            batch_data.append((sp_key, species_data))
        if not batch_data:
            print(f"Batch {batch_num}/{total_batches}: no valid data, skipping.")
            continue
        Parallel(n_jobs=-1)(delayed(process_species_pdf)(sp, sd, data_taxonomy, genus_taxonomy) for sp, sd in batch_data)
        print(f"Batch {batch_num}/{total_batches}: done ({len(batch_data)} species).")

def main():
    output_dir = 'img'
    seed = 1234
    overwrite = False
    batch_size = 50
    set_seed(seed)
    generate_figure_from_dl_without_prob_filter(SHORT_FILE)
    print_target_species_diagnostics(TARGET_SPECIES, probability_threshold)
    datasets_dict, _, data_taxonomy, genus_taxonomy = setup_data(min_records=min_records)
    remaining = get_species_to_process(TARGET_SPECIES, output_dir, overwrite)
    generate_dl_predictions_pdfs(remaining, datasets_dict, data_taxonomy, genus_taxonomy, batch_size=batch_size)

if __name__ == "__main__":
    main()
