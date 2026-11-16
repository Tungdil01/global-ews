#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from main_dl import load_combined_data
from main_orchestrator import min_records, probability_threshold
from functools import lru_cache
tipping_point_colors = {'no-bifurcation': '#a6a6a6', 'fold bifurcation': '#1f77b4', 'transcritical bifurcation': '#ff7f0e', 'Hopf bifurcation': '#4B0082', 'undetermined': '#d62728'}

def set_plot_style():
    plt.rcParams.update({'axes.labelsize': 24, 'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 22, 'axes.titlesize': 26, 'font.size': 22, 'lines.linewidth': 1.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 1.0})

def effective_df_lowess(x, y, span, epsilon=1e-6):
    n = len(y)
    y_base = lowess(y, x, frac=span, return_sorted=False)
    diag = np.zeros(n)
    for i in range(n):
        y_pert = y.copy()
        y_pert[i] += epsilon
        y_pert_fit = lowess(y_pert, x, frac=span, return_sorted=False)
        diag[i] = (y_pert_fit[i] - y_base[i]) / epsilon
    return np.sum(diag)

def select_best_span(x, y, spans):
    best_span = spans[0]
    best_aicc = np.inf
    best_fit = None
    n = len(y)
    for span in spans:
        fit = lowess(y, x, frac=span, return_sorted=False)
        rss = np.sum((y - fit)**2)
        df = effective_df_lowess(x, y, span)
        if df >= n - 1:
            continue
        aicc = n * np.log(rss / n) + 2 * df + (2 * df * (df + 1)) / (n - df - 1)
        if aicc < best_aicc:
            best_aicc = aicc
            best_span = span
            best_fit = fit
    return best_span, best_fit
@lru_cache(maxsize=None)

def get_species_time_spans():
    datasets_dict, _, _, _ = load_combined_data(min_records=min_records, prefilter=False)
    frames = []
    for ds_name, df in datasets_dict.items():
        if len(df) == 0:
            continue
        df = df.copy()
        if 'DATASET' in df.columns and 'dataset' not in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
        df = df.dropna(subset=['TIME', 'SPECIES_ID'])
        df['SPECIES_ID'] = df['SPECIES_ID'].astype(str).str.strip()
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].astype(str).str.strip()
        else:
            df['dataset'] = ds_name
        grouped = df.groupby(['SPECIES_ID', 'dataset']).agg(min_year=('TIME', 'min'), max_year=('TIME', 'max')).reset_index()
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(columns=['VALID_NAME', 'dataset', 'min_year', 'max_year'])
    spans = pd.concat(frames, ignore_index=True)
    return spans.rename(columns={'SPECIES_ID': 'VALID_NAME'})

def classify_dev_mode(row):
    if row['early_warning'] == True and row['bifurcation_type'] != 'no-bifurcation':
        return classify_bifurcation_type(row['bifurcation_type'])
    else:
        return 'no-bif'

def plot_transition_years(bifurcation_csv, img_folder, probability_threshold, mode='dl'):
    if mode == 'dev':
        print("Skipping transition year plot for DEV mode (no transition year).")
        return
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    df = pd.read_csv(bifurcation_csv)
    df = df[df['best_probability'] >= probability_threshold]
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    df_filtered = df[df['tipping_point_type'].isin(valid_types)].copy()
    df_filtered['year_int'] = df_filtered['transition_year'].astype(int)
    counts = df_filtered.groupby('year_int').size().reset_index(name='count')
    print("\n=== TRANSITION YEAR FIGURE DIAGNOSTICS ===")
    print(f"probability_threshold = {probability_threshold:.3g}")
    print(f"number of populations with valid tipping type = {len(df_filtered)}")
    print("counts by transition year:")
    print(counts.to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(counts['year_int'], counts['count'], width=0.8, label='Populations transitioning', color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Candidate transition year')
    ax1.set_ylabel('Populations transitioning', color='red')
#    ax1.set_xlim(left=1900)
    spans = np.arange(0.2, 0.61, 0.01)
    best_span, best_fit = select_best_span(counts['year_int'].values, counts['count'].values, spans)
    peak_idx = np.argmax(best_fit)
    peak_year = counts['year_int'].values[peak_idx]
    peak_value = best_fit[peak_idx]
    print(f"\nLOESS best span = {best_span:.2g}")
    print(f"LOESS fitted peak year = {int(peak_year)}, fitted peak count = {peak_value:.3g}")
    above_half = counts['year_int'].values[best_fit >= 0.5 * peak_value]
    if len(above_half) > 0:
        print(f"Years with fitted count >= 50% of peak: {int(above_half.min())} to {int(above_half.max())}")
    else:
        print("No years with fitted count >= 50% of peak.")
    print("\nFitted red-line values:")
    for yr, fit_val in zip(counts['year_int'].values, best_fit):
        print(f"  {int(yr)}: {fit_val:.3g}")
    ax1.plot(counts['year_int'], best_fit, color='red', linewidth=3, linestyle='--', label='')
    df_analysis = df[['species', 'dataset']].drop_duplicates()
    df_analysis['species'] = df_analysis['species'].astype(str).str.strip()
    df_analysis['dataset'] = df_analysis['dataset'].astype(str).str.strip()
    spans_df_all = get_species_time_spans()
    if not spans_df_all.empty:
        spans_df_all['species'] = spans_df_all['VALID_NAME'].astype(str).str.strip()
        spans_df_all['dataset'] = spans_df_all['dataset'].astype(str).str.strip()
        spans_df_all = spans_df_all.merge(df_analysis, on=['species', 'dataset'], how='inner')
    if spans_df_all.empty:
        print("Warning: No population spans found in raw data; skipping blue line.")
        plt.tight_layout()
        os.makedirs(img_folder, exist_ok=True)
        plt.savefig(os.path.join(img_folder, f'transition_year_plot_{probability_threshold:.2g}.pdf'), dpi=300)
        plt.close()
        return
    min_year_all = spans_df_all['min_year'].min()
    max_year_all = spans_df_all['max_year'].max()
    years = np.arange(min_year_all, max_year_all + 1)
    active_counts = []
    for y in years:
        cnt = ((spans_df_all['min_year'] <= y) & (spans_df_all['max_year'] >= y)).sum()
        active_counts.append(cnt)
    active_counts = np.array(active_counts)
    print("\n=== ACTIVE POPULATIONS / BLUE LINE DIAGNOSTICS ===")
    print(f"min_year_all = {int(min_year_all)}")
    print(f"max_year_all = {int(max_year_all)}")
    print(f"number of active populations at max_year_all = {int(active_counts[-1])}")
    peak_active_idx = np.argmax(active_counts)
    print(f"peak active-population year = {int(years[peak_active_idx])}, count = {int(active_counts[peak_active_idx])}")
    print("\nLast 10 years of active-population counts:")
    for yr, cnt in zip(years[-10:], active_counts[-10:]):
        print(f"  {int(yr)}: {int(cnt)}")
    ax2 = ax1.twinx()
    ax2.plot(years, active_counts, color='blue', linewidth=2, label='Number of populations')
    ax2.set_ylabel('Number of populations', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = [handles2[0], handles1[0]]
    labels = [labels2[0], labels1[0]]
    ax1.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f'transition_year_plot_{probability_threshold:.2g}.pdf'), dpi=300)
    plt.close()

def classify_bifurcation_type(t):
    t = str(t).lower().strip()
    if 'no-bifurcation' in t:
        return 'no-bif'
    if 'fold' in t:
        return 'fold'
    if 'transcritical' in t:
        return 'transcritical'
    if 'hopf' in t:
        return 'Hopf'
    return None

def abbreviate_bifurcation_type(name):
    if name == 'fold bifurcation':
        return 'fold bif.'
    elif name == 'transcritical bifurcation':
        return 'transc. bif.'
    elif name == 'Hopf bifurcation':
        return 'Hopf bif.'
    elif name == 'no-bifurcation':
        return 'no-bif.'
    elif name == 'undetermined':
        return 'undet.'
    else:
        return name

def add_broken_axis_indicator(ax_top, ax_bottom):
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

def _plot_prevalence_line(ax, df_counts_valid, state, y_key, label_abbr, color, spans):
    y = df_counts_valid[y_key].values
    if len(y) < 5:
        ax.plot(df_counts_valid.index, y, linewidth=2, color=color, label=label_abbr)
        return
    best_span, best_fit = select_best_span(df_counts_valid.index.values, y, spans)
    if best_fit is not None:
        ax.plot(df_counts_valid.index, best_fit, linestyle='--', linewidth=2, color=color, label=label_abbr)
    ax.plot(df_counts_valid.index, y, marker='o', linestyle='none', color=color, alpha=0.3)

def compute_prevalence_df(bifurcation_csv, probability_threshold, mode='dl', include_undetermined_in_total=False):
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return None
    df_bif = pd.read_csv(bifurcation_csv)
    if mode == 'dev':
        if 'early_warning' not in df_bif.columns or 'bifurcation_type' not in df_bif.columns:
            print(f"Skipping DEV mode plot: {bifurcation_csv} missing DEV columns.")
            return None
        df_bif = df_bif[df_bif['early_warning'].notna()].copy()
        df_bif['type_clean'] = df_bif.apply(classify_dev_mode, axis=1)
        df_bif = df_bif.dropna(subset=['type_clean']).copy()
        df_bif['has_transition'] = False
    else:
        df_bif = df_bif[df_bif['best_probability'] >= probability_threshold]
        df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
        if include_undetermined_in_total:
            df_bif['type_clean'] = df_bif['type_clean'].fillna('undetermined')
        df_bif = df_bif.dropna(subset=['type_clean']).copy()
        df_bif['has_transition'] = True
    spans_df = get_species_time_spans().copy()
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    spans_df['dataset'] = spans_df['dataset'].astype(str).str.strip()
    df_bif['species_clean'] = df_bif['species'].astype(str).str.strip()
    df_bif['dataset'] = df_bif['dataset'].astype(str).str.strip()
    merged = pd.merge(spans_df, df_bif, left_on=['species_clean', 'dataset'], right_on=['species_clean', 'dataset'], how='inner')
    if include_undetermined_in_total:
        merged = merged[merged['type_clean'].notna()]
    else:
        merged = merged[merged['type_clean'].notna() & (merged['type_clean'] != 'undetermined')]
    if merged.empty:
        print("Warning: No species with determined bifurcation type found.")
        return None
    bif_info = {}
    for _, row in merged.iterrows():
        sp = row['species_clean']
        bif_type = row['type_clean']
        if bif_type in ('no-bif', 'undetermined'):
            bif_info[sp] = {'trans_year': None, 'bif_type': bif_type}
        else:
            if mode == 'dl':
                bif_info[sp] = {'trans_year': int(row['transition_year']), 'bif_type': bif_type}
            else:
                bif_info[sp] = {'trans_year': None, 'bif_type': bif_type}
    all_years = range(int(merged['min_year'].min()), int(merged['max_year'].max()) + 1)
    state_keys = ['no-bif', 'fold', 'transcritical', 'Hopf']
    state_counts = {year: {k: 0 for k in state_keys} for year in all_years}
    active_counts = {year: 0 for year in all_years}
    for _, row in merged.iterrows():
        sp = row['species_clean']
        min_y = int(row['min_year'])
        max_y = int(row['max_year'])
        info = bif_info[sp]
        trans_year = info['trans_year']
        bif_type = info['bif_type']
        for y in range(min_y, max_y + 1):
            if include_undetermined_in_total:
                active_counts[y] += 1
            if bif_type == 'no-bif':
                state_counts[y]['no-bif'] += 1
            elif bif_type == 'undetermined':
                pass
            else:
                if trans_year is not None and y < trans_year:
                    state_counts[y]['no-bif'] += 1
                else:
                    if bif_type == 'fold':
                        state_counts[y]['fold'] += 1
                    elif bif_type == 'transcritical':
                        state_counts[y]['transcritical'] += 1
                    elif bif_type == 'Hopf':
                        state_counts[y]['Hopf'] += 1
    df_counts = pd.DataFrame.from_dict(state_counts, orient='index').sort_index()
    if include_undetermined_in_total:
        df_counts['total'] = [active_counts[y] for y in df_counts.index]
    else:
        df_counts['total'] = df_counts[state_keys].sum(axis=1)
    df_counts = df_counts[df_counts['total'] > 0]
    for state in state_keys:
        df_counts[state + '_pct'] = df_counts[state] / df_counts['total'] * 100
    valid_years = df_counts['total'] >= 20
    df_counts_valid = df_counts[valid_years]
    if df_counts_valid.empty:
        print("Warning: No year with at least 20 active species; plot not created.")
        return None
    return df_counts_valid

def plot_type_percentages(bifurcation_csv, img_folder, probability_threshold, use_two_axes=False, mode='dl', broken_yaxis=False, states_to_plot=None, filename_suffix='', save_csv=None):
    set_plot_style()
    df_counts_valid = compute_prevalence_df(bifurcation_csv, probability_threshold, mode=mode)
    if df_counts_valid is None:
        return
    if states_to_plot == ['fold'] and save_csv is not None:
        df_out = df_counts_valid[['fold_pct']].reset_index()
        df_out.columns = ['year', 'fold_prevalence']
        df_out.to_csv(save_csv, index=False)
        print("\n=== FOLD PREVALENCE DIAGNOSTICS ===")
        print(df_out.to_string(index=False))
        min_idx = df_out['fold_prevalence'].idxmin()
        max_idx = df_out['fold_prevalence'].idxmax()
        print(f"\nFold minimum: year = {int(df_out.loc[min_idx, 'year'])}, prevalence = {df_out.loc[min_idx, 'fold_prevalence']:.3g}%")
        print(f"Fold maximum: year = {int(df_out.loc[max_idx, 'year'])}, prevalence = {df_out.loc[max_idx, 'fold_prevalence']:.3g}%")
        before_max = df_out.loc[:max_idx]
        start_idx = before_max['fold_prevalence'].idxmin()
        print(f"Fold increase start (minimum before maximum): year = {int(df_out.loc[start_idx, 'year'])}, prevalence = {df_out.loc[start_idx, 'fold_prevalence']:.3g}%")
        print(f"Fold increase end (maximum): year = {int(df_out.loc[max_idx, 'year'])}, prevalence = {df_out.loc[max_idx, 'fold_prevalence']:.3g}%")
    spans = np.arange(0.2, 0.61, 0.01)
    state_to_key = {'no-bif': 'no-bifurcation', 'fold': 'fold bifurcation', 'transcritical': 'transcritical bifurcation', 'Hopf': 'Hopf bifurcation'}
    left_states = ['no-bif', 'fold', 'transcritical', 'Hopf'] if states_to_plot is None else states_to_plot
    print("\n=== BIFURCATION PREVALENCE AT FIRST AND LAST VALID YEAR ===")
    print(df_counts_valid[['no-bif_pct', 'fold_pct', 'transcritical_pct', 'Hopf_pct']].iloc[[0, -1]].to_string())
    fig = None
    ax1 = None
    ax2 = None
    if broken_yaxis:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10,6), gridspec_kw={'height_ratios':[1,1], 'hspace':0.05})
        if mode == 'dev':
            bottom_break = 10
            top_break = 90
        else:
            bottom_break = 15
            top_break = 85
        all_y_vals = []
        for state in left_states:
            y_key = state + '_pct'
            all_y_vals.extend(df_counts_valid[y_key].values)
        all_y_vals = np.array(all_y_vals)
        bottom_vals = all_y_vals[all_y_vals <= bottom_break]
        if len(bottom_vals) > 0:
            bottom_min = np.min(bottom_vals)
            bottom_max = np.max(bottom_vals)
            pad = 0.5
            bottom_min = max(0, bottom_min - pad)
            bottom_max = bottom_max + pad
        else:
            bottom_min = 0
            bottom_max = bottom_break
        top_vals = all_y_vals[all_y_vals >= top_break]
        if len(top_vals) > 0:
            top_min = np.min(top_vals)
            top_max = np.max(top_vals)
            pad = 0.5
            top_min = top_min - pad
            top_max = top_max + pad
        else:
            top_min = top_break
            top_max = 100
        ax_bottom.set_ylim(bottom_min, bottom_max)
        ax_top.set_ylim(top_min, top_max)
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_bottom.set_xlabel('Year')
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('')
        add_broken_axis_indicator(ax_top, ax_bottom)
        ax1 = ax_top
        ax2 = ax_bottom
    else:
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax2 = None
    if broken_yaxis:
        for state in left_states:
            y_key = state + '_pct'
            full_name = {'no-bif':'no-bifurcation','fold':'fold bifurcation','transcritical':'transcritical bifurcation','Hopf':'Hopf bifurcation'}[state]
            label_abbr = abbreviate_bifurcation_type(full_name)
            color = tipping_point_colors[state_to_key[state]]
            _plot_prevalence_line(ax_top, df_counts_valid, state, y_key, label_abbr, color, spans)
            _plot_prevalence_line(ax_bottom, df_counts_valid, state, y_key, label_abbr, color, spans)
#        ax_top.set_xlim(left=1900)
        state_list = left_states
        full_name_map = {'no-bif':'no-bifurcation', 'fold':'fold bifurcation', 'transcritical':'transcritical bifurcation', 'Hopf':'Hopf bifurcation'}
        color_map = {'no-bif': tipping_point_colors['no-bifurcation'], 'fold': tipping_point_colors['fold bifurcation'], 'transcritical': tipping_point_colors['transcritical bifurcation'], 'Hopf': tipping_point_colors['Hopf bifurcation']}
        proxies = []
        labels = []
        for st in state_list:
            proxies.append(plt.Line2D([0], [0], color=color_map[st], lw=2, linestyle='--'))
            labels.append(abbreviate_bifurcation_type(full_name_map[st]))
        fig.legend(handles=proxies, labels=labels, loc='center right', bbox_to_anchor=(0.85, 0.5))
        fig.text(0.04, 0.5, 'Prevalence of bifurcation\ntypes (%)', va='center', ha='center', rotation=90, fontsize=24)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85)
        plt.tight_layout(rect=[0,0,0.85,1])
        os.makedirs(img_folder, exist_ok=True)
        plt.savefig(os.path.join(img_folder, f'type_prevalence_incidence_{probability_threshold:.2g}_{mode}{filename_suffix}.pdf'), dpi=300)
        plt.close(fig)
        return
    fold_only = (states_to_plot == ['fold'] and filename_suffix == '_fold')
    for state in left_states:
        y_key = state + '_pct'
        full_name = {'no-bif':'no-bifurcation','fold':'fold bifurcation','transcritical':'transcritical bifurcation','Hopf':'Hopf bifurcation'}[state]
        label_abbr = abbreviate_bifurcation_type(full_name)
        color = 'black' if fold_only else tipping_point_colors[state_to_key[state]]
        y = df_counts_valid[y_key].values
        if fold_only:
            ax1.plot(df_counts_valid.index, y, linestyle='-', linewidth=2, marker='o', markersize=6, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1, color='black', label=label_abbr)
            continue
        if len(y) < 5:
            ax1.plot(df_counts_valid.index, y, linewidth=2, color=color, label=label_abbr)
            continue
        best_span, best_fit = select_best_span(df_counts_valid.index.values, y, spans)
        if best_fit is not None:
            ax1.plot(df_counts_valid.index, best_fit, linestyle='--', linewidth=2, color=color, label=label_abbr)
        ax1.plot(df_counts_valid.index, y, marker='o', linestyle='none', color=color, alpha=0.3)
    ax1.set_xlabel('Year')
#    ax1.set_xlim(left=1900)
    if left_states == ['fold']:
        ax1.set_ylabel('Prevalence of fold bif. (%)')
    else:
        ax1.set_ylabel('Prevalence of bifurcation\ntypes (%)')
    if left_states == ['fold']:
#        ax1.set_ylim(0, 5)
         print("\nCheck the ylim\n")
    else:
        state_list = left_states
        full_name_map = {'no-bif':'no-bifurcation', 'fold':'fold bifurcation', 'transcritical':'transcritical bifurcation', 'Hopf':'Hopf bifurcation'}
        color_map = {'no-bif': tipping_point_colors['no-bifurcation'], 'fold': tipping_point_colors['fold bifurcation'], 'transcritical': tipping_point_colors['transcritical bifurcation'], 'Hopf': tipping_point_colors['Hopf bifurcation']}
        proxies = []
        labels = []
        for st in state_list:
            proxies.append(plt.Line2D([0], [0], color=color_map[st], lw=2, linestyle='--'))
            labels.append(abbreviate_bifurcation_type(full_name_map[st]))
        ax1.legend(handles=proxies, labels=labels, loc='best')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f'type_prevalence_incidence_{probability_threshold:.2g}_{mode}{filename_suffix}.pdf'), dpi=300)
    plt.close(fig)

def plot_taxonomic_type_percentages(bifurcation_csv, img_folder, probability_threshold, mode='dl'):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    df_bif = pd.read_csv(bifurcation_csv)
    if mode == 'dev':
        if 'early_warning' not in df_bif.columns or 'bifurcation_type' not in df_bif.columns:
            print(f"Skipping DEV mode plot: {bifurcation_csv} missing DEV columns.")
            return
        df_bif = df_bif[df_bif['early_warning'].notna()].copy()
        df_bif['type_clean'] = df_bif.apply(classify_dev_mode, axis=1)
        df_bif = df_bif.dropna(subset=['type_clean', 'taxonomic_group']).copy()
        df_bif['has_transition'] = False
    else:
        df_bif = df_bif[df_bif['best_probability'] >= probability_threshold]
        df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
        df_bif = df_bif.dropna(subset=['type_clean', 'taxonomic_group']).copy()
        df_bif['has_transition'] = True
    spans_df = get_species_time_spans().copy()
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    spans_df['dataset'] = spans_df['dataset'].astype(str).str.strip()
    df_bif['species_clean'] = df_bif['species'].astype(str).str.strip()
    df_bif['dataset'] = df_bif['dataset'].astype(str).str.strip()
    merged = pd.merge(spans_df, df_bif, left_on=['species_clean', 'dataset'], right_on=['species_clean', 'dataset'], how='inner')
    merged = merged[merged['type_clean'].notna() & (merged['type_clean'] != 'undetermined')]
    if merged.empty:
        print("Warning: No species with determined bifurcation type and taxonomic group found.")
        return
    bif_info = {}
    for _, row in merged.iterrows():
        sp = row['species_clean']
        bif_type = row['type_clean']
        tax = row['taxonomic_group']
        if mode == 'dl':
            if bif_type == 'no-bif':
                bif_info[sp] = {'trans_year': None, 'bif_type': 'no-bif', 'tax_group': tax}
            else:
                bif_info[sp] = {'trans_year': int(row['transition_year']), 'bif_type': bif_type, 'tax_group': tax}
        else:
            bif_info[sp] = {'trans_year': None, 'bif_type': bif_type, 'tax_group': tax}
    merged['taxonomic_group'] = merged['species_clean'].map(lambda sp: bif_info[sp]['tax_group'])
    group_counts = merged.groupby('taxonomic_group')['species_clean'].nunique().sort_values(ascending=False)
    top_groups = [g for g in group_counts.index if g != 'Other'][:8]
    valid_groups = []
    group_percents = []
    for group in top_groups:
        group_spans = merged[merged['taxonomic_group'] == group].copy()
        if group_spans.empty:
            continue
        min_year = group_spans['min_year'].min()
        max_year = group_spans['max_year'].max()
        years = np.arange(min_year, max_year + 1)
        state_keys = ['no-bif', 'fold', 'transcritical', 'Hopf']
        state_counts = {y: {k: 0 for k in state_keys} for y in years}
        for _, row in group_spans.iterrows():
            sp = row['species_clean']
            min_y = int(row['min_year'])
            max_y = int(row['max_year'])
            info = bif_info[sp]
            trans_year = info['trans_year']
            bif_type = info['bif_type']
            for y in range(min_y, max_y + 1):
                if bif_type == 'no-bif':
                    state_counts[y]['no-bif'] += 1
                else:
                    if trans_year is not None and y < trans_year:
                        state_counts[y]['no-bif'] += 1
                    else:
                        if bif_type == 'fold':
                            state_counts[y]['fold'] += 1
                        elif bif_type == 'transcritical':
                            state_counts[y]['transcritical'] += 1
                        elif bif_type == 'Hopf':
                            state_counts[y]['Hopf'] += 1
        df_counts = pd.DataFrame.from_dict(state_counts, orient='index').sort_index()
        df_counts['total'] = df_counts[state_keys].sum(axis=1)
        df_counts = df_counts[df_counts['total'] > 0]
        for state in state_keys:
            df_counts[state + '_pct'] = df_counts[state] / df_counts['total'] * 100
        valid = df_counts['total'] >= 10
        df_counts_valid = df_counts[valid]
        if df_counts_valid.empty or len(df_counts_valid) < 5:
            continue
        valid_groups.append(group)
        group_percents.append(df_counts_valid)
    if not valid_groups:
        print("Warning: No taxonomic groups with sufficient data found.")
        return
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharex=True)
    axes_flat = axes.flatten()
    spans_loess = np.arange(0.2, 0.61, 0.01)
    state_to_key = {'no-bif': 'no-bifurcation', 'fold': 'fold bifurcation', 'transcritical': 'transcritical bifurcation', 'Hopf': 'Hopf bifurcation'}
    state_list = ['no-bif', 'fold', 'transcritical', 'Hopf']
    full_name_map = {'no-bif':'no-bifurcation', 'fold':'fold bifurcation', 'transcritical':'transcritical bifurcation', 'Hopf':'Hopf bifurcation'}
    color_map = {'no-bif': tipping_point_colors['no-bifurcation'], 'fold': tipping_point_colors['fold bifurcation'], 'transcritical': tipping_point_colors['transcritical bifurcation'], 'Hopf': tipping_point_colors['Hopf bifurcation']}
    proxies = []
    labels = []
    for st in state_list:
        proxies.append(plt.Line2D([0], [0], color=color_map[st], lw=2, linestyle='--'))
        labels.append(abbreviate_bifurcation_type(full_name_map[st]))
    for idx, (group, df_counts_valid) in enumerate(zip(valid_groups, group_percents)):
        ax1 = axes_flat[idx]
        for state in state_list:
            y = df_counts_valid[state + '_pct'].values
            full_name = full_name_map[state]
            label_abbr = abbreviate_bifurcation_type(full_name)
            color = tipping_point_colors[state_to_key[state]]
            if len(y) < 5:
                ax1.plot(df_counts_valid.index, y, linewidth=2, color=color, label=label_abbr)
                continue
            best_span, best_fit = select_best_span(df_counts_valid.index.values, y, spans_loess)
            if best_fit is not None:
                ax1.plot(df_counts_valid.index, best_fit, linestyle='--', linewidth=2, color=color, label=label_abbr)
            ax1.plot(df_counts_valid.index, y, marker='o', linestyle='none', color=color, alpha=0.3)
        ax1.set_ylabel(f'Prevalence for {group}')
        ax1.legend(handles=proxies, labels=labels, loc='best')
    for idx in range(len(valid_groups), len(axes_flat)):
        axes_flat[idx].axis('off')
    axes_flat[-1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, f'taxonomic_type_prevalence_incidence_{probability_threshold:.2g}_{mode}.pdf'), dpi=300)
    plt.close(fig)

def aggregate_prevalence_by_decade(df_counts_valid, state_keys):
    df = df_counts_valid.copy()
    df['decade'] = (df.index // 10) * 10
    agg = df.groupby('decade')[state_keys + ['total']].sum()
    for st in state_keys:
        agg[st + '_pct'] = agg[st] / agg['total'] * 100
    return agg

def plot_threshold_sensitivity_prevalence(bifurcation_csv, img_folder, thresholds, mode='dl'):
    set_plot_style()
    state_keys = ['Hopf', 'fold', 'transcritical', 'no-bif']
    cmap = plt.get_cmap('viridis')
    colors = {t: cmap(i / max(1, len(thresholds) - 1)) for i, t in enumerate(thresholds)}
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    for thr in thresholds:
        dfc = compute_prevalence_df(bifurcation_csv, thr, mode=mode)
        if dfc is None:
            continue
        dfc_dec = aggregate_prevalence_by_decade(dfc, state_keys)
        for ax, st in zip(axes.flatten(), state_keys):
            ax.plot(dfc_dec.index, dfc_dec[st + '_pct'], 'o-', lw=1.8, color=colors[thr], label=f'{thr:.2g}')
    for ax, st in zip(axes.flatten(), state_keys):
        st_label = st[0].upper() + st[1:]
        ax.set_ylabel(f'{st_label} prevalence (%)')
        ax.set_xlabel('Year')
        ax.legend(title='prob. threshold')
    for ax, panel_label in zip(axes.flatten(), ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.0, 1.02, panel_label, transform=ax.transAxes, ha='left', va='bottom')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f'threshold_sensitivity_prevalence_{mode}.pdf'), dpi=300)
    plt.close(fig)

def plot_threshold_sensitivity_slopes(bifurcation_csv, img_folder, thresholds, mode='dl'):
    set_plot_style()
    state_keys = ['no-bif', 'fold', 'transcritical', 'Hopf']
    state_to_key = {'no-bif': 'no-bifurcation', 'fold': 'fold bifurcation', 'transcritical': 'transcritical bifurcation', 'Hopf': 'Hopf bifurcation'}
    full_name_map = {'no-bif': 'no-bifurcation', 'fold': 'fold bifurcation', 'transcritical': 'transcritical bifurcation', 'Hopf': 'Hopf bifurcation'}
    rows = []
    for thr in thresholds:
        dfc = compute_prevalence_df(bifurcation_csv, thr, mode=mode)
        if dfc is None or len(dfc) < 5:
            continue
        x = dfc.index.values.astype(float)
        for st in state_keys:
            y = dfc[st + '_pct'].values
            slope, intercept = np.polyfit(x, y, 1)
            rows.append({'threshold': thr, 'type': st, 'slope_pct_per_decade': slope * 10})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 6))
    for st in state_keys:
        g = df[df['type'] == st].sort_values('threshold')
        ax.plot(g['threshold'], g['slope_pct_per_decade'], 'o-', lw=2, color=tipping_point_colors[state_to_key[st]], label=abbreviate_bifurcation_type(full_name_map[st]))
    ax.axhline(0, color='k', lw=1, ls=':')
    ax.set_xlabel('probability threshold')
    ax.set_ylabel('Trend in prevalence\n(% per decade)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f'threshold_sensitivity_slopes_{mode}.pdf'), dpi=300)
    plt.close(fig)

def run_pipeline(bifurcation_csv, img_folder, modes, probability_thresholds, prevalence_thresholds, slope_thresholds):
    for mode in modes:
        if mode == 'dl':
            for thresh in probability_thresholds:
                plot_transition_years(bifurcation_csv, img_folder, thresh, mode=mode)
                plot_type_percentages(bifurcation_csv, img_folder, thresh, use_two_axes=False, mode=mode, broken_yaxis=False)
                plot_type_percentages(bifurcation_csv, img_folder, thresh, mode=mode, broken_yaxis=False, states_to_plot=['fold'], filename_suffix='_fold', save_csv='csv/fold_prevalence.csv')
        else:
            plot_transition_years(bifurcation_csv, img_folder, probability_threshold, mode=mode)
            plot_type_percentages(bifurcation_csv, img_folder, probability_threshold, use_two_axes=False, mode=mode, broken_yaxis=False)
    plot_threshold_sensitivity_prevalence(bifurcation_csv, img_folder, prevalence_thresholds, mode='dl')
#    plot_threshold_sensitivity_slopes(bifurcation_csv, img_folder, slope_thresholds, mode='dl')

def main():
    modes = ['dl']
    probability_thresholds = [probability_threshold]
    prevalence_thresholds = np.array([0.5, 0.6, 0.7, 0.8])
    slope_thresholds = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8])
    bifurcation_csv = 'csv/bifurcation_results_short.csv'
    img_folder = 'img'
    run_pipeline(bifurcation_csv, img_folder, modes, probability_thresholds, prevalence_thresholds, slope_thresholds)

if __name__ == "__main__":
    main()
