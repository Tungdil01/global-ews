#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.patches import Patch

tipping_point_colors = {'no-bifurcation': '#999999', 'fold bifurcation': '#1f77b4', 'transcritical bifurcation': '#ff7f0e', 'Hopf bifurcation': '#9467bd', 'undetermined': '#d62728'}

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'axes.titlesize': 26,
        'font.size': 22,
        'lines.linewidth': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0
    })

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

def find_column(df, candidates):
    for col in df.columns:
        if col.lower() in [c.lower() for c in candidates]:
            return col
    return None

def get_species_time_spans(raw_data_files):
    if isinstance(raw_data_files, str):
        raw_data_files = [raw_data_files]
    all_records = []
    for f in raw_data_files:
        f_lower = f.lower()
        try:
            if 'biotime' in f_lower:
                df = pd.read_csv(f, encoding='latin-1', low_memory=False)
            elif 'rivfishtime' in f_lower:
                df = pd.read_csv(f, encoding='latin-1', low_memory=False)
            elif 'knb-lter-cdr' in f_lower or 'cedar' in f_lower:
                if 'arthropod' in f_lower:
                    df = pd.read_csv(f, sep='\t', low_memory=False)
                else:
                    df = pd.read_csv(f, low_memory=False)
            elif 'gpdd' in f_lower:
                df = pd.read_csv(f, low_memory=False)
            else:
                continue
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        if 'arthropod' in f_lower:
            genus_col = find_column(df, ['genus'])
            epithet_col = find_column(df, ['specific.epithet'])
            year_col = find_column(df, ['year'])
            if genus_col is None or epithet_col is None or year_col is None:
                print(f"Available columns in {f}: {list(df.columns)}")
                raise ValueError(f"Arthropod file missing genus, epithet, or year column in {f}")
            df = df[[genus_col, epithet_col, year_col]].dropna()
            df['SPECIES'] = df[genus_col].astype(str).str.strip() + ' ' + df[epithet_col].astype(str).str.strip()
            df = df.rename(columns={year_col: 'YEAR'})
            df = df[['SPECIES', 'YEAR']]
        else:
            species_candidates = ['valid_name', 'species', 'SPECIES', 'Species', 'scientific_name']
            year_candidates = ['year', 'YEAR', 'Year']
            species_col = find_column(df, species_candidates)
            year_col = find_column(df, year_candidates)
            if species_col is None:
                print(f"Available columns in {f}: {list(df.columns)}")
                raise ValueError(f"Species column not found in {f}")
            if year_col is None:
                print(f"Available columns in {f}: {list(df.columns)}")
                raise ValueError(f"Year column not found in {f}")
            df = df[[species_col, year_col]].dropna()
            df = df.rename(columns={species_col: 'SPECIES', year_col: 'YEAR'})
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df = df.dropna(subset=['YEAR', 'SPECIES'])
        df['SPECIES'] = df['SPECIES'].astype(str).str.strip()
        all_records.append(df[['SPECIES', 'YEAR']])
    if not all_records:
        raise ValueError("No valid records found in any raw data file.")
    combined = pd.concat(all_records, ignore_index=True)
    spans = combined.groupby('SPECIES').agg(min_year=('YEAR', 'min'), max_year=('YEAR', 'max')).reset_index()
    return spans.rename(columns={'SPECIES': 'VALID_NAME'})

def classify_dev_mode(row):
    if row['early_warning'] == True and row['bifurcation_type'] != 'no-bifurcation':
        return classify_bifurcation_type(row['bifurcation_type'])
    else:
        return 'no-bif'

def plot_transition_years(bifurcation_csv, raw_data_files, img_folder, probability_threshold, mode='dl'):
    if mode == 'dev':
        print("Skipping transition year plot for DEV mode (no transition year).")
        return
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    df = pd.read_csv(bifurcation_csv)
    df = df[df['best_probability'] >= probability_threshold]
    # Keep only fold, transcritical, Hopf (exclude no-bifurcation and undetermined)
    valid_types = ['fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    df_filtered = df[df['tipping_point_type'].isin(valid_types)].copy()
    df_filtered['year_int'] = df_filtered['transition_year'].astype(int)
    counts = df_filtered.groupby('year_int').size().reset_index(name='count')
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(counts['year_int'], counts['count'], width=0.8, label='Species count per year', color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Tipping point candidate year')
    ax1.set_ylabel('Number of Species', color='red')
    ax1.set_xlim(left=1900)
    spans = np.arange(0.2, 0.61, 0.01)
    best_span, best_fit = select_best_span(counts['year_int'].values, counts['count'].values, spans)
    ax1.plot(counts['year_int'], best_fit, color='red', linewidth=3, linestyle='--', label='')
    # Blue line: active time series across all raw data (all species)
    spans_df_all = get_species_time_spans(raw_data_files)
    if spans_df_all.empty:
        print("Warning: No species spans found in raw data; skipping blue line.")
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
    ax2 = ax1.twinx()
    ax2.plot(years, active_counts, color='blue', linewidth=2, label='Number of time series')
    ax2.set_ylabel('Number of time series', color='blue')
    ax2.tick_params(axis='y', labelcolor='black')
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
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
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

def plot_type_percentages(bifurcation_csv, raw_data_files, img_folder, probability_threshold, use_two_axes=False, mode='dl', broken_yaxis=False):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    spans_df = get_species_time_spans(raw_data_files)
    df_bif = pd.read_csv(bifurcation_csv)
    if mode == 'dl':
        df_bif = df_bif[df_bif['best_probability'] >= probability_threshold]
        df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
        df_bif = df_bif.dropna(subset=['type_clean']).copy()
        df_bif['has_transition'] = True
    else:
        df_bif = df_bif[df_bif['early_warning'].notna()].copy()
        df_bif['type_clean'] = df_bif.apply(classify_dev_mode, axis=1)
        df_bif = df_bif.dropna(subset=['type_clean']).copy()
        df_bif['has_transition'] = False
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    df_bif['species_clean'] = df_bif['species'].astype(str).str.strip()
    merged = pd.merge(spans_df, df_bif, left_on='species_clean', right_on='species_clean', how='inner')
    merged = merged[merged['type_clean'].notna() & (merged['type_clean'] != 'undetermined')]
    if merged.empty:
        print("Warning: No species with determined bifurcation type found.")
        return
    bif_info = {}
    for _, row in merged.iterrows():
        sp = row['species_clean']
        bif_type = row['type_clean']
        if mode == 'dl':
            if bif_type == 'no-bif':
                bif_info[sp] = {'trans_year': None, 'bif_type': 'no-bif'}
            else:
                bif_info[sp] = {'trans_year': int(row['transition_year']), 'bif_type': bif_type}
        else:
            bif_info[sp] = {'trans_year': None, 'bif_type': bif_type}
    all_years = np.arange(merged['min_year'].min(), merged['max_year'].max() + 1)
    state_keys = ['no-bif', 'fold', 'transcritical', 'Hopf']
    state_counts = {year: {k: 0 for k in state_keys} for year in all_years}
    for _, row in merged.iterrows():
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
    valid_years = df_counts['total'] >= 20
    df_counts_valid = df_counts[valid_years]
    if df_counts_valid.empty:
        print("Warning: No year with at least 20 active species; plot not created.")
        return
    spans = np.arange(0.2, 0.61, 0.01)
    state_to_key = {'no-bif': 'no-bifurcation', 'fold': 'fold bifurcation', 'transcritical': 'transcritical bifurcation', 'Hopf': 'Hopf bifurcation'}
    left_states = ['no-bif', 'fold', 'transcritical', 'Hopf']
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
        ax_top.set_xlim(left=1900)
        state_list = ['no-bif', 'fold', 'transcritical', 'Hopf']
        full_name_map = {'no-bif':'no-bifurcation', 'fold':'fold bifurcation', 'transcritical':'transcritical bifurcation', 'Hopf':'Hopf bifurcation'}
        color_map = {'no-bif': tipping_point_colors['no-bifurcation'],
                     'fold': tipping_point_colors['fold bifurcation'],
                     'transcritical': tipping_point_colors['transcritical bifurcation'],
                     'Hopf': tipping_point_colors['Hopf bifurcation']}
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
        plt.savefig(os.path.join(img_folder, f'type_prevalence_incidence_{probability_threshold:.2g}_{mode}.pdf'), dpi=300)
        plt.close(fig)
        return
    for state in left_states:
        y_key = state + '_pct'
        full_name = {'no-bif':'no-bifurcation','fold':'fold bifurcation','transcritical':'transcritical bifurcation','Hopf':'Hopf bifurcation'}[state]
        label_abbr = abbreviate_bifurcation_type(full_name)
        color = tipping_point_colors[state_to_key[state]]
        y = df_counts_valid[y_key].values
        if len(y) < 5:
            ax1.plot(df_counts_valid.index, y, linewidth=2, color=color, label=label_abbr)
            continue
        best_span, best_fit = select_best_span(df_counts_valid.index.values, y, spans)
        if best_fit is not None:
            ax1.plot(df_counts_valid.index, best_fit, linestyle='--', linewidth=2, color=color, label=label_abbr)
        ax1.plot(df_counts_valid.index, y, marker='o', linestyle='none', color=color, alpha=0.3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Prevalence of bifurcation\ntypes (%)')
    state_list = ['no-bif', 'fold', 'transcritical', 'Hopf']
    full_name_map = {'no-bif':'no-bifurcation', 'fold':'fold bifurcation', 'transcritical':'transcritical bifurcation', 'Hopf':'Hopf bifurcation'}
    color_map = {'no-bif': tipping_point_colors['no-bifurcation'],
                 'fold': tipping_point_colors['fold bifurcation'],
                 'transcritical': tipping_point_colors['transcritical bifurcation'],
                 'Hopf': tipping_point_colors['Hopf bifurcation']}
    proxies = []
    labels = []
    for st in state_list:
        proxies.append(plt.Line2D([0], [0], color=color_map[st], lw=2, linestyle='--'))
        labels.append(abbreviate_bifurcation_type(full_name_map[st]))
    ax1.legend(handles=proxies, labels=labels, loc='best')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f'type_prevalence_incidence_{probability_threshold:.2g}_{mode}.pdf'), dpi=300)
    plt.close(fig)

def plot_taxonomic_type_percentages(bifurcation_csv, raw_data_files, img_folder, probability_threshold, mode='dl'):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    spans_df = get_species_time_spans(raw_data_files)
    df_bif = pd.read_csv(bifurcation_csv)
    if mode == 'dl':
        df_bif = df_bif[df_bif['best_probability'] >= probability_threshold]
        df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
        df_bif = df_bif.dropna(subset=['type_clean', 'taxonomic_group']).copy()
        df_bif['has_transition'] = True
    else:
        df_bif = df_bif[df_bif['early_warning'].notna()].copy()
        df_bif['type_clean'] = df_bif.apply(classify_dev_mode, axis=1)
        df_bif = df_bif.dropna(subset=['type_clean', 'taxonomic_group']).copy()
        df_bif['has_transition'] = False
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    df_bif['species_clean'] = df_bif['species'].astype(str).str.strip()
    merged = pd.merge(spans_df, df_bif, left_on='species_clean', right_on='species_clean', how='inner')
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
    color_map = {'no-bif': tipping_point_colors['no-bifurcation'],
                 'fold': tipping_point_colors['fold bifurcation'],
                 'transcritical': tipping_point_colors['transcritical bifurcation'],
                 'Hopf': tipping_point_colors['Hopf bifurcation']}
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

def main():
    modes = ['dl', 'dev']
    probability_thresholds = [0.7]
    bifurcation_csv = 'csv/bifurcation_results_short.csv'
    data_dir = 'data'
    raw_data_files = [
        os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv'),
        os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv'),
        os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
        os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
        os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv'),
        os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
    ]
    img_folder = 'img'
    for mode in modes:
        if mode == 'dl':
            for thresh in probability_thresholds:
                plot_transition_years(bifurcation_csv, raw_data_files, img_folder, thresh, mode=mode)
                plot_type_percentages(bifurcation_csv, raw_data_files, img_folder, thresh, use_two_axes=False, mode=mode, broken_yaxis=False)
#                plot_taxonomic_type_percentages(bifurcation_csv, raw_data_files, img_folder, thresh, mode=mode)
        else:
            plot_transition_years(bifurcation_csv, raw_data_files, img_folder, 0.7, mode=mode)
            plot_type_percentages(bifurcation_csv, raw_data_files, img_folder, 0.7, use_two_axes=False, mode=mode, broken_yaxis=False)
#            plot_taxonomic_type_percentages(bifurcation_csv, raw_data_files, img_folder, 0.7, mode=mode)

if __name__ == "__main__":
    main()
