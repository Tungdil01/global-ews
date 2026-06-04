import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

tipping_point_colors = {'no-bifurcation': '#999999', 'fold bifurcation': '#1f77b4', 'transcritical bifurcation': '#ff7f0e', 'Hopf bifurcation': '#9467bd'}

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

def get_species_time_spans(raw_data_csv):
    cols = ['valid_name', 'YEAR']
    try:
        df = pd.read_csv(raw_data_csv, encoding='latin-1', usecols=cols)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(raw_data_csv, encoding='ISO-8859-1', usecols=cols)
        except:
            df = pd.read_csv(raw_data_csv, usecols=cols)
    df.columns = df.columns.str.strip().str.upper()
    if 'VALID_NAME' not in df.columns or 'YEAR' not in df.columns:
        raise ValueError("Required columns not found in raw data.")
    df = df.dropna(subset=['YEAR', 'VALID_NAME'])
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df = df.dropna(subset=['YEAR'])
    spans = df.groupby('VALID_NAME').agg(min_year=('YEAR', 'min'), max_year=('YEAR', 'max')).reset_index()
    return spans

def plot_transition_years(bifurcation_csv, raw_data_csv, img_folder):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    df = pd.read_csv(bifurcation_csv)
    df_filtered = df[df['tipping_point_type'] != 'no-bifurcation'].copy()
    df_filtered['year_int'] = df_filtered['transition_year'].astype(int)
    counts = df_filtered.groupby('year_int').size().reset_index(name='count')
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(counts['year_int'], counts['count'], width=0.8, label='Species count per year', color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Changepoint Year')
    ax1.set_ylabel('Number of Species', color='red')
    spans = np.arange(0.2, 0.61, 0.01)
    best_span, best_fit = select_best_span(counts['year_int'].values, counts['count'].values, spans)
    ax1.plot(counts['year_int'], best_fit, color='red', linewidth=3, label=f'LOESS trend (span={best_span:.2g})')
    if os.path.exists(raw_data_csv):
        spans_df = get_species_time_spans(raw_data_csv)
        min_year_all = spans_df['min_year'].min()
        max_year_all = spans_df['max_year'].max()
        years = np.arange(min_year_all, max_year_all + 1)
        active_counts = []
        for y in years:
            cnt = ((spans_df['min_year'] <= y) & (spans_df['max_year'] >= y)).sum()
            active_counts.append(cnt)
        active_counts = np.array(active_counts)
        ax2 = ax1.twinx()
        ax2.plot(years, active_counts, color='blue', linewidth=2, label='Active time series')
        ax2.set_ylabel('Number of active time series', color='blue')
        ax2.tick_params(axis='y', labelcolor='black')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        desired_order = [2, 0, 1]
        handles = [handles2[0], handles1[0], handles1[1]]
        labels = [labels2[0], labels1[0], labels1[1]]
        ax1.legend(handles, labels, loc='upper left')
    else:
        print(f"Warning: {raw_data_csv} not found; coverage line omitted.")
        ax1.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, 'transition_year_plot.pdf'), dpi=300)
    plt.close()

def plot_relative_transition_years(bifurcation_csv, raw_data_csv, img_folder):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    if not os.path.exists(raw_data_csv):
        print(f"Error: file {raw_data_csv} not found.")
        return
    df_bif = pd.read_csv(bifurcation_csv)
    df_bif_filtered = df_bif[df_bif['tipping_point_type'] != 'no-bifurcation'].copy()
    df_bif_filtered['year_int'] = df_bif_filtered['transition_year'].astype(int)
    df_span = get_species_time_spans(raw_data_csv)
    df_bif_filtered['species_clean'] = df_bif_filtered['species'].astype(str).str.strip()
    df_span['VALID_NAME_clean'] = df_span['VALID_NAME'].astype(str).str.strip()
    df_merged = pd.merge(df_bif_filtered, df_span, left_on='species_clean', right_on='VALID_NAME_clean', how='inner')
    if df_merged.empty:
        print("No matching species found.")
        print(f"Unique species in bifurcation: {df_bif_filtered['species_clean'].nunique()}")
        print(f"Unique VALID_NAME in raw data: {df_span['VALID_NAME_clean'].nunique()}")
        return
    df_merged = df_merged[df_merged['max_year'] > df_merged['min_year']].copy()
    df_merged['relative_year'] = (df_merged['year_int'] - df_merged['min_year']) / (df_merged['max_year'] - df_merged['min_year'])
    df_merged = df_merged[(df_merged['relative_year'] >= 0) & (df_merged['relative_year'] <= 1)]
    plt.figure(figsize=(10,6))
    plt.hist(df_merged['relative_year'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Relative transition year (0=start, 1=end)')
    plt.ylabel('Number of species')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Midpoint')
    plt.legend()
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, 'relative_transition_year_plot.pdf'), dpi=300)
    plt.close()

def classify_bifurcation_type(t):
    t = str(t).lower().strip()
    if 'no-bifurcation' in t:
        return 'no-bifurcation'
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
    else:
        return name

def plot_type_percentages(bifurcation_csv, img_folder, raw_data_csv=None):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    if not raw_data_csv or not os.path.exists(raw_data_csv):
        print("Error: raw data file required for active species counts.")
        return
    df_bif = pd.read_csv(bifurcation_csv)
    df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
    df_bif = df_bif.dropna(subset=['type_clean']).copy()
    spans_df = get_species_time_spans(raw_data_csv)
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    bif_info = {}
    for _, row in df_bif.iterrows():
        sp = str(row['species']).strip()
        bif_type = row['type_clean']
        if bif_type == 'no-bifurcation':
            continue
        bif_info[sp] = {'trans_year': int(row['transition_year']), 'bif_type': bif_type}
    all_years = np.arange(spans_df['min_year'].min(), spans_df['max_year'].max() + 1)
    state_counts = {year: {'no-bif': 0, 'fold': 0, 'transcritical': 0, 'Hopf': 0} for year in all_years}
    incidence = {year: 0 for year in all_years}
    for _, row in spans_df.iterrows():
        sp = row['species_clean']
        min_y = int(row['min_year'])
        max_y = int(row['max_year'])
        info = bif_info.get(sp)
        trans_year = info['trans_year'] if info else None
        for y in range(min_y, max_y + 1):
            if info is None or y < trans_year:
                state_counts[y]['no-bif'] += 1
            else:
                if info['bif_type'] == 'fold':
                    state_counts[y]['fold'] += 1
                elif info['bif_type'] == 'transcritical':
                    state_counts[y]['transcritical'] += 1
                elif info['bif_type'] == 'Hopf':
                    state_counts[y]['Hopf'] += 1
                if y == trans_year:
                    incidence[y] += 1
    df_counts = pd.DataFrame.from_dict(state_counts, orient='index').sort_index()
    active = df_counts.sum(axis=1)
    df_percent = df_counts.div(active, axis=0) * 100
    valid_years = active >= 20
    df_percent = df_percent[valid_years]
    active_valid = active[valid_years]
    if df_percent.empty:
        print("Warning: No year with at least 20 active species; plot not created.")
        return
    inc_series = pd.Series(incidence, index=all_years)
    inc_series = inc_series[valid_years]
    no_bif_counts = df_counts['no-bif'][valid_years]
    at_risk = no_bif_counts + inc_series
    rate = (inc_series / at_risk) * 1000
    fig, ax1 = plt.subplots(figsize=(10,6))
    spans = np.arange(0.2, 0.61, 0.01)
    for state in ['no-bif', 'fold', 'transcritical', 'Hopf']:
        if state not in df_percent.columns:
            continue
        y = df_percent[state].values
        full_name = {'no-bif':'no-bifurcation','fold':'fold bifurcation','transcritical':'transcritical bifurcation','Hopf':'Hopf bifurcation'}[state]
        label_abbr = abbreviate_bifurcation_type(full_name)
        if len(y) < 5:
            ax1.plot(df_percent.index, y, linewidth=2, color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), label=label_abbr)
            continue
        best_span, best_fit = select_best_span(df_percent.index.values, y, spans)
        if best_fit is not None:
            ax1.plot(df_percent.index, best_fit, linewidth=2, color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), label=label_abbr)
        ax1.plot(df_percent.index, y, marker='o', linestyle='none', color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), alpha=0.3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Prevalence of bifurcation types')
    ax2 = ax1.twinx()
    ax2.plot(rate.index, rate, marker='o', linewidth=2, color='darkgreen', alpha=0.7, label='')
    pre_span_incidence = 0.37
    if len(rate) >= 5:
        if pre_span_incidence is not None:
            best_fit = lowess(rate.values, rate.index, frac=pre_span_incidence, return_sorted=False)
            ax2.plot(rate.index, best_fit, linestyle='--', linewidth=2, color='darkgreen', label=f'LOESS (span={pre_span_incidence:.2g})')
        else:
            best_span, best_fit = select_best_span(rate.index.values, rate.values, spans)
            if best_fit is not None:
                ax2.plot(rate.index, best_fit, linestyle='--', linewidth=2, color='darkgreen', label=f'LOESS (span={best_span:.2g})')
    ax2.set_ylabel('Incidence Rate of bifurcations', color='darkgreen')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, 'type_prevalence_incidence.pdf'), dpi=300)
    plt.close(fig)

def plot_taxonomic_type_percentages(bifurcation_csv, img_folder, raw_data_csv=None):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    if not raw_data_csv or not os.path.exists(raw_data_csv):
        print("Error: raw data file required for active species counts.")
        return
    df_bif = pd.read_csv(bifurcation_csv)
    df_bif['type_clean'] = df_bif['tipping_point_type'].apply(classify_bifurcation_type)
    df_bif = df_bif.dropna(subset=['type_clean', 'taxonomic_group']).copy()
    spans_df = get_species_time_spans(raw_data_csv)
    spans_df['species_clean'] = spans_df['VALID_NAME'].astype(str).str.strip()
    bif_info = {}
    for _, row in df_bif.iterrows():
        sp = str(row['species']).strip()
        bif_type = row['type_clean']
        if bif_type == 'no-bifurcation':
            continue
        tax = row['taxonomic_group']
        bif_info[sp] = {'trans_year': int(row['transition_year']), 'bif_type': bif_type, 'tax_group': tax}
    species_taxon = {sp: info['tax_group'] for sp, info in bif_info.items()}
    spans_df['taxonomic_group'] = spans_df['species_clean'].map(species_taxon)
    spans_df = spans_df.dropna(subset=['taxonomic_group'])
    group_counts = spans_df.groupby('taxonomic_group')['species_clean'].nunique().sort_values(ascending=False)
    top_groups = [g for g in group_counts.index if g != 'Other'][:8]
    valid_groups = []
    group_percents = []
    group_incidence = []
    pre_spans_tax = {
        'Actinopterygii': 0.24, 'Hexapoda': 0.31, 'Tracheophyta': 0.21, 'Aves': 0.22,
        'Malacostraca': 0.22, 'Annelida': 0.52, 'Ochrophyta': 0.45, 'Myzozoa': 0.55,
        'Echinodermata': 0.54, 'Mammalia': 0.25, 'Chelicerata': 0.2, 'Lissamphibia': 0.3,
        'Tunicata': 0.54, 'Bryozoa': 0.5, 'Bacteria': 0.23, 'Copepoda': 0.24, 'Mollusca': 0.6, 'Cnidaria': 0.6, 'Maxillopoda': 0.6
    }
    expand_groups = None
    for group in top_groups:
        group_spans = spans_df[spans_df['taxonomic_group'] == group].copy()
        if group_spans.empty:
            continue
        min_year = group_spans['min_year'].min()
        max_year = group_spans['max_year'].max()
        years = np.arange(min_year, max_year + 1)
        state_counts = {y: {'no-bif': 0, 'fold': 0, 'transcritical': 0, 'Hopf': 0} for y in years}
        incidence = {y: 0 for y in years}
        for _, row in group_spans.iterrows():
            sp = row['species_clean']
            min_y = int(row['min_year'])
            max_y = int(row['max_year'])
            info = bif_info.get(sp)
            trans_year = info['trans_year'] if info else None
            for y in range(min_y, max_y + 1):
                if info is None or y < trans_year:
                    state_counts[y]['no-bif'] += 1
                else:
                    if info['bif_type'] == 'fold':
                        state_counts[y]['fold'] += 1
                    elif info['bif_type'] == 'transcritical':
                        state_counts[y]['transcritical'] += 1
                    elif info['bif_type'] == 'Hopf':
                        state_counts[y]['Hopf'] += 1
                    if y == trans_year:
                        incidence[y] += 1
        df_counts = pd.DataFrame.from_dict(state_counts, orient='index').sort_index()
        active = df_counts.sum(axis=1)
        df_percent = df_counts.div(active, axis=0) * 100
        valid = active >= 10
        df_percent = df_percent[valid]
        active_valid = active[valid]
        inc_series = pd.Series(incidence).reindex(df_percent.index, fill_value=0)
        if df_percent.empty or len(df_percent) < 5:
            continue
        valid_groups.append(group)
        group_percents.append(df_percent)
        group_incidence.append((inc_series, df_counts['no-bif'][valid]))
    if not valid_groups:
        print("Warning: No taxonomic groups with sufficient data found.")
        return
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharex=True)
    axes_flat = axes.flatten()
    spans_loess = np.arange(0.2, 0.61, 0.01)
    for idx, (group, df_percent) in enumerate(zip(valid_groups, group_percents)):
        ax1 = axes_flat[idx]
        for state in ['no-bif', 'fold', 'transcritical', 'Hopf']:
            if state not in df_percent.columns:
                continue
            y = df_percent[state].values
            full_name = {'no-bif':'no-bifurcation','fold':'fold bifurcation','transcritical':'transcritical bifurcation','Hopf':'Hopf bifurcation'}[state]
            label_abbr = abbreviate_bifurcation_type(full_name)
            if len(y) < 5:
                ax1.plot(df_percent.index, y, linewidth=2, color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), label=label_abbr)
                continue
            best_span, best_fit = select_best_span(df_percent.index.values, y, spans_loess)
            if best_fit is not None:
                ax1.plot(df_percent.index, best_fit, linewidth=2, color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), label=label_abbr)
            ax1.plot(df_percent.index, y, marker='o', linestyle='none', color=tipping_point_colors.get(state + ' bifurcation', tipping_point_colors['no-bifurcation']), alpha=0.3)
        ax1.set_ylabel(f'Prevalence for {group}')
        inc_series, no_bif_counts = group_incidence[idx]
        at_risk = no_bif_counts + inc_series
        rate = (inc_series / at_risk) * 1000
        ax2 = ax1.twinx()
        ax2.plot(rate.index, rate, marker='o', linewidth=2, color='darkgreen', alpha=0.7, label='')
        if len(rate) >= 8:
            if group in pre_spans_tax:
                span = pre_spans_tax[group]
                best_fit = lowess(rate.values, rate.index, frac=span, return_sorted=False)
                ax2.plot(rate.index, best_fit, linestyle='--', linewidth=2, color='darkgreen', label=f'LOESS (span={span:.2g})')
            elif group in expand_groups:
                span = 0.6
                best_fit = lowess(rate.values, rate.index, frac=span, return_sorted=False)
                ax2.plot(rate.index, best_fit, linestyle='--', linewidth=2, color='darkgreen', label=f'LOESS (span={span:.2g})')
            else:
                best_span, best_fit = select_best_span(rate.index.values, rate.values, spans_loess)
                if best_fit is not None:
                    ax2.plot(rate.index, best_fit, linestyle='--', linewidth=2, color='darkgreen', label=f'LOESS (span={best_span:.2g})')
        ax2.set_ylabel('Incidence Rate', color='darkgreen')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')
    for idx in range(len(valid_groups), len(axes_flat)):
        axes_flat[idx].axis('off')
    axes_flat[-1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'taxonomic_type_prevalence_incidence.pdf'), dpi=300)
    plt.close(fig)

def main():
    bifurcation_csv = 'csv/bifurcation_results_short.csv'
    raw_data_csv = 'data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv'
    img_folder = 'img'
    plot_transition_years(bifurcation_csv, raw_data_csv, img_folder)
    plot_type_percentages(bifurcation_csv, img_folder, raw_data_csv)
    plot_taxonomic_type_percentages(bifurcation_csv, img_folder, raw_data_csv)

if __name__ == "__main__":
    main()
