import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
tipping_point_colors = {'no bifurcation': '#999999', 'fold bifurcation': '#1f77b4', 'transcritical bifurcation': '#ff7f0e', 'Hopf bifurcation': '#9467bd'}

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
    df_filtered = df[df['tipping_point_type'] != 'no bifurcation'].copy()
    df_filtered['year_int'] = df_filtered['transition_year'].astype(int)
    counts = df_filtered.groupby('year_int').size().reset_index(name='count')
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(counts['year_int'], counts['count'], width=0.8, label='Species count per year', color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Changepoint Year')
    ax1.set_ylabel('Number of Species', color='red')
    spans = np.arange(0.1, 0.6, 0.05)
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
    df_bif_filtered = df_bif[df_bif['tipping_point_type'] != 'no bifurcation'].copy()
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
    if 'no bifurcation' in t:
        return 'no bifurcation'
    if 'fold' in t:
        return 'fold'
    if 'transcritical' in t:
        return 'transcritical'
    return None

def plot_type_percentages(bifurcation_csv, img_folder):
    set_plot_style()
    if not os.path.exists(bifurcation_csv):
        print(f"Error: file {bifurcation_csv} not found.")
        return
    df = pd.read_csv(bifurcation_csv)
    df['type_clean'] = df['tipping_point_type'].apply(classify_bifurcation_type)
    df = df.dropna(subset=['type_clean', 'transition_year']).copy()
    df['year_int'] = df['transition_year'].astype(int)
    grouped = df.groupby(['year_int', 'type_clean']).size().reset_index(name='count')
    pivot = grouped.pivot(index='year_int', columns='type_clean', values='count').fillna(0)
    for col in ['fold', 'transcritical', 'no bifurcation']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[['fold', 'transcritical', 'no bifurcation']]
    total = pivot.sum(axis=1)
    start_candidates = total[total >= 20]
    if start_candidates.empty:
        print("Warning: No year with at least 20 species; plot not created.")
        return
    start_year = start_candidates.index.min()
    pivot = pivot.loc[start_year:]
    total = total.loc[start_year:]
    years_full = np.arange(start_year, pivot.index.max() + 1)
    pivot = pivot.reindex(years_full, fill_value=0)
    total = pivot.sum(axis=1)
    percentages = pd.DataFrame(0, index=pivot.index, columns=pivot.columns)
    mask = total > 0
    percentages.loc[mask] = pivot.loc[mask].div(total[mask], axis=0) * 100
    plt.figure(figsize=(10,6))
    for col in percentages.columns:
        if col == 'fold':
            color = tipping_point_colors['fold bifurcation']
        elif col == 'transcritical':
            color = tipping_point_colors['transcritical bifurcation']
        else:
            color = tipping_point_colors['no bifurcation']
        plt.plot(percentages.index, percentages[col], marker='o', linewidth=2, color=color, label=col)
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, 'type_percentages_plot.pdf'), dpi=300)
    plt.close()

def main():
    bifurcation_csv = 'csv/bifurcation_results_short.csv'
    raw_data_csv = 'data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv'
    img_folder = 'img'
    plot_transition_years(bifurcation_csv, raw_data_csv, img_folder)
    plot_relative_transition_years(bifurcation_csv, raw_data_csv, img_folder)
    plot_type_percentages(bifurcation_csv, img_folder)

if __name__ == "__main__":
    main()
