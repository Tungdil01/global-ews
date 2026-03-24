import os
import time
import ewstools
import numpy as np
import pandas as pd
import ruptures as rpt
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
SPECIES_TO_PLOT = ['Canthigaster solandri', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
#SPECIES_TO_PLOT = ['Canthigaster solandri', 'Balfourodendron riedelianum']
#SPECIES_TO_PLOT = ['Aglaozonia cutleria', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
#SPECIES_TO_PLOT = ['Anaxyrus sp7', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
#SPECIES_TO_PLOT = ['Anaxyrus americanus', 'Alcyonium digitatum', 'Apostichopus parvimensis']

# Safe division function to handle zero mean case
def safe_divide(data): # Safe division to handle zero mean case
    mean_abs = abs(data).mean()
    if mean_abs == 0:
        return np.zeros_like(data)
    return data / mean_abs

def patched_apply_classifier(self, classifier, tmin, tmax, name="c1", verbose=1):
    input_len = classifier.layers[0].input_shape[1]
    if "residuals" in self.state.columns:
        series = self.state[(self.state.index >= tmin) & (self.state.index < tmax)]["residuals"]
    else:
        series = self.state[(self.state.index >= tmin) & (self.state.index < tmax)]["state"]
    data = series.values
    if len(data) > input_len:
        print("ERROR: Length of time series segment is too long for the classifier.")
        return
    data_norm = safe_divide(data) # Use safe division instead of direct division
    num_zeros = input_len - len(data_norm)
    input_data = np.concatenate((np.zeros(num_zeros), data_norm)).reshape(1, -1, 1)
    dl_pred = classifier.predict(input_data, verbose=verbose)[0]
    dict_dl_pred = {i: val for (i, val) in zip(np.arange(len(dl_pred)), dl_pred)}
    dict_dl_pred["time"] = series.index[-1]
    dict_dl_pred["classifier"] = name
    df_dl_pred = pd.DataFrame(dict_dl_pred, index=[len(self.dl_preds)])
    self.dl_preds = pd.concat([self.dl_preds, df_dl_pred], ignore_index=True)

# Apply patches
import ewstools.core
ewstools.core.normalize_data = safe_divide
ewstools.core.TimeSeries.apply_classifier = patched_apply_classifier

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titlesize': 16,
        'font.size': 12,
        'lines.linewidth': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0
    })

def load_biotime_data(filepath, top_n=100):
    os.makedirs('csv', exist_ok=True)
    species_count_file = 'csv/biotime_species_count.csv'
    top_species_file = f'csv/biotime_top{top_n}.csv'
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
        except:
            df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    taxon_mapping = {} # Load taxon column if available
    if 'TAXON' in df.columns:
        taxon_mapping = df[['VALID_NAME', 'TAXON']].dropna().set_index('VALID_NAME')['TAXON'].to_dict()
        taxon_mapping = {k.lower(): v for k, v in taxon_mapping.items()}
    required = ["YEAR", "ABUNDANCE", "VALID_NAME", "STUDY_ID", "LATITUDE", "LONGITUDE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in BioTIME data: {missing}")
    df = df.dropna(subset=["YEAR", "ABUNDANCE"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    time_val = df["YEAR"].copy()
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        mask_month = df["MONTH"].notna()
        time_val.loc[mask_month] = df.loc[mask_month, "YEAR"] + (df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
        if "DAY" in df.columns:
            df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce")
            mask_day = df["DAY"].notna()
            mask = mask_month & mask_day
            time_val.loc[mask] += (df.loc[mask, "DAY"].astype(int) - 1) / 365.0
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["VALID_NAME"]
    taxon_mapping_file = 'csv/taxon_mapping.csv' # Save taxon mapping for later use
    pd.DataFrame(list(taxon_mapping.items()), columns=['species', 'taxon']).to_csv(taxon_mapping_file, index=False)
    species_count = df['SPECIES_ID'].value_counts().reset_index() # For our purposes, we just need the basic data structure
    species_count.columns = ['species', 'count']
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE"]], species_count

def load_and_filter_data(species_list, threshold=0.6):
    results_file = 'csv/bifurcation_results_short.csv'
#    extinction_file = 'csv/extinction_risk_assessment.csv'
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return None, None
    try: # Load results for target species only
        results_df = pd.read_csv(results_file)
        results_df['tipping_point_type'] = results_df.apply(
            lambda row: row['raw_tipping_point_type'] if row['best_probability'] >= threshold else 'no bifurcation',
            axis=1
        )
        results_df = results_df[results_df['species'].isin(species_list)] # Filter for target species
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None, None
    # Load extinction risk data if available
#    extinction_df = None
#    if os.path.exists(extinction_file):
#        try:
#            extinction_df = pd.read_csv(extinction_file)
#            extinction_df = extinction_df[extinction_df['species'].isin(species_list)]
#        except Exception as e:
#            print(f"Error loading extinction data: {str(e)}")
    return results_df#, extinction_df

def load_species_raw_data(species_name):
    try:
        biotime_path = "data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv"
        if not os.path.exists(biotime_path):
            print(f"BioTIME data not found at {biotime_path}")
            return None
        full_data, _ = load_biotime_data(biotime_path, top_n=100)
        species_data = full_data[full_data['SPECIES_ID'] == species_name] # Filter for the specific species
        if species_data.empty:
            print(f"No data found for species: {species_name}")
            return None
        series = species_data[["TIME", "ABUNDANCE"]].copy()
        series.columns = ["YEAR", "ABUNDANCE"]
        print(f"Loaded {len(series)} data points for {species_name}")
        return series
    except Exception as e:
        print(f"Error loading raw data for {species_name}: {str(e)}")
        return None

def adjust_series_size(series_df, max_size=1500): # Adjust time series to maximum size of 1500 points
    abundance = series_df['ABUNDANCE'].values
    years = series_df['YEAR'].values
    if len(abundance) < max_size:
        padded_abundance = np.zeros(max_size)
        padded_abundance[:len(abundance)] = abundance
        padded_years = np.zeros(max_size)
        padded_years[:len(years)] = years
        if len(years) > 0:
            padded_years[len(years):] = years[-1]
        return pd.DataFrame({'YEAR': padded_years, 'ABUNDANCE': padded_abundance})
    if len(abundance) == max_size:
        return pd.DataFrame({'YEAR': years, 'ABUNDANCE': abundance})
    non_zero_mask = abundance != 0
    non_zero_abundance = abundance[non_zero_mask]
    non_zero_years = years[non_zero_mask]
    if len(non_zero_abundance) <= max_size:
        if len(non_zero_abundance) < max_size:
            padded_abundance = np.zeros(max_size)
            padded_abundance[:len(non_zero_abundance)] = non_zero_abundance
            padded_years = np.zeros(max_size)
            padded_years[:len(non_zero_years)] = non_zero_years
            padded_years[len(non_zero_years):] = non_zero_years[-1] if len(non_zero_years) > 0 else 0
            return pd.DataFrame({'YEAR': padded_years, 'ABUNDANCE': padded_abundance})
        return pd.DataFrame({'YEAR': non_zero_years, 'ABUNDANCE': non_zero_abundance})
    step = len(non_zero_abundance) / max_size
    indices = [int(i * step) for i in range(max_size)]
    uniform_abundance = non_zero_abundance[indices]
    uniform_years = non_zero_years[indices]
    return pd.DataFrame({'YEAR': uniform_years, 'ABUNDANCE': uniform_abundance})

def find_transition(series_df, species):
    points = series_df['ABUNDANCE'].values
    years = series_df['YEAR'].values
    if len(points) < 10:
        transition_index = len(points) // 2
        return transition_index
    methods_results = [] # Try multiple changepoint detection methods
    try: # Method 1: Pelt
        algo = rpt.Pelt(model="rbf", min_size=5, jump=5).fit(points)
        penalty_value = np.log(len(points)) * np.var(points)
        bkps = algo.predict(pen=penalty_value)
        if len(bkps) > 1 and not (len(bkps) == 2 and bkps[0] == 0):
            methods_results.append(("Pelt", bkps))
    except Exception as e:
        print(f"Pelt method failed for {species}: {str(e)}")
    try: # Method 2: Window
        window_size = max(10, min(40, len(points) // 4))
        algo = rpt.Window(width=window_size, model="l2").fit(points)
        n_bkps = min(3, len(points) // 20)
        bkps = algo.predict(n_bkps=n_bkps)
        if len(bkps) > 1 and not (len(bkps) == 2 and bkps[0] == 0):
            methods_results.append(("Window", bkps))
    except Exception as e:
        print(f"Window method failed for {species}: {str(e)}")
    try: # Method 3: Binary Segmentation
        algo = rpt.Binseg(model="l2", min_size=5).fit(points)
        n_bkps = min(3, len(points) // 20)
        bkps = algo.predict(n_bkps=n_bkps)
        if len(bkps) > 1 and not (len(bkps) == 2 and bkps[0] == 0):
            methods_results.append(("BinSeg", bkps))
    except Exception as e:
        print(f"BinSeg method failed for {species}: {str(e)}")
    selected_bkps = None
    selected_method = "Fallback"
    for method, bkps in methods_results: # Prefer methods with exactly 2 breakpoints (1 changepoint + end)
        if len(bkps) == 2:
            selected_bkps = bkps
            selected_method = method
            break
    if selected_bkps is None and methods_results: # If no ideal method found, use the one with fewest changepoints
        methods_results.sort(key=lambda x: len(x[1]))
        selected_method, selected_bkps = methods_results[0]
    # Fallback to midpoint if no valid changepoints
    if selected_bkps is None or len(selected_bkps) <= 1:
        transition_index = len(points) // 2
        selected_method = "MidpointFallback"
    else:
        # Exclude boundary points and the final index
        boundary_buffer = max(5, len(points) // 5)
        valid_bkps = [
            b for b in selected_bkps 
            if boundary_buffer <= b <= (len(points) - boundary_buffer)
            and b != len(points)
        ]
        if valid_bkps:
            # Select changepoint closest to median position
            median_pos = len(points) // 2
            transition_index = min(valid_bkps, key=lambda x: abs(x - median_pos))
        else:
            transition_index = len(points) // 2
            selected_method = "MidpointFallback"
    print(f"Transition found for {species} using {selected_method} at index {transition_index}")
    return transition_index

def robust_detrending(ts, series_length, method_preference='Lowess'):
    methods_to_try = []
    if method_preference == 'Lowess':
        methods_to_try = [
            ('Lowess', {'span': min(0.3, max(0.1, 50/series_length))}),
            ('Lowess', {'span': 0.2}),
            ('Gaussian', {'bandwidth': min(0.3, max(0.1, 50/series_length))}),
            ('Gaussian', {'bandwidth': 0.2}),
            ('Linear', {}),
        ]
    else:
        methods_to_try = [
            ('Gaussian', {'bandwidth': min(0.3, max(0.1, 50/series_length))}),
            ('Gaussian', {'bandwidth': 0.2}),
            ('Lowess', {'span': min(0.3, max(0.1, 50/series_length))}),
            ('Linear', {}),
        ]
    last_error = None
    for method, params in methods_to_try:
        try:
            if method == 'Lowess':
                ts.detrend(method='Lowess', span=params['span'])
            elif method == 'Gaussian':
                ts.detrend(method='Gaussian', bandwidth=params['bandwidth'])
            elif method == 'Linear':
                ts.detrend(method='Linear')
            if 'smoothing' in ts.state.columns: # Check for overfitting
                smoothing_vals = ts.state['smoothing'].values
                original_vals = ts.state['state'].values
                valid_mask = ~np.isnan(smoothing_vals)
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(original_vals[valid_mask], smoothing_vals[valid_mask])[0,1]
                    if abs(corr) < 0.95:
                        print(f"Successfully detrended using {method} with params {params}")
                        return True
                    else:
                        raise ValueError("Potential overfitting detected")
            print(f"Successfully detrended using {method} with params {params}")
            return True
        except Exception as e:
            last_error = e
            continue
    try: # If all methods fail, use linear as last resort
        ts.detrend(method='Linear')
        print("Used linear detrending as fallback")
        return True
    except Exception as e:
        print(f"All detrending methods failed: {last_error}")
        return False

def load_classifiers():
    classifier_paths_500 = []
    classifier_paths_1500 = []
    root_path_500 = '../saved_classifiers/bury_pnas_21/len500/' # Paths for 500-length classifiers
    if os.path.exists(root_path_500):
        classifier_names_500 = [name for name in os.listdir(root_path_500) if name.endswith('.pkl') or name.endswith('.h5')]
        classifier_paths_500 = [os.path.join(root_path_500, name) for name in classifier_names_500[:5]]
    root_path_1500 = '../saved_classifiers/bury_pnas_21/len1500/' # Paths for 1500-length classifiers
    if os.path.exists(root_path_1500):
        classifier_names_1500 = [name for name in os.listdir(root_path_1500) if name.endswith('.pkl') or name.endswith('.h5')]
        classifier_paths_1500 = [os.path.join(root_path_1500, name) for name in classifier_names_1500[:5]]
    return classifier_paths_500, classifier_paths_1500

def plot_dl_predictions_pdf(species_name, results_df):
    # Create dummy plot to initialize MathJax
    dummy_file = "dummy_plot.pdf"
    fig_dummy = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig_dummy.write_image(dummy_file, format="pdf")
    time.sleep(0.1)
    dummy_path = Path(dummy_file)
    dummy_path.unlink(missing_ok=True)
    set_plot_style()
    os.makedirs('img', exist_ok=True)
    raw_series = load_species_raw_data(species_name)
    if raw_series is None:
        print(f"Cannot create DL predictions for {species_name}: no raw data available")
        return
    print(f"Processing {species_name} for DL predictions...")
    try:
        original_length = len(raw_series)
        if original_length <= 500:
            series = adjust_series_size(raw_series, max_size=500)
            max_size = 500
        else:
            series = adjust_series_size(raw_series, max_size=1500)
            max_size = 1500
        transition_idx = find_transition(series, species_name)
        ts = ewstools.TimeSeries(series['ABUNDANCE'].values, transition=transition_idx)
        ts.time_values = series['YEAR'].values
        ts.state['year'] = series['YEAR'].values
        detrend_success = robust_detrending(ts, original_length, method_preference='Lowess')
        if not detrend_success:
            print(f"Detrending failed for {species_name}, using fallback approach")
            ts.detrend(method='Linear')
        eval_index = int(0.9 * transition_idx) if transition_idx > 0 else 0
        classifier_paths_500, classifier_paths_1500 = load_classifiers()
        if max_size == 500:
            classifier_paths = classifier_paths_500
        else:
            classifier_paths = classifier_paths_1500
        ts.clear_dl_preds()
        for classifier_path in classifier_paths:
            if os.path.exists(classifier_path):
                try:
                    classifier = load_model(classifier_path)
                    ts.apply_classifier_inc(classifier, inc=10, verbose=0, name=os.path.basename(classifier_path))
                    tf.keras.backend.clear_session()
                except Exception as e:
                    print(f"Error applying classifier {classifier_path}: {str(e)}")
        fig = ts.make_plotly(ens_avg=True, eval_index=eval_index)
        fig.add_vline(x=eval_index, line_dash="dash", line_color="grey")
        fig.update_layout(
            font=dict(family="Arial, sans-serif"), title_font=dict(family="Arial, sans-serif")
        )
        filename = f"img/{species_name.replace(' ', '_')}_dl_predictions.pdf"
        fig.write_image(filename)
        print(f"Created actual DL predictions PDF for {species_name}")
    except Exception as e:
        print(f"Error creating DL predictions for {species_name}: {str(e)}")

def plot_species_summary(target_species): # , extinction_df
    set_plot_style()
    os.makedirs('img', exist_ok=True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    # Plot 1: Tipping point types
    if not target_species.empty and 'tipping_point_type' in target_species.columns:
        tipping_counts = target_species['tipping_point_type'].value_counts()
        if not tipping_counts.empty:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            bars = ax1.bar(tipping_counts.index, tipping_counts.values, 
                          color=colors[:len(tipping_counts)], alpha=0.7)
            for bar, count in zip(bars, tipping_counts.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
#    # Plot 2: Extinction risk distribution
#    if extinction_df is not None and not extinction_df.empty and 'extinction_risk' in extinction_df.columns:
#        risk_counts = extinction_df['extinction_risk'].value_counts()
#        if not risk_counts.empty:
#            # Define colors for risk levels
#            risk_colors = {
#                'very_low': 'green',
#                'low': 'lightgreen', 
#                'medium': 'yellow',
#                'high': 'orange',
#                'very_high': 'red',
#                'unknown': 'gray'
#            }
#            colors = [risk_colors.get(risk, 'blue') for risk in risk_counts.index]
#            bars = ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
#            # Add counts on bars
#            for bar, count in zip(bars, risk_counts.values):
#                height = bar.get_height()
#                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                        f'{int(count)}', ha='center', va='bottom', fontsize=12)
#        ax2.set_ylabel('Count', fontsize=12)
#        ax2.tick_params(axis='x', rotation=45)
    # Plot 2: Current vs Maximum Abundance (scatter plot)
    if not target_species.empty and 'current_abundance' in target_species.columns and 'max_abundance' in target_species.columns:
        # Filter out zero or invalid values
        valid_data = target_species[
            (target_species['current_abundance'] > 0) & 
            (target_species['max_abundance'] > 0)
        ]
        if not valid_data.empty:
            valid_data = valid_data.copy() # Calculate decline percentage
            valid_data['decline_pct'] = (1 - valid_data['current_abundance'] / valid_data['max_abundance']) * 100
            scatter = ax3.scatter(valid_data['current_abundance'], valid_data['max_abundance'], c=valid_data['decline_pct'], cmap='RdYlGn_r', s=100, alpha=0.7)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Population Decline (%)', fontsize=10)
            max_val = max(valid_data[['current_abundance', 'max_abundance']].max())
            ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No decline')
            ax3.set_xlabel('Current Abundance', fontsize=12)
            ax3.set_ylabel('Maximum Abundance', fontsize=12)
            ax3.legend()
#    # Plot 4: Risk factors summary (if available in extinction data)
#    if extinction_df is not None and not extinction_df.empty and 'risk_factors' in extinction_df.columns:
#        # Count risk factors across all species
#        all_factors = []
#        for factors_str in extinction_df['risk_factors'].dropna():
#            try:
#                if isinstance(factors_str, str) and factors_str.startswith('['):
#                    factors = eval(factors_str)
#                else:
#                    factors = [factors_str]
#                all_factors.extend(factors)
#            except:
#                continue
#        if all_factors:
#            factor_counts = pd.Series(all_factors).value_counts().head(8)
#            if not factor_counts.empty:
#                bars = ax4.barh(range(len(factor_counts)), factor_counts.values, 
#                               color='lightcoral', alpha=0.7)
#                ax4.set_yticks(range(len(factor_counts)))
#                ax4.set_yticklabels(factor_counts.index, fontsize=10)
#                ax4.set_xlabel('Frequency', fontsize=12)
#                # Add counts on bars
#                for i, (bar, count) in enumerate(zip(bars, factor_counts.values)):
#                    width = bar.get_width()
#                    ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
#                            f'{int(count)}', ha='left', va='center', fontsize=10)
#    # Set consistent style for all subplots
#    for ax in [ax1, ax2, ax3, ax4]:
#        for spine in ax.spines.values():
#            spine.set_edgecolor('black')
#            spine.set_linewidth(1.0)
#    plt.tight_layout()
#    plt.savefig('img/species_risk_assessment.pdf', dpi=300, bbox_inches='tight')
#    plt.close()
#    print(f"Created risk assessment plot for {len(target_species)} species")

def print_species_details(target_species): # , extinction_df
    print("\n" + "="*80)
    print("SPECIES DETAILS")
    print("="*80)
    for _, row in target_species.iterrows():
        species_name = row['species']
        print(f"\nSpecies: {species_name}")
        print(f"  Taxonomic Group: {row.get('taxonomic_group', 'Unknown')}")
        print(f"  Tipping Point Type: {row.get('tipping_point_type', 'Unknown')}")
        print(f"  Transition Year: {row.get('transition_year', 'Unknown')}")
#        # Get extinction risk info
#        extinction_info = ""
#        if extinction_df is not None:
#            species_risk = extinction_df[extinction_df['species'] == species_name]
#            if not species_risk.empty:
#                risk_row = species_risk.iloc[0]
#                extinction_info = f"{risk_row.get('extinction_risk', 'Unknown')}"
#                risk_factors = risk_row.get('risk_factors', '')
#                if risk_factors and str(risk_factors) != 'nan':
#                    extinction_info += f" - Factors: {risk_factors}"
#        print(f"  Extinction Risk: {extinction_info if extinction_info else 'Unknown'}")
        # Abundance information
        current_abundance = row.get('current_abundance', 0)
        max_abundance = row.get('max_abundance', 0)
        if max_abundance > 0:
            decline_pct = (1 - current_abundance / max_abundance) * 100
            print(f"  Population Status: {current_abundance:.0f} (current) / {max_abundance:.0f} (max) - {decline_pct:.1f}% decline")
        print("-" * 40)

def main():
    threshold=0.4
    print("Starting PDF plot generation for target species...")
    print(f"Species to plot: {SPECIES_TO_PLOT}")
    target_species = load_and_filter_data(SPECIES_TO_PLOT, threshold) # , extinction_df
    if target_species is None or target_species.empty:
        print("No data found for the specified species")
        return
    print("\nCreating individual DL prediction PDFs using EWS analysis...")
    for species in SPECIES_TO_PLOT:
        if species in target_species['species'].values:
            plot_dl_predictions_pdf(species, target_species)
        else:
            print(f"Warning: {species} not found in results data")
    plot_species_summary(target_species) # , extinction_df
    print_species_details(target_species) # , extinction_df
    print("\nPlot generation complete!")
    print("Check the 'img' folder for the output PDF files:")
    print("  - Individual species: {species_name}_dl_predictions.pdf")
#    print("  - species_risk_assessment.pdf: Comprehensive risk analysis")

if __name__ == "__main__":
    main()
