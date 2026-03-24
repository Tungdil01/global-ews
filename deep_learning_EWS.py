import os
import ewstools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from tensorflow.keras.models import load_model

# Set seed for reproducibility
np.random.seed(0)

def count_species_points(df, output_path='csv/biotime_species_count.csv'):
    # Filter out zero abundance values
    non_zero_df = df[df['ABUNDANCE'] != 0]
    # Group by species and taxonomic group, count non-zero points
    species_counts = (non_zero_df.groupby(['species', 'taxon'])
                      .size()
                      .reset_index(name='count')
                      .sort_values('count', ascending=False))
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save results
    species_counts.to_csv(output_path, index=False)
    return species_counts

def get_top_species(df, top_n=10):
    counts_path = 'csv/biotime_species_count.csv'
    # Generate or read species counts
    if not os.path.exists(counts_path):
        species_counts = count_species_points(df, counts_path)
    else:
        species_counts = pd.read_csv(counts_path)
    return species_counts['species'].head(top_n).tolist()

def adjust_series_size(series_df, max_size=1500):
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

def find_transition(series_df, species, variable="ABUNDANCE"):
    os.makedirs('img', exist_ok=True)
    points = series_df[variable].values
    # Change point detection
    model = "l2"
    algo = rpt.Window(width=40, model=model).fit(points)
    my_bkps = algo.predict(n_bkps=5)
    # Plot results
    rpt.show.display(points, my_bkps, figsize=(6.4, 4))
    plt.title(f'Window-Based Search Method for {species}')
    plt.xlabel('Year')
    plt.ylabel(variable)
    plt.tight_layout()
    plt.savefig(f'img/{species}_changepoint_Window_Based.png')
    plt.close()
    # Get transition point
    transition_index = my_bkps[-2] if len(my_bkps) > 1 else len(points) // 2
    transition_year = series_df['YEAR'].iloc[transition_index]
    print(f"Transition Value (year) for {species}: {transition_year}")
    return transition_index

def save_results(species_name, taxonomic_group, bifurcation_type, transition_year):
    os.makedirs('csv', exist_ok=True)
    results_file = 'csv/bifurcation_results.csv'
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['species', 'taxonomic_group', 'bifurcation_type', 'transition_year'])
    new_result = pd.DataFrame({
        'species': [species_name],
        'taxonomic_group': [taxonomic_group],
        'bifurcation_type': [bifurcation_type],
        'transition_year': [transition_year]
    })
    results_df = pd.concat([results_df, new_result], ignore_index=True)
    results_df.to_csv(results_file, index=False)
    print(f"Results saved for {species_name}")

def process_species(species_name, df, classifier, bifurcation_map):
    series = df[df["valid_name"] == species_name][["YEAR", "ABUNDANCE", "taxon"]]
    series = adjust_series_size(series)
    transition_idx = find_transition(series, species_name.split()[0])
    ts = ewstools.TimeSeries(series['ABUNDANCE'].values, transition=transition_idx)
    ts.state['year'] = series['YEAR'].values
    # Handle detrending with possible length mismatch
    try:
        ts.detrend(method='Lowess', span=0.2)
    except ValueError as e:
        if "Length of values" in str(e) and "does not match length of index" in str(e):
            # Fallback to Gaussian detrending if Lowess fails
            ts.detrend(method='Gaussian', bandwidth=0.2)
        else:
            raise e
    # Calculate evaluation time as 75% of total time points
    t_total = len(series)
    t_eval = int(0.75 * t_total)
    ts.apply_classifier(classifier, tmin=0, tmax=t_eval)
    # Calculate mean probabilities and determine bifurcation type
    mean_probs = ts.dl_preds[[0, 1, 2, 3]].mean()
    pred_class = mean_probs.idxmax()
    bifurcation_type = bifurcation_map.get(pred_class, 'unknown')
    taxonomic_group = df[df["valid_name"] == species_name]["taxon"].iloc[0]
    save_results(species_name, taxonomic_group, bifurcation_type, series['YEAR'].iloc[transition_idx])
    ts.clear_dl_preds()
    return ts

def run_analysis_pipeline(df, top_n=10):
    # Get top species
    top_species = get_top_species(df, top_n)
    # Load classifier
    classifier_path = '../saved_classifiers/bury_pnas_21/len1500/best_model_1_1_len1500.pkl'
    classifier = load_model(classifier_path)
    # Define bifurcation map
    bifurcation_map = {
        0: 'fold bifurcation',
        1: 'Hopf bifurcation',
        2: 'transcritical bifurcation',
        3: 'null'
    }
    # Process each species
    for species in top_species:
        print(f"Processing {species}...")
        process_species(species, df, classifier, bifurcation_map)

def main():
    path = 'data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv'
    df = pd.read_csv(path)
    run_analysis_pipeline(df, top_n=10)

if __name__ == "__main__":
    main()
