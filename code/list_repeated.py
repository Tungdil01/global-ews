import os
import pandas as pd
from main_dl import load_combined_data

def load_non_gbif_data():
    datasets_dict, _, _, _ = load_combined_data(min_records=1, force_reload=False, prefilter=False, min_time=0, whitelist={})
    if not datasets_dict:
        return pd.DataFrame()
    frames = []
    for ds_name, df in datasets_dict.items():
        if str(ds_name).strip().upper() == 'GBIF':
            continue
        if len(df) == 0:
            continue
        frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def normalize_dataset_column(df):
    if 'DATASET' in df.columns:
        df = df.rename(columns={'DATASET': 'dataset'})
    elif 'dataset' not in df.columns:
        for col in df.columns:
            if col.upper() == 'DATASET':
                df = df.rename(columns={col: 'dataset'})
                break
    return df

def clean_data(df):
    return df.dropna(subset=['SPECIES_ID', 'dataset'])

def find_repeated_species(df):
    counts = df.groupby('SPECIES_ID')['dataset'].nunique()
    return counts[counts > 1].index.tolist()

def build_repeated_report(df, repeated_species):
    rows = []
    for sp in repeated_species:
        datasets = sorted(df[df['SPECIES_ID'] == sp]['dataset'].unique())
        rows.append({'species': sp, 'num_datasets': len(datasets), 'datasets': ', '.join(datasets)})
    return pd.DataFrame(rows).sort_values('num_datasets', ascending=False)

def save_report(df, output_path='csv/repeated_species.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def print_report(df):
    print(f"Found {len(df)} species repeated across non-GBIF datasets.")
    print(df.to_string(index=False))

def main():
    full_data = load_non_gbif_data()
    if full_data is None or full_data.empty:
        print("No data loaded. Exiting.")
        return
    full_data = normalize_dataset_column(full_data)
    full_data = clean_data(full_data)
    repeated_species = find_repeated_species(full_data)
    report = build_repeated_report(full_data, repeated_species)
    print_report(report)
    save_report(report)

if __name__ == "__main__":
    main()
