import os, pandas as pd
from main_dl import load_combined_data, EXCLUDED_GROUPS, EXCLUDED_SPECIES_LABELS, clean_taxonomic_groups
from species_status_assessment import SpeciesDataCollector

def preprocess_species_name(species_name):
    if pd.isna(species_name):
        return None
    cleaned = clean_taxonomic_groups(str(species_name).strip())
    if cleaned == 'Other':
        return None
    parts = cleaned.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return cleaned

def load_dl_results(dl_file):
    df = pd.read_csv(dl_file)
    df['species'] = df['species'].apply(preprocess_species_name)
    return df

def load_iucn(iucn_file):
    df = pd.read_csv(iucn_file)
    df['species'] = df['species'].apply(preprocess_species_name)
    df['iucn_status'] = df['iucn_status'].astype(str).fillna('').str.strip().str.upper()
    return df[['species', 'iucn_status']].drop_duplicates(subset=['species']).reset_index(drop=True)

def merge_with_iucn(dl, iucn):
    return pd.merge(dl, iucn, on='species', how='inner')

def filter_threatened(merged, threatened_statuses):
    return merged[merged['iucn_status'].isin(threatened_statuses)].copy()

def format_transition_year(df):
    df = df.copy()
    df['transition_year'] = pd.to_numeric(df['transition_year']).round().astype('Int64')
    return df

def select_columns(df, columns):
    return df[columns].reset_index(drop=True)

def write_output(df, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Wrote {len(df):.3g} threatened segments to {out_file}.")

def filter_positive_probability(df):
    return df[pd.to_numeric(df['best_probability'], errors='coerce') > 0].copy()

def populate_iucn_species(iucn_file='csv/iucn_species.csv'):
    datasets_dict, species_list, data_taxonomy, genus_taxonomy = load_combined_data(min_records=1, prefilter=False, min_time=1900)
    filtered_species = []
    for sp in species_list:
        sp_str = str(sp).strip()
        if sp_str == 'Other' or sp_str in EXCLUDED_SPECIES_LABELS:
            continue
        tax_group = data_taxonomy.get(sp_str, 'Other')
        if tax_group in EXCLUDED_GROUPS:
            continue
        filtered_species.append(sp_str)
    if not filtered_species:
        return
    species_df = pd.DataFrame({'species': filtered_species})
    collector = SpeciesDataCollector()
    collector.build_conservation_database(species_df)

def sync_taxonomic_cache_from_iucn(iucn_file='csv/iucn_species.csv', cache_file='csv/taxonomic_cache.csv'):
    if not os.path.exists(iucn_file) or os.path.getsize(iucn_file) == 0:
        return
    iucn_df = pd.read_csv(iucn_file)
    if 'species' not in iucn_df.columns or 'taxonomic_group' not in iucn_df.columns:
        return
    iucn_df['species'] = iucn_df['species'].astype(str).str.strip()
    iucn_df['taxonomic_group'] = iucn_df['taxonomic_group'].astype(str).str.strip()
    iucn_df = iucn_df[(iucn_df['species'] != '') & (iucn_df['taxonomic_group'] != '') & (iucn_df['taxonomic_group'] != 'nan')]
    if len(iucn_df) == 0:
        return
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache_df = pd.read_csv(cache_file)
        if 'species' not in cache_df.columns or 'taxonomic_group' not in cache_df.columns:
            cache_df = pd.DataFrame(columns=['species', 'taxonomic_group'])
        else:
            cache_df['species'] = cache_df['species'].astype(str).str.strip()
            cache_df['taxonomic_group'] = cache_df['taxonomic_group'].astype(str).str.strip()
    else:
        cache_df = pd.DataFrame(columns=['species', 'taxonomic_group'])
    existing_species = set(cache_df['species'])
    missing_df = iucn_df[~iucn_df['species'].isin(existing_species)][['species', 'taxonomic_group']].drop_duplicates(subset=['species'])
    if len(missing_df) == 0:
        print(f"No missing species to add to {cache_file}.")
        return
    combined = pd.concat([cache_df, missing_df], ignore_index=True).drop_duplicates(subset=['species'], keep='last')
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    combined.to_csv(cache_file, index=False)
    print(f"Added {len(missing_df):.3g} species to {cache_file} from {iucn_file}.")

def main():
    dl_file = 'csv/bifurcation_results_dl_ews.csv'
    iucn_file = 'csv/iucn_species.csv'
    cache_file = 'csv/taxonomic_cache.csv'
    out_file = 'csv/list_threatened.csv'
    threatened_statuses = {'VU', 'CE', 'EN', 'CR', 'VULNERABLE', 'ENDANGERED', 'CRITICALLY_ENDANGERED'}
    output_columns = ['species', 'dataset', 'timeseries_id', 'taxonomic_group', 'iucn_status', 'transition_year', 'tipping_point_type', 'best_probability']
    POPULATE_IUCN = True
    SYNC_TAXONOMIC_CACHE = True
    if POPULATE_IUCN:
        populate_iucn_species(iucn_file)
    if SYNC_TAXONOMIC_CACHE:
        sync_taxonomic_cache_from_iucn(iucn_file, cache_file)
    dl = load_dl_results(dl_file)
    dl = filter_positive_probability(dl)
    iucn = load_iucn(iucn_file)
    merged = merge_with_iucn(dl, iucn)
    threatened = filter_threatened(merged, threatened_statuses)
    threatened = format_transition_year(threatened)
    out = select_columns(threatened, output_columns)
    write_output(out, out_file)

if __name__ == '__main__':
    main()
