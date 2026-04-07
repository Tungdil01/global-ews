import os
import pandas as pd

def remove_duplicates_pandas(input_file, output_file, subset=None, keep='first'):
    try:
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False, 0, 0
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        original_count = len(df)
        print(f"Original rows: {original_count}")
        if 'species' in df.columns and 'taxonomic_group' in df.columns:
            keep_mask = pd.Series(True, index=df.index)
            for species, group in df.groupby('species'):
                if len(group) > 1:
                    groups = set(group['taxonomic_group'])
                    if 'Annelida' in groups and 'Polychaeta' in groups:
                        keep_mask[group[group['taxonomic_group'] == 'Polychaeta'].index] = False
                    if 'Tracheophyta' in groups and (('Rhodophyta' in groups) or ('Onychophora' in groups) or ('Streptophyta' in groups) or ('Chlorophyta' in groups) or ('Anthocerotophyta' in groups)):
                        keep_mask[group[group['taxonomic_group'] == 'Tracheophyta'].index] = False
                    if 'Actinopterygii' in groups and 'Agnatha' in groups:
                        keep_mask[group[group['taxonomic_group'] == 'Actinopterygii'].index] = False
                    if 'Elasmobranchii' in groups and 'Actinopterygii' in groups:
                        keep_mask[group[group['taxonomic_group'] == 'Actinopterygii'].index] = False
                    if 'Actinopterygii' in groups and (('Anguilliformes' in groups) or ('Holocephali' in groups)):
                        keep_mask[group[group['taxonomic_group'] == 'Anguilliformes'].index] = False
                        keep_mask[group[group['taxonomic_group'] == 'Holocephali'].index] = False
                    if 'Copepoda' in groups and 'Maxillopoda' in groups:
                        keep_mask[group[group['taxonomic_group'] == 'Maxillopoda'].index] = False
                    if 'Bryophyta' in groups and (('Bryopsida' in groups) or ('Tracheophyta' in groups)):
                        keep_mask[group[group['taxonomic_group'] == 'Bryopsida'].index] = False
                        keep_mask[group[group['taxonomic_group'] == 'Tracheophyta'].index] = False
                    if 'Ochrophyta' in groups and 'Bacillariophyceae' in groups:
                        keep_mask[group[group['taxonomic_group'] == 'Bacillariophyceae'].index] = False
            df = df[keep_mask]
            dup_species = df['species'].value_counts()
            dup_species = dup_species[dup_species > 1].index
            if len(dup_species) > 0:
                repeated_df = df[df['species'].isin(dup_species)]
                repeated_file = output_file.replace('_cleaned', '_repeated') if '_cleaned' in output_file else output_file + '_repeated'
                repeated_df.to_csv(repeated_file, index=False)
                print(f"Repeated species rows saved to: {repeated_file}")
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        cleaned_count = len(df_clean)
        duplicates_removed = original_count - cleaned_count
        df_clean.to_csv(output_file, index=False)
        print(f"After removing duplicates: {cleaned_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Clean data saved to: {output_file}")
        return True, original_count, cleaned_count
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return False, 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return False, 0, 0

def quick_clean():
    input_file = "csv/bifurcation_results.csv"
    output_file = "csv/bifurcation_results_cleaned.csv"
    print("Performing quick clean with default settings...")
    success, original, cleaned = remove_duplicates_pandas(input_file, output_file)
    if success:
        print(f"Quick clean completed! Removed {original - cleaned} duplicates.")

def main():
    print("CSV Duplicate Remover using Pandas")
    print("=" * 50)
    quick_clean()

if __name__ == "__main__":
    main()
