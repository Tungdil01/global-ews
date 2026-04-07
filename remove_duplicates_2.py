import os
import pandas as pd

def remove_duplicates_pandas(input_file, output_file, subset=None, keep='first'):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False, 0, 0
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    original_count = len(df)
    print(f"Original rows: {original_count}")
    required_cols = ['species', 'taxonomic_group']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found.")
        return False, original_count, 0
    df['priority'] = df['taxonomic_group'] != 'Other'
    df_sorted = df.sort_values(['species', 'priority'], ascending=[True, False])
    df_clean = df_sorted.drop_duplicates(subset='species', keep='first').drop(columns='priority').reset_index(drop=True)
    cleaned_count = len(df_clean)
    duplicates_removed = original_count - cleaned_count
    df_clean.to_csv(output_file, index=False)
    print(f"After removing duplicates: {cleaned_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Clean data saved to: {output_file}")
    return True, original_count, cleaned_count

def main():
    print("CSV Duplicate Remover using Pandas")
    print("=" * 50)
    input_file = "csv/bifurcation_results_cleaned.csv"
    output_file = input_file
    success, original, cleaned = remove_duplicates_pandas(input_file, output_file)

if __name__ == "__main__":
    main()
