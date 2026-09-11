import pandas as pd

def main():
    old_path = "csv/taxonomic_cache.csv"
    update_path = "csv/taxonomic_cache_update.csv"
    old_df = pd.read_csv(old_path)
    update_df = pd.read_csv(update_path)
    merged = old_df.merge(update_df, on="species", how="left", suffixes=("_old", "_new"))
    merged["taxonomic_group"] = merged["taxonomic_group_new"].combine_first(merged["taxonomic_group_old"])
    result = merged[["species", "taxonomic_group"]]
    result.to_csv(old_path, index=False)

if __name__ == "__main__":
    main()
