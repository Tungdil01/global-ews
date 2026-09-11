#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from main_dl import create_summary_figure

DL_FILE = 'csv/bifurcation_results_dl_ews.csv'

def generate_figure_from_dl_without_prob_filter(dl_file_path):
    if not os.path.exists(dl_file_path) or os.path.getsize(dl_file_path) == 0:
        print("DL file missing or empty.")
        return
    df = pd.read_csv(dl_file_path)
    if 'dataset' in df.columns:
        df = df[df['dataset'] != 'GBIF']
    if 'transition_year' in df.columns:
        df = df[df['transition_year'].notna() & (df['transition_year'] != '')]
    if 'taxonomic_group' in df.columns:
        df = df[df['taxonomic_group'] != 'Other']
    print(f"Generating figure from {len(df)} species (non-GBIF, with transition_year, no probability filter).")
    create_summary_figure(df)

def main():
    generate_figure_from_dl_without_prob_filter(DL_FILE)

if __name__ == "__main__":
    main()
