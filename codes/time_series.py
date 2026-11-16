import os, difflib
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 32, 'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 30, 'axes.titlesize': 34, 'font.size': 30, 'lines.linewidth': 4, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 3
    })

def load_biotime_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
        except:
            df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    required = ["YEAR", "ABUNDANCE", "VALID_NAME"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in BioTIME data: {missing}")
    df = df.dropna(subset=["YEAR", "ABUNDANCE"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    time_val = df["YEAR"].astype(float).copy()
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        mask_month = df["MONTH"].notna()
        time_val.loc[mask_month] = df.loc[mask_month, "YEAR"].astype(float) + (pd.to_numeric(df.loc[mask_month, "MONTH"], errors="coerce").astype(int) - 1) / 12.0
        if "DAY" in df.columns:
            df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce")
            mask_day = df["DAY"].notna()
            mask = mask_month & mask_day
            time_val.loc[mask] += (pd.to_numeric(df.loc[mask, "DAY"], errors="coerce").astype(int) - 1) / 365.0
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["VALID_NAME"]
    return df.groupby(["SPECIES_ID", "TIME"], as_index=False).agg({"ABUNDANCE": "sum"})

def fyear_to_date(fyear):
    year = int(fyear)
    frac = fyear - year
    day_of_year = int(round(frac * 365)) + 1
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year-1)

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

def plot_time_series(species, species_data, transition_year, tipping_type, iucn_status, invasive_status, output_dir):
    set_plot_style()
    dates = species_data['TIME'].apply(fyear_to_date)
    abundances = species_data['ABUNDANCE']
    plt.figure(figsize=(8, 6))
    plt.plot(dates, abundances, 'o', markersize=8)
    if pd.notna(transition_year):
        plt.axvline(x=pd.Timestamp(year=int(transition_year), month=1, day=1), color='red', linestyle='--', linewidth=4)
    plt.xlabel('Years')
    plt.ylabel('Population size')
    plt.gca().yaxis.set_label_coords(-0.08, 0.45)
    first_date, last_date = dates.min(), dates.max()
    first_year, last_year = first_date.year, last_date.year
    plt.xticks([first_date, last_date], [str(first_year), str(last_year)])
    max_abundance = abundances.max()
    plt.yticks([max_abundance], [str(int(max_abundance))])
    plt.tight_layout()
    tipping_str = abbreviate_bifurcation_type(str(tipping_type)) if pd.notna(tipping_type) else ''
    iucn_str = str(iucn_status) if pd.notna(iucn_status) else ''
    invasive_str = str(invasive_status) if pd.notna(invasive_status) else ''
    filename = f"{tipping_str}{iucn_str}{invasive_str}{species.replace(' ', '_')}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def match_species_in_biotime(species, biotime_species_list):
    matches = difflib.get_close_matches(species, biotime_species_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

def score_change_points(bkps, signal, method):
    if len(bkps) == 0:
        return -np.inf
    segments = np.split(signal, bkps)
    residuals = [seg - np.mean(seg) for seg in segments]
    sse = sum(np.sum(r**2) for r in residuals)
    return -sse

def find_transition(times, abundances):
    if len(abundances) < 10:
        return np.nan
    points = abundances.copy()
    methods_results = []
    if np.std(points) < 1e-10:
        return times[len(times) // 2]
    mean_val = np.mean(points)
    if not isinstance(mean_val, (int, float, np.number)):
        try:
            mean_val = float(mean_val)
        except:
            return np.nan
    if mean_val > 0:
        relative_range = (np.max(points) - np.min(points)) / mean_val
        if relative_range < 0.01:
            return times[len(times) // 2]
    best_transition_time = np.nan
    best_score = -np.inf
    best_method = None
    # Method 1: Pelt with various parameters
    pelt_models = ["rbf", "l2", "linear", "normal", "ar"]
    penalties = [1, 2, 3, 5, 10]
    for model in pelt_models:
        for pen in penalties:
            try:
                algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(points)
                penalty_value = pen * np.log(len(points)) * np.var(points)
                bkps = algo.predict(pen=penalty_value)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, "Pelt")
                if score > best_score and len(valid_bkps) == 1:
                    best_score = score
                    best_transition_time = times[valid_bkps[0]]
                    best_method = f"Pelt_{model}_pen{pen}"
            except Exception:
                continue
    # Method 2: Window with various parameters
    window_sizes = [20, 30, 40, 50, 60]
    n_bkps_range = [1, 2, 3, 4, 5]
    for window_size in window_sizes:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Window(width=window_size, model="l2").fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, "Window")
                if score > best_score and len(valid_bkps) == 1:
                    best_score = score
                    best_transition_time = times[valid_bkps[0]]
                    best_method = f"Window_width{window_size}_nbps{n_bkps}"
            except Exception:
                continue
    # Method 3: Binary Segmentation with various parameters
    binseg_models = ["l2", "rbf", "linear", "normal", "ar"]
    for model in binseg_models:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Binseg(model=model, min_size=5).fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, "BinSeg")
                if score > best_score and len(valid_bkps) == 1:
                    best_score = score
                    best_transition_time = times[valid_bkps[0]]
                    best_method = f"BinSeg_{model}_nbps{n_bkps}"
            except Exception:
                continue
    return best_transition_time

def main():
    config = {
        "biotime_path": os.path.join("data", "biotime_v2_full_2025", "biotime_v2_rawdata_2025.csv"),
        "bifurcation_path": os.path.join("csv", "bifurcation_results_short.csv"),
        "invasive_path": os.path.join("csv", "invasive_species.csv"),
        "iucn_path": os.path.join("csv", "iucn_species.csv"),
        "output_dir": "img"
    }
    target_species = ["Empidonax alnorum", "Gallinula melanops", "Notropis lutipinnis", "Setophaga tigrina"]
    os.makedirs(config["output_dir"], exist_ok=True)
    biotime_df = load_biotime_data(config["biotime_path"])
    bifurcation_df = pd.read_csv(config["bifurcation_path"])
    invasive_df = pd.read_csv(config["invasive_path"])
    iucn_df = pd.read_csv(config["iucn_path"])
    bifurcation_species = bifurcation_df['species'].dropna().unique().tolist()
    biotime_species_list = biotime_df['SPECIES_ID'].unique().tolist()
    iucn_dict = pd.Series(iucn_df.iucn_status.values, index=iucn_df.species).to_dict()
    invasive_dict = pd.Series(invasive_df.invasive_status.values, index=invasive_df.species).to_dict()
    for target in target_species:
        # find in bifurcation data
        matches = difflib.get_close_matches(target, bifurcation_species, n=1, cutoff=0.8)
        if matches:
            matched_species = matches[0]
            row = bifurcation_df[bifurcation_df['species'] == matched_species].iloc[0]
            tipping_type = row['tipping_point_type'] if pd.notna(row['tipping_point_type']) else ''
            transition_year = row['transition_year'] if pd.notna(row['transition_year']) else None
        else:
            matched_species = target
            tipping_type = ''
            transition_year = None
            print(f"No bifurcation data for {target}, will compute changepoint")
        iucn_status = iucn_dict.get(matched_species, 'Unknown')
        if pd.isna(iucn_status):
            iucn_status = ''
        invasive_status = invasive_dict.get(matched_species, 'Unknown')
        if pd.isna(invasive_status):
            invasive_status = ''
        biotime_match = match_species_in_biotime(target, biotime_species_list)
        if biotime_match is None:
            print(f"Skipping {target}: no close match in BioTIME data")
            continue
        species_data = biotime_df[biotime_df['SPECIES_ID'] == biotime_match]
        if species_data.empty:
            print(f"Skipping {target}: no data after aggregation")
            continue
        if transition_year is None or pd.isna(transition_year):
            times = species_data['TIME'].values
            abundances = species_data['ABUNDANCE'].values
            computed_year = find_transition(times, abundances)
            if not np.isnan(computed_year):
                transition_year = computed_year
                if tipping_type == '':
                    tipping_type = 'Changepoint'
        plot_time_series(matched_species, species_data, transition_year, tipping_type, iucn_status, invasive_status, config["output_dir"])
        print(f"Plotted {matched_species} (from target {target})")

if __name__ == "__main__":
    main()
