import os
import pandas as pd
import matplotlib.pyplot as plt
import difflib
from datetime import datetime

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 30, 'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 26, 'axes.titlesize': 30, 'font.size': 26, 'lines.linewidth': 4, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 3
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

def plot_time_series(species, species_data, transition_year, tipping_type, iucn_status, invasive_status, output_dir):
    set_plot_style()
    dates = species_data['TIME'].apply(fyear_to_date)
    abundances = species_data['ABUNDANCE']
    plt.figure(figsize=(10, 6))
    plt.plot(dates, abundances, 'o', markersize=8)
    plt.axvline(x=pd.Timestamp(year=int(transition_year), month=1, day=1), color='red', linestyle='--', linewidth=4)
    plt.xlabel('T')
    plt.ylabel('N')
    first_date, last_date = dates.min(), dates.max()
    first_year, last_year = first_date.year, last_date.year
    plt.xticks([first_date, last_date], [str(first_year), str(last_year)])
    max_abundance = abundances.max()
    plt.yticks([max_abundance], [str(int(max_abundance))])
    plt.tight_layout()
    filename = f"{tipping_type}{iucn_status}{invasive_status}{species.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    config = {
        "biotime_path": os.path.join("data", "biotime_v2_full_2025", "biotime_v2_rawdata_2025.csv"),
        "bifurcation_path": os.path.join("csv", "bifurcation_results_short.csv"),
        "invasive_path": os.path.join("csv", "invasive_species.csv"),
        "iucn_path": os.path.join("csv", "iucn_species.csv"),
        "output_dir": "img"
    }
    target_species = ["Pterostichus oblongopunctatus", "Achlya flavicornis", "Adicella reducta", "Lithostege farinata", "Noctua interposita", "Diachrysia stenochrysis", "Ammodramus leconteii", "Oecanthus"]
#    target_species = [
#        "Abrothrix longipilis", "Acacia celsa", "Abudefduf saxatilis", "Achillea millefolium", "Achlya flavicornis", "Acontia lucida", "Acontia trabealis", "Acronicta aceris", "Acronicta tridens", "Adicella reducta", "Agonum thoreyi", "Ammodramus leconteii", "Antrostomus carolinensis", "Artemisia ludoviciana", "Cenchrus stramineus", "Chrysomus icterocephalus", "Cryptomonas caudata", "Cynanchum viminale", "Diachrysia stenochrysis", "Dipturus batis", "Dryobates nuttallii", "Elaeocarpus bancroftii", "Elassoma evergladei", "Empidonax alnorum", "Enteropogon macrostachyus", "Eragrostis papposa", "Eriocoma hymenoides", "Eriogonum ovalifolium", "Etheostoma vulneratum", "Festuca ovina", "Furnarius rufus", "Gabrius splendidulus", "Gallinula melanops", "Glochidion sessiliflorum", "Gongylidiellum vivum", "Gutenbergia cordifolia", "Hippopotamus amphibius", "Hydraecia nordstroemi", "Lacanobia suasa", "Leiothlypis peregrina", "Lepidorhombus whiffiagonis", "Leptothrium senegalense", "Leucoma salicis", "Leucoraja naevus", "Libellula quadrimaculata", "Lippia javanica", "Lithostege farinata", "Lutjanus apodus", "Macaria brunneata", "Macaria wauaria", "Mamestra brassicae", "Microspathodon crysurus", "Mungos mungo", "Mythimma pudorina", "Noctua interposita", "Notropis lutipinnis", "Oecanthus", "Osteospermum vaillantii", "Packera multilobata", "Padina pavonica", "Panicum maximum", "Paracalanus sp", "Paramonacanthus frenatus", "Parascorpaena mossambica", "Pardosa pullata", "Pasiphila debiliata", "Phacochoerus aethiopicus", "Phyllanthus maderaspatensis", "Phyllotis darwini", "Pipilo erythrophthalmus", "Pirata piraticus", "Plectobranchus evides", "Poecile carolinensis", "Pollichia campestris", "Polyosma alangiacea", "Potamophylax luctuosus", "Pterostichus oblongopunctatus", "Pupalia lappacea", "Ruellia patula", "Sarpa salpa", "Setophaga tigrina", "Solanum campylacanthum", "Spirorbinae", "Steganthera laxiflora", "Stegastes adustus", "Tachinus mimulus", "Thera cognata", "Tringa solitaria", "Vanellus chilensis", "Vireo flavifrons"
#    ]
    os.makedirs(config["output_dir"], exist_ok=True)
    biotime_df = load_biotime_data(config["biotime_path"])
    bifurcation_df = pd.read_csv(config["bifurcation_path"])
    invasive_df = pd.read_csv(config["invasive_path"])
    iucn_df = pd.read_csv(config["iucn_path"])
    bifurcation_species = bifurcation_df['species'].dropna().unique().tolist()
    iucn_dict = pd.Series(iucn_df.iucn_status.values, index=iucn_df.species).to_dict()
    invasive_dict = pd.Series(invasive_df.invasive_status.values, index=invasive_df.species).to_dict()
    for target in target_species:
        matches = difflib.get_close_matches(target, bifurcation_species, n=1, cutoff=0.8)
        if not matches:
            continue
        matched_species = matches[0]
        row = bifurcation_df[bifurcation_df['species'] == matched_species].iloc[0]
        tipping_type = row['tipping_point_type']
        transition_year = row['transition_year']
        iucn_status = iucn_dict.get(matched_species, 'Unknown')
        invasive_status = invasive_dict.get(matched_species, 'Unknown')
        species_data = biotime_df[biotime_df['SPECIES_ID'] == matched_species]
        if species_data.empty:
            continue
        plot_time_series(matched_species, species_data, transition_year, tipping_type, iucn_status, invasive_status, config["output_dir"])

if __name__ == "__main__":
    main()
