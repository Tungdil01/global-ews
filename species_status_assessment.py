import os
import re
import time
import random
import csv
import requests
import warnings
import pandas as pd
import urllib.parse
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
warnings.filterwarnings('ignore')

class SpeciesDataCollector:
    def __init__(self):
        try:
            dotenv_available = True
        except Exception:
            dotenv_available = False
        loaded_from = None
        if loaded_from is None and Path(".env").is_file() is False and Path(".env").is_dir():
            token_path = Path(".env") / "token.txt"
            if token_path.exists():
                try:
                    with open(token_path, "r", encoding="utf-8") as fh:
                        content = fh.read().strip()
                        if content:
                            os.environ["IUCN_TOKEN"] = content.strip().strip('"').strip("'")
                            loaded_from = ".env/token.txt"
                except Exception:
                    pass
        self.session = self._create_session()
        self.iucn_base_url = "https://api.iucnredlist.org/api/v4"
        self.iucn_token = os.getenv("IUCN_TOKEN", "").strip()
        if not self.iucn_token:
            print("Warning: IUCN token not found. Place a line in token.txt or .env exactly like: IUCN_TOKEN=\"your_token_here\" or put the raw token in .env/token.txt")
        else:
            if loaded_from:
                print(f"IUCN token loaded from {loaded_from} (value not displayed).")
            else:
                print("IUCN token loaded from environment (value not displayed).")
        # We will not load combined databases from CSVs anymore.
        self.invasive_species_df = None
        self.iucn_species_df = None
        # New attribute for invasive index
        self.invasive_index = None

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(['GET','POST','PUT','DELETE','HEAD','OPTIONS']))
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def generate_name_variants(name):
        """
        Generate a list of possible cleaned scientific names from the input.
        Handles parentheses, var., subsp., subg., Candidatus, complex, etc.
        Returns a set of unique candidate strings.
        """
        if pd.isnull(name) or not isinstance(name, str):
            return set()
        original = name.strip()
        candidates = {original}
        # 1) Remove parenthetical content entirely (variant A)
        no_paren = re.sub(r'\([^)]*\)', '', original).strip()
        no_paren = re.sub(r'\s+', ' ', no_paren)
        if no_paren and no_paren != original:
            candidates.add(no_paren)
        # 2) If parentheses present, try using the inside as genus (variant B)
        paren_match = re.search(r'\(([^)]+)\)', original)
        if paren_match:
            inside = paren_match.group(1).strip()
            # Replace the first word with inside, keep rest after closing parenthesis
            # Example: "Acartia (Acanthacartia) tonsa" -> "Acanthacartia tonsa"
            parts = re.split(r'\([^)]*\)', original)
            if len(parts) == 2:
                before = parts[0].strip()
                after = parts[1].strip()
                if before and after:
                    # take first word of before? Actually we want to replace the genus
                    before_words = before.split()
                    if before_words:
                        # new name: inside + rest of before (without first word?) + after
                        # Simpler: just inside + " " + after
                        candidate = f"{inside} {after}".strip()
                        candidates.add(candidate)
        # 3) Remove var., subsp., subg., f., etc. and everything after
        for pattern in [r'\s+var\.\s+.*', r'\s+subsp\.\s+.*', r'\s+ssp\.\s+.*', r'\s+subg\.\s+.*', r'\s+f\.\s+.*', r'\s+complex\s*$']:
            cleaned = re.sub(pattern, '', original, flags=re.IGNORECASE).strip()
            if cleaned and cleaned != original:
                candidates.add(cleaned)
        # 4) Handle "subg." specially: take everything after "subg." and treat as name
        if 'subg.' in original.lower():
            # find index of "subg." and take substring after it
            idx = original.lower().find('subg.')
            if idx != -1:
                after = original[idx + 5:].strip()
                if after:
                    candidates.add(after)
        # 5) Handle "Candidatus": remove it if present
        if 'candidatus' in original.lower():
            without_cand = re.sub(r'Candidatus\s+', '', original, flags=re.IGNORECASE).strip()
            candidates.add(without_cand)
        # 6) Remove trailing numbers, underscores, etc. (like clean_taxonomic_groups)
        cleaned = re.sub(r'[\d_]+$', '', original).strip()
        if cleaned and cleaned != original:
            candidates.add(cleaned)
        # Remove any empty strings
        candidates = {c for c in candidates if c}
        return candidates

    def load_invasive_sources(self):
        """Load all invasive source files and build an index mapping cleaned name variants to source info."""
        print("Building invasive species index from source files...")
        index = {}  # key: cleaned variant (string), value: (source_name, classification)
        # GIATAR file
        giatar_file = "data/GIATAR/dataset/species lists/invasive_all_source.csv"
        if os.path.exists(giatar_file):
            try:
                df = pd.read_csv(giatar_file)
                # columns to search
                cols = ['taxonCABI', 'taxonSINAS', 'taxonEPPO', 'taxonDAISIE', 'scientificName', 'canonicalName', 'taxonomic_species', 'gbif_species']
                for col in cols:
                    if col in df.columns:
                        for val in df[col].dropna().astype(str):
                            variants = self.generate_name_variants(val)
                            for v in variants:
                                if v not in index:
                                    index[v] = ('GIATAR', 'Invasive (GIATAR)')
            except Exception as e:
                print(f"Error processing GIATAR file: {e}")
        # GISD+EICAT exports
        eicat_files = ["data/invasive/GISD+EICAT+export_1.xlsx","data/invasive/GISD+EICAT+export_2.xlsx","data/invasive/GISD+EICAT+export_3.xlsx","data/invasive/GISD+EICAT+export_4.xlsx","data/invasive/GISD+EICAT+export_5.xlsx"]
        for f in eicat_files:
            if os.path.exists(f):
                try:
                    df = pd.read_excel(f)
                    if 'Species' in df.columns:
                        for val in df['Species'].dropna().astype(str):
                            variants = self.generate_name_variants(val)
                            for v in variants:
                                if v not in index:
                                    # get classification if available
                                    cat = ''
                                    if 'EICAT Category' in df.columns and val in df['Species'].values:
                                        cat_row = df[df['Species'] == val].iloc[0]
                                        cat = cat_row.get('EICAT Category', '')
                                    index[v] = (os.path.basename(f), cat if cat else 'Listed in GISD')
                except Exception as e:
                    print(f"Error processing {f}: {e}")
        # e120_Plant_aboveground_biomass.csv
        plant_file = "data/invasive/e120_Plant_aboveground_biomass.csv"
        if os.path.exists(plant_file):
            try:
                df = pd.read_csv(plant_file)
                if 'Species' in df.columns:
                    for val in df['Species'].dropna().astype(str):
                        variants = self.generate_name_variants(val)
                        for v in variants:
                            if v not in index:
                                index[v] = ('e120_Plant_aboveground_biomass.csv', 'Listed')
            except Exception as e:
                print(f"Error processing {plant_file}: {e}")
        # E244_Enemy_Removal_Aboveground_Biomass.csv
        enemy_file = "data/invasive/E244_Enemy_Removal_Aboveground_Biomass.csv"
        if os.path.exists(enemy_file):
            try:
                df = pd.read_csv(enemy_file)
                if 'Species' in df.columns:
                    for val in df['Species'].dropna().astype(str):
                        variants = self.generate_name_variants(val)
                        for v in variants:
                            if v not in index:
                                index[v] = ('E244_Enemy_Removal_Aboveground_Biomass.csv', 'Listed')
            except Exception as e:
                print(f"Error processing {enemy_file}: {e}")
        # e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt
        arthropod_file = "data/invasive/e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt"
        if os.path.exists(arthropod_file):
            try:
                df = pd.read_csv(arthropod_file, sep='\t')
                if 'Genus' in df.columns and 'Specific.epithet' in df.columns:
                    # combine Genus and Specific.epithet to form scientific name
                    combined = df['Genus'].astype(str).str.strip() + ' ' + df['Specific.epithet'].astype(str).str.strip()
                    for name in combined:
                        if name and name != 'nan nan':
                            variants = self.generate_name_variants(name)
                            for v in variants:
                                if v not in index:
                                    index[v] = ('e120_Arthropod_Sweepnet.txt', 'Listed')
            except Exception as e:
                print(f"Error processing {arthropod_file}: {e}")
        self.invasive_index = index
        print(f"Invasive index built with {len(index)} unique name variants.")

    def get_invasive_status_from_index(self, species_name):
        """Return (status, source) for the species by checking variants against index."""
        if self.invasive_index is None:
            self.load_invasive_sources()
        variants = self.generate_name_variants(species_name)
        for v in variants:
            if v in self.invasive_index:
                source, classification = self.invasive_index[v]
                if classification:
                    return classification, source
                else:
                    return f"Listed (Source: {source})", source
        return None, None

    def get_iucn_status_gbif(self, species_name):
        """
        Query GBIF for IUCN status using the given species name.
        Returns (status, source) where source is 'GBIF'.
        """
        gbif_base = "https://api.gbif.org/v1"
        variants = self.generate_name_variants(species_name)
        for name in variants:
            try:
                # name match
                match_url = f"{gbif_base}/species/match"
                mresp = self.session.get(match_url, params={"name": name}, timeout=15)
                if mresp.status_code == 200:
                    mjson = mresp.json()
                    usage_key = mjson.get("usageKey") or mjson.get("speciesKey")
                    if usage_key:
                        # get species details
                        spec_url = f"{gbif_base}/species/{usage_key}"
                        sresp = self.session.get(spec_url, timeout=15)
                        if sresp.status_code == 200:
                            sjson = sresp.json()
                            iucn_cat = sjson.get("iucnRedListCategory") or sjson.get("threatenedStatus") or ""
                            if iucn_cat:
                                return str(iucn_cat).strip(), "GBIF"
                        # try explicit iucn endpoint
                        cat_url = f"{gbif_base}/species/{usage_key}/iucnRedListCategory"
                        crep = self.session.get(cat_url, timeout=15)
                        if crep.status_code == 200:
                            cjson = crep.json()
                            if isinstance(cjson, dict):
                                cat = cjson.get("category") or cjson.get("iucnRedListCategory") or ""
                                if cat:
                                    return str(cat).strip(), "GBIF"
                            elif isinstance(cjson, list) and len(cjson) > 0:
                                cat = cjson[0].get("category") or cjson[0].get("iucnRedListCategory") or ""
                                if cat:
                                    return str(cat).strip(), "GBIF"
                    # if we got a match but no IUCN status, check confidence
                    confidence = mjson.get("confidence") or int(mjson.get("score", 0))
                    if confidence >= 50:
                        return "Not Assessed", "GBIF"
                # fallback to next variant
            except Exception:
                continue
        return "Species Not Found", "GBIF"

    def prepare_bifurcation_shortlist(self):
        print("Preparing bifurcation shortlist...")
        bif_file = "csv/bifurcation_results_cleaned.csv"
        tax_file = "csv/taxonomic_cache_cleaned.csv"
        if not os.path.exists(bif_file):
            print(f"Error: {bif_file} not found.")
            return None
        bif_df = pd.read_csv(bif_file)
        if os.path.exists(tax_file):
            tax_df = pd.read_csv(tax_file)
            if 'species' in tax_df.columns and 'taxonomic_group' in tax_df.columns:
                bif_df = bif_df.merge(tax_df[['species', 'taxonomic_group']], on='species', how='left', suffixes=('', '_from_tax'))
                if 'taxonomic_group_from_tax' in bif_df.columns:
                    bif_df['taxonomic_group'] = bif_df['taxonomic_group_from_tax'].combine_first(bif_df['taxonomic_group'])
                    bif_df.drop(columns=['taxonomic_group_from_tax'], inplace=True)
                bif_df.to_csv(bif_file, index=False)
                print("Updated bifurcation_results_cleaned.csv with taxonomic_group from cache.")
        if 'transition_year' not in bif_df.columns:
            print("Error: 'transition_year' column not found in bifurcation_results_cleaned.csv")
            return None
        short_df = bif_df[bif_df['transition_year'].notna() & (bif_df['transition_year'].astype(str).str.strip() != '')].copy()
        os.makedirs("csv", exist_ok=True)
        short_df.to_csv("csv/bifurcation_results_short.csv", index=False)
        print(f"Saved {len(short_df)} species to csv/bifurcation_results_short.csv")
        return short_df

    def build_conservation_database(self, species_df):
        """
        For each species in species_df (with columns species, taxonomic_group),
        query GBIF for IUCN status and source, and append to csv/iucn_species.csv.
        """
        print("Building IUCN conservation database from GBIF...")
        out_file = "csv/iucn_species.csv"
        os.makedirs("csv", exist_ok=True)
        # if file exists, load already processed species
        processed = set()
        if os.path.exists(out_file):
            try:
                existing = pd.read_csv(out_file)
                if 'species' in existing.columns:
                    processed = set(existing['species'].astype(str).str.strip())
                    print(f"Found existing {out_file} with {len(processed)} species.")
            except Exception:
                pass
        # open file in append mode
        mode = 'a' if os.path.exists(out_file) else 'w'
        with open(out_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['species', 'taxonomic_group', 'iucn_status', 'source'])
            if mode == 'w':
                writer.writeheader()
            total = len(species_df)
            for i, row in species_df.iterrows():
                species = str(row['species']).strip()
                if species in processed:
                    print(f"Skipping already processed: {species}")
                    continue
                tax_group = row.get('taxonomic_group', '')
                print(f"GBIF lookup {i+1}/{total}: {species}")
                status, source = self.get_iucn_status_gbif(species)
                writer.writerow({'species': species, 'taxonomic_group': tax_group, 'iucn_status': status, 'source': source})
                f.flush()
                processed.add(species)
                time.sleep(random.uniform(0.5, 1.2))  # be gentle to GBIF
        print(f"Conservation database saved to {out_file}")

    def build_invasive_database(self, species_df):
        """
        For each species in species_df, check invasive status using the index,
        and append to csv/invasive_species.csv.
        """
        print("Building invasive species database from source files...")
        if self.invasive_index is None:
            self.load_invasive_sources()
        out_file = "csv/invasive_species.csv"
        os.makedirs("csv", exist_ok=True)
        processed = set()
        if os.path.exists(out_file):
            try:
                existing = pd.read_csv(out_file)
                if 'species' in existing.columns:
                    processed = set(existing['species'].astype(str).str.strip())
                    print(f"Found existing {out_file} with {len(processed)} species.")
            except Exception:
                pass
        mode = 'a' if os.path.exists(out_file) else 'w'
        with open(out_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['species', 'taxonomic_group', 'invasive_status', 'source'])
            if mode == 'w':
                writer.writeheader()
            total = len(species_df)
            for i, row in species_df.iterrows():
                species = str(row['species']).strip()
                if species in processed:
                    print(f"Skipping already processed: {species}")
                    continue
                tax_group = row.get('taxonomic_group', '')
                print(f"Invasive check {i+1}/{total}: {species}")
                status, source = self.get_invasive_status_from_index(species)
                if status is None:
                    status = "Not Listed"
                    source = ""
                writer.writerow({'species': species, 'taxonomic_group': tax_group, 'invasive_status': status, 'source': source})
                f.flush()
                processed.add(species)
        print(f"Invasive database saved to {out_file}")

    def load_combined_databases(self):
        print("Loading and combining invasive species databases...")
        self.create_invasive_species_csv()
        self.create_iucn_species_csv()
        try:
            if os.path.exists("csv/invasive_species.csv"):
                self.invasive_species_df = pd.read_csv("csv/invasive_species.csv")
                print(f" Combined invasive species database loaded: {len(self.invasive_species_df)} records")
            else:
                print(" Combined invasive species database not found")
            if os.path.exists("csv/iucn_species.csv"):
                self.iucn_species_df = pd.read_csv("csv/iucn_species.csv")
                print(f" Combined IUCN species database loaded: {len(self.iucn_species_df)} records")
            else:
                print(" Combined IUCN species database not found")
        except Exception as e:
            print(f"Error loading combined databases: {str(e)}")

    def create_invasive_species_csv(self):
        print("Creating combined invasive species database...")
        all_invasive_species = []
        eicat_files = ["data/invasive/GISD+EICAT+export_1.xlsx","data/invasive/GISD+EICAT+export_2.xlsx","data/invasive/GISD+EICAT+export_3.xlsx","data/invasive/GISD+EICAT+export_4.xlsx","data/invasive/GISD+EICAT+export_5.xlsx"]
        eicat_count = 0
        for eicat_file in eicat_files:
            try:
                if os.path.exists(eicat_file):
                    eicat_df = pd.read_excel(eicat_file)
                    valid_categories = ['MN', 'MO', 'MR', 'MV']
                    if 'EICAT Category' in eicat_df.columns:
                        filtered_eicat = eicat_df[eicat_df['EICAT Category'].isin(valid_categories)]
                    else:
                        filtered_eicat = eicat_df
                    for _, row in filtered_eicat.iterrows():
                        species_data = {'species': row.get('Species', ''),'kingdom': row.get('Kingdom', ''),'phylum': row.get('Phylum', ''),'class': row.get('Class', ''),'order': row.get('Order', ''),'family': row.get('Family', ''),'genus': '','classification': row.get('EICAT Category', ''),'source': os.path.basename(eicat_file)}
                        all_invasive_species.append(species_data)
                        eicat_count += 1
            except Exception as e:
                print(f" Error processing EICAT file {eicat_file}: {str(e)}")
        if eicat_count > 0:
            print(f" GISD EICAT exports: {eicat_count} species (filtered to exclude DD and MC)")
        try:
            giatar_file = "data/GIATAR/dataset/species lists/invasive_all_source.csv"
            if os.path.exists(giatar_file):
                giatar_df = pd.read_csv(giatar_file)
                invasive_columns = [c for c in ['invasiveCABI','invasiveEPPO'] if c in giatar_df.columns]
                if invasive_columns:
                    invasive_filter = giatar_df[invasive_columns].any(axis=1)
                    invasive_giatar = giatar_df[invasive_filter]
                else:
                    invasive_giatar = giatar_df
                species_columns = ['taxonCABI','taxonSINAS','taxonEPPO','taxonDAISIE','canonicalName','taxonomic_species','gbif_species']
                for _, row in invasive_giatar.iterrows():
                    species_name = ''
                    for col in species_columns:
                        if col in row and pd.notna(row[col]) and row[col] != '':
                            species_name = row[col]
                            break
                    if species_name:
                        species_data = {'species': species_name,'kingdom': row.get('kingdom', ''),'phylum': row.get('phylum', ''),'class': row.get('class', ''),'order': row.get('order', ''),'family': row.get('family', ''),'genus': row.get('genus', ''),'classification': 'Invasive (GIATAR)','source': 'GIATAR_invasive_all_source'}
                        all_invasive_species.append(species_data)
                print(f" GIATAR: {len(invasive_giatar)} invasive species")
            else:
                print(f" GIATAR file not found: {giatar_file}")
        except Exception as e:
            print(f" Error processing GIATAR file: {str(e)}")
        try:
            taxon_file = "data/invasive/taxon.txt"
            if os.path.exists(taxon_file):
                taxon_df = pd.read_csv(taxon_file, sep='\t')
                for _, row in taxon_df.iterrows():
                    species_data = {'species': row.get('scientificName', ''),'kingdom': row.get('kingdom', ''),'phylum': row.get('phylum', ''),'class': row.get('class', ''),'order': row.get('order', ''),'family': row.get('family', ''),'genus': row.get('genus', ''),'classification': 'Listed in GISD','source': 'GISD_taxon'}
                    all_invasive_species.append(species_data)
                print(f" GISD taxon: {len(taxon_df)} species")
            else:
                print(f" GISD taxon file not found: {taxon_file}")
        except Exception as e:
            print(f" Error processing GISD taxon file: {str(e)}")
        if all_invasive_species:
            invasive_df = pd.DataFrame(all_invasive_species)
            invasive_df = invasive_df.drop_duplicates(subset=['species'])
            os.makedirs("csv", exist_ok=True)
            invasive_df.to_csv("csv/invasive_species.csv", index=False)
            print(f" Combined invasive species database saved: {len(invasive_df)} unique species")
        else:
            print(" No invasive species data found from any source")

    def create_iucn_species_csv(self):
        """
        Create combined IUCN species database:
          1) Try IUCN v4 bulk /species endpoint (requires token).
          2) If bulk fails or is not allowed, fall back to GBIF name-match -> species -> iucnRedListCategory.
        Saves progress files during processing and the final CSV to csv/iucn_species.csv.
        """
        print("Creating combined IUCN species database (attempt bulk, then GBIF fallback)...")
        # Attempt IUCN bulk (paginated) only if token present
        if self.iucn_token:
            try:
                all_iucn = []
                page = 1
                max_empty_pages = 3
                consecutive_empty = 0
                max_pages_safety = 20000
                print("Attempting bulk download from IUCN v4 species endpoint (paginated). This may take a while.")
                while page <= max_pages_safety:
                    params = {"page": page, "token": self.iucn_token}
                    url = f"{self.iucn_base_url}/species"
                    resp = self.session.get(url, params=params, timeout=30)
                    if resp.status_code == 401:
                        print("IUCN token invalid or unauthorized for bulk endpoint.")
                        all_iucn = []
                        break
                    if resp.status_code == 429:
                        print("Bulk endpoint rate-limited (429); backing off for 5s.")
                        time.sleep(5)
                        continue
                    if resp.status_code != 200:
                        print(f"Bulk endpoint returned status {resp.status_code} on page {page}; stopping bulk attempt.")
                        all_iucn = []
                        break
                    try:
                        data = resp.json()
                    except Exception as e:
                        print(f"Warning: could not decode JSON on page {page}: {str(e)}; stopping bulk attempt.")
                        all_iucn = []
                        break
                    results = data.get("result", []) if isinstance(data, dict) else []
                    if not results:
                        consecutive_empty += 1
                        if consecutive_empty >= max_empty_pages:
                            break
                        page += 1
                        continue
                    consecutive_empty = 0
                    for item in results:
                        sci = item.get("scientific_name") or item.get("scientificName") or item.get("name") or item.get("species") or ""
                        cat = item.get("category") or item.get("classification") or item.get("status") or ""
                        sci_clean = sci.strip() if isinstance(sci, str) else ""
                        cat_clean = cat.strip() if isinstance(cat, str) else cat
                        if sci_clean:
                            all_iucn.append({"species": sci_clean, "iucn_status": cat_clean})
                    page += 1
                    if page % 50 == 0:
                        print(f"  Bulk download progress: reached page {page}")
                if all_iucn:
                    os.makedirs("csv", exist_ok=True)
                    iucn_df = pd.DataFrame(all_iucn)
                    iucn_df = iucn_df.drop_duplicates(subset=["species"])
                    iucn_df.to_csv("csv/iucn_species.csv", index=False)
                    print(f" Bulk IUCN species file saved: csv/iucn_species.csv ({len(iucn_df)} unique species)")
                    self.iucn_species_df = iucn_df
                    return
                print("Bulk download did not produce usable species data; will try GBIF fallback.")
            except Exception as e:
                print(f"Error during bulk download attempt: {str(e)}")
        else:
            print("No IUCN token available for bulk download; will try GBIF fallback.")
        # GBIF fallback (does not require IUCN token)
        biotime_species_file = "csv/biotime_species_count.csv"
        if not os.path.exists(biotime_species_file):
            print(f"BioTime species file not found for fallback: {biotime_species_file}")
            empty_df = pd.DataFrame(columns=["species", "iucn_status", "source"])
            os.makedirs("csv", exist_ok=True)
            empty_df.to_csv("csv/iucn_species.csv", index=False)
            print("Empty IUCN database structure created: csv/iucn_species.csv")
            return
        print("Loading BioTime species list to attempt GBIF-based retrieval of IUCN status (fallback mode)...")
        biotime_df = pd.read_csv(biotime_species_file)
        if "species" not in biotime_df.columns:
            print("BioTime species file missing 'species' column; cannot perform GBIF fallback.")
            empty_df = pd.DataFrame(columns=["species", "iucn_status", "source"])
            os.makedirs("csv", exist_ok=True)
            empty_df.to_csv("csv/iucn_species.csv", index=False)
            return
        biotime_sorted = biotime_df.sort_values("count", ascending=False) if "count" in biotime_df.columns else biotime_df
        target_species = biotime_sorted.head(5687)
        # Check for existing progress file and load already processed species
        processed_species = set()
        all_iucn_data = []
        progress_file = "csv/iucn_species_progress.csv"
        if os.path.exists(progress_file):
            try:
                progress_df = pd.read_csv(progress_file)
                if "species" in progress_df.columns:
                    processed_species = set(progress_df["species"].astype(str).str.strip())
                    all_iucn_data = progress_df.to_dict('records')
                    print(f"Found existing progress file with {len(processed_species)} processed species")
            except Exception as e:
                print(f"Warning: Could not read progress file {progress_file}: {str(e)}")
        # Filter target species to only unprocessed ones
        unprocessed_species = []
        for idx, row in target_species.iterrows():
            species_name = str(row["species"]).strip()
            if species_name and species_name not in processed_species:
                unprocessed_species.append((idx, row))
        print(f"Will query GBIF for up to {len(unprocessed_species)} unprocessed species (fallback).")
        print(f"Already processed: {len(processed_species)} species")
        gbif_base = "https://api.gbif.org/v1"
        total = len(unprocessed_species)
        processed = 0
        for idx, row in unprocessed_species:
            species_name = str(row["species"]).strip()
            if not species_name:
                all_iucn_data.append({"species": species_name, "iucn_status": "Species Not Found", "source": "GBIF_fallback"})
                continue
            print(f"GBIF lookup {processed+1}/{total}: {species_name}")
            try:
                # 1) name match to get usageKey
                match_url = f"{gbif_base}/species/match"
                mresp = self.session.get(match_url, params={"name": species_name}, timeout=15)
                usage_key = None
                matched_confidence = 0
                if mresp.status_code == 200:
                    mjson = mresp.json()
                    usage_key = mjson.get("usageKey") or mjson.get("speciesKey") or None
                    matched_confidence = mjson.get("confidence") or int(mjson.get("score", 0))
                # 2) if usageKey found, request species endpoint and look for iucnRedListCategory
                iucn_cat = ""
                if usage_key:
                    spec_url = f"{gbif_base}/species/{usage_key}"
                    sresp = self.session.get(spec_url, timeout=15)
                    if sresp.status_code == 200:
                        sjson = sresp.json()
                        iucn_cat = sjson.get("iucnRedListCategory", "") or sjson.get("threatenedStatus", "") or ""
                        if not iucn_cat:
                            try:
                                cat_url = f"{gbif_base}/species/{usage_key}/iucnRedListCategory"
                                crep = self.session.get(cat_url, timeout=15)
                                if crep.status_code == 200:
                                    cjson = crep.json()
                                    if isinstance(cjson, dict):
                                        if "category" in cjson and cjson.get("category"):
                                            iucn_cat = cjson.get("category")
                                        elif "results" in cjson and isinstance(cjson["results"], list) and len(cjson["results"]) > 0:
                                            first = cjson["results"][0]
                                            iucn_cat = first.get("category") or first.get("iucnRedListCategory") or ""
                            except Exception:
                                iucn_cat = ""
                # Normalize result
                if iucn_cat:
                    final_status = str(iucn_cat).strip()
                    source = "GBIF_iucnRedListCategory"
                else:
                    if usage_key and (matched_confidence and int(matched_confidence) >= 50):
                        final_status = "Not Assessed"
                    else:
                        final_status = "Species Not Found"
                    source = "GBIF_fallback"
                all_iucn_data.append({"species": species_name, "iucn_status": final_status, "source": source})
            except Exception as e:
                print(f" Warning: GBIF lookup failed for '{species_name}': {str(e)}")
                all_iucn_data.append({"species": species_name, "iucn_status": f"GBIF Error: {str(e)}", "source": "GBIF_fallback"})
            processed += 1
            # Save progress every 50 records
            if processed % 50 == 0:
                os.makedirs("csv", exist_ok=True)
                progress_df = pd.DataFrame(all_iucn_data)
                progress_df.to_csv(progress_file, index=False)
                print(f" GBIF progress saved: {progress_file} ({len(progress_df)} records)")
            # Respect remote servers
            time.sleep(random.uniform(0.5, 1.2))
        # Save final fallback CSV
        if all_iucn_data:
            os.makedirs("csv", exist_ok=True)
            iucn_df = pd.DataFrame(all_iucn_data)
            iucn_df = iucn_df.drop_duplicates(subset=["species"])
            iucn_df.to_csv("csv/iucn_species.csv", index=False)
            print(f" Fallback IUCN species database saved (GBIF-derived): csv/iucn_species.csv ({len(iucn_df)} records)")
            self.iucn_species_df = iucn_df
            # Remove progress file after successful completion
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                    print(f"Removed progress file: {progress_file}")
                except Exception as e:
                    print(f"Warning: Could not remove progress file {progress_file}: {str(e)}")
            return
        print("No IUCN data collected via GBIF fallback.")
        os.makedirs("csv", exist_ok=True)
        empty_df = pd.DataFrame(columns=["species", "iucn_status", "source"])
        empty_df.to_csv("csv/iucn_species.csv", index=False)

    def get_iucn_status(self, species_name):
        """
        Get IUCN Red List status for a species.
        Workflow:
          1) Clean species name.
          2) If IUCN token present, query IUCN v4 /species/name/{name}.
          3) If IUCN call fails or returns no result, fallback to GBIF name-match -> species -> iucnRedListCategory.
        Returns canonical status string (e.g., 'LC', 'EN', 'Species Not Found', 'IUCN_TOKEN_INVALID', etc.).
        """
        try:
            name = str(species_name).strip()
            name = re.sub(r"\[.*?\]", "", name)
            name = re.sub(r"\(.*?\)", "", name)
            name = re.split(r",|;| subsp\.| ssp\.| var\.| f\.", name)[0].strip()
            name = re.sub(r"\s+", " ", name)
            if name == "":
                return "Species Not Found"
            # 1) IUCN v4 API (preferred) if token present
            if self.iucn_token:
                encoded_name = urllib.parse.quote(name, safe="")
                url_name = f"{self.iucn_base_url}/species/name/{encoded_name}"
                params = {"token": self.iucn_token}
                try:
                    resp = self.session.get(url_name, params=params, timeout=15)
                except requests.exceptions.RequestException:
                    resp = None
                if resp is not None:
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            return "API Error: invalid JSON"
                        results = data.get("result", []) if isinstance(data, dict) else []
                        if results:
                            first = results[0]
                            if "category" in first and first["category"]:
                                return first["category"]
                            if "classification" in first and first["classification"]:
                                return first["classification"]
                            return "Not Assessed"
                        # fall through to GBIF fallback if no results
                    elif resp.status_code == 401:
                        return "IUCN_TOKEN_INVALID"
                    elif resp.status_code == 404:
                        # let GBIF fallback handle it
                        pass
                    else:
                        # allow GBIF fallback
                        pass
            # 2) GBIF fallback: match name -> usageKey -> iucnRedListCategory
            gbif_base = "https://api.gbif.org/v1"
            try:
                match_url = f"{gbif_base}/species/match"
                mresp = self.session.get(match_url, params={"name": name}, timeout=15)
                if mresp.status_code == 200:
                    mjson = mresp.json()
                    usage_key = mjson.get("usageKey") or mjson.get("speciesKey") or None
                    matched_confidence = mjson.get("confidence") or int(mjson.get("score", 0))
                    if usage_key:
                        spec_url = f"{gbif_base}/species/{usage_key}"
                        sresp = self.session.get(spec_url, timeout=15)
                        if sresp.status_code == 200:
                            sjson = sresp.json()
                            iucn_cat = sjson.get("iucnRedListCategory") or sjson.get("threatenedStatus") or ""
                            if iucn_cat:
                                return str(iucn_cat).strip()
                        # explicit iucnRedListCategory endpoint
                        cat_url = f"{gbif_base}/species/{usage_key}/iucnRedListCategory"
                        crep = self.session.get(cat_url, timeout=15)
                        if crep.status_code == 200:
                            try:
                                cjson = crep.json()
                            except Exception:
                                return "GBIF API Error: invalid JSON"
                            if isinstance(cjson, dict):
                                if "category" in cjson and cjson.get("category"):
                                    return cjson.get("category")
                                if "results" in cjson and isinstance(cjson["results"], list) and len(cjson["results"]) > 0:
                                    first = cjson["results"][0]
                                    if isinstance(first, dict):
                                        return first.get("category") or first.get("iucnRedListCategory") or "Not Assessed"
                            elif isinstance(cjson, list) and len(cjson) > 0 and isinstance(cjson[0], dict):
                                return cjson[0].get("category") or cjson[0].get("iucnRedListCategory") or "Not Assessed"
                        # No category found at GBIF
                        if matched_confidence and int(matched_confidence) >= 50:
                            return "Not Assessed"
                        return "Species Not Found"
                else:
                    return f"GBIF API Error: {mresp.status_code}"
            except requests.exceptions.RequestException as e:
                return f"Request Error: {str(e)}"
            except Exception as e:
                return f"Error: {str(e)}"
        except requests.exceptions.RequestException as e:
            return f"Request Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def check_invasive_potential(self, species_name):
        if self.invasive_species_df is None:
            return "Database Not Loaded"
        try:
            species_lower = str(species_name).lower().strip()
            exact_matches = self.invasive_species_df[self.invasive_species_df['species'].astype(str).str.lower() == species_lower]
            if not exact_matches.empty:
                row = exact_matches.iloc[0]
                classification = row.get('classification', '')
                source = row.get('source', '')
                if classification and pd.notna(classification):
                    return f"Listed (Classification: {classification}, Source: {source})"
                else:
                    return f"Listed (Source: {source})"
            partial_matches = self.invasive_species_df[self.invasive_species_df['species'].astype(str).str.lower().str.contains(species_lower, na=False)]
            if not partial_matches.empty:
                return "Potential match in invasive database"
            return "Not Listed"
        except Exception as e:
            return f"Error: {str(e)}"

    def process_species_data(self, input_file, output_file, top_n=5687):
        try:
            df = pd.read_csv(input_file)
            print(f" Input file loaded: {len(df)} total species")
        except FileNotFoundError:
            print(f" Error: The file {input_file} does not exist!")
            return
        except Exception as e:
            print(f" Error reading file: {str(e)}")
            return
        if "count" not in df.columns:
            print(" Error: 'count' column not found in the CSV file!")
            return
        if "species" not in df.columns:
            print(" Error: 'species' column not found in the CSV file!")
            return
        df_sorted = df.sort_values("count", ascending=False)
        top_species = df_sorted.head(top_n).copy()
        if top_species.shape[0] == 0:
            print("No species to process after applying top_n; exiting.")
            empty_df = pd.DataFrame(columns=["species", "IUCN_Status", "Invasive_Potential"])
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            empty_df.to_csv(output_file, index=False)
            return
        print(f"Processing top {len(top_species)} species by occurrence count...")
        print(f"Species with highest occurrences: {top_species.iloc[0]['species']} ({top_species.iloc[0]['count']} occurrences)")
        print(f"Species with lowest in top {top_n}: {top_species.iloc[-1]['species']} ({top_species.iloc[-1]['count']} occurrences)")
        if "IUCN_Status" not in top_species.columns:
            top_species["IUCN_Status"] = ""
        if "Invasive_Potential" not in top_species.columns:
            top_species["Invasive_Potential"] = ""
        processed_set = set()
        header_needed = True
        processed_count = 0
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        # Check for existing output file to resume processing
        if os.path.exists(output_file):
            try:
                existing = pd.read_csv(output_file)
                if "species" in existing.columns:
                    processed_set = set(existing["species"].astype(str).str.strip())
                    print(f"Found existing output file with {len(processed_set)} processed species")
                else:
                    print(" Warning: existing output file lacks 'species' column; resume not attempted.")
                if len(existing) > 0:
                    header_needed = False
                for i, r in existing.iterrows():
                    if "species" in r and pd.notna(r.get("species", None)):
                        mask = top_species["species"].astype(str).str.strip() == str(r["species"]).strip()
                        if mask.any():
                            idx = top_species[mask].index[0]
                            if "IUCN_Status" in existing.columns:
                                top_species.at[idx, "IUCN_Status"] = r.get("IUCN_Status", "")
                            if "Invasive_Potential" in existing.columns:
                                top_species.at[idx, "Invasive_Potential"] = r.get("Invasive_Potential", "")
            except Exception as e:
                print(f" Warning: could not read existing output file to resume: {str(e)}")
        try:
            for idx, row in top_species.iterrows():
                species = str(row["species"]).strip()
                if species in processed_set:
                    print(f"Skipping already-processed species: {species}")
                    continue
                print(f"Processing {idx+1}/{len(top_species)}: {species}")
                status = None
                # First try to get status from bulk IUCN database
                if self.iucn_species_df is not None and "species" in self.iucn_species_df.columns:
                    matched = self.iucn_species_df[self.iucn_species_df["species"].astype(str).str.strip().str.lower() == species.lower()]
                    if not matched.empty:
                        status = str(matched.iloc[0].get("iucn_status", "")).strip()
                        print(f"  Retrieved status from bulk file: {status}")
                # If not found in bulk, query API
                if status is None or status == "":
                    status = self.get_iucn_status(species)
                    print(f"  Queried API status: {status}")
                invasive_status = self.check_invasive_potential(species)
                top_species.at[idx, "IUCN_Status"] = status
                top_species.at[idx, "Invasive_Potential"] = invasive_status
                row_out = {"species": species, "IUCN_Status": status, "Invasive_Potential": invasive_status}
                try:
                    file_exists = os.path.exists(output_file)
                    with open(output_file, "a", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=["species", "IUCN_Status", "Invasive_Potential"])
                        if not file_exists and header_needed:
                            writer.writeheader()
                        writer.writerow(row_out)
                        fh.flush()
                        os.fsync(fh.fileno())
                    header_needed = False
                except Exception as e:
                    print(f" Error appending to output file for {species}: {str(e)}")
                processed_set.add(species)
                processed_count += 1
                time.sleep(random.uniform(1, 2))
                if (processed_count % 50) == 0:
                    backup_file = f"csv/progress_backup_{processed_count}.csv"
                    try:
                        os.makedirs("csv", exist_ok=True)
                        top_species[top_species["species"].astype(str).str.strip().isin(list(processed_set))].to_csv(backup_file, index=False)
                        print(f" Progress saved: {backup_file}")
                        self.print_interim_summary(top_species, processed_count)
                    except Exception as e:
                        print(f" Warning: could not save progress backup: {str(e)}")
        except Exception as e:
            print(f" Error during processing: {str(e)}")
        try:
            if os.path.exists(output_file):
                final_df = pd.read_csv(output_file)
                try:
                    final_df.to_csv(output_file, index=False)
                    print(f" Final results saved to: {output_file}")
                except Exception as e:
                    print(f" Error re-saving final results: {str(e)}")
            else:
                top_species.to_csv(output_file, index=False)
                print(f" Final results saved to: {output_file}")
        except Exception as e:
            print(f" Error saving results: {str(e)}")
        self.print_summary(top_species)

    def print_interim_summary(self, df, current_count):
        print(f"\n--- INTERIM SUMMARY (after {current_count} species) ---")
        current_df = df.head(current_count)
        iucn_summary = current_df['IUCN_Status'].value_counts()
        invasive_summary = current_df['Invasive_Potential'].value_counts()
        print("IUCN Status:")
        for status, count in iucn_summary.items():
            print(f"  {status}: {count}")
        print("Invasive Potential:")
        for status, count in invasive_summary.items():
            print(f"  {status}: {count}")
        print()

    def print_summary(self, df):
        print("\n" + "="*60)
        print("FINAL PROCESSING SUMMARY")
        print("="*60)
        print(f"Total species processed: {len(df)}")
        print("\nIUCN Status Distribution:")
        iucn_summary = df['IUCN_Status'].value_counts()
        for status, count in iucn_summary.items():
            percentage = (count / len(df)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")
        print("\nInvasive Potential Distribution:")
        invasive_summary = df['Invasive_Potential'].value_counts()
        for status, count in invasive_summary.items():
            percentage = (count / len(df)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")
        invasive_count = len(df[df['Invasive_Potential'].str.contains('Listed|Potential', na=False)])
        invasive_percentage = (invasive_count / len(df)) * 100
        print(f"\nSpecies with invasive potential: {invasive_count} ({invasive_percentage:.1f}%)")
        threatened_categories = ['CR', 'EN', 'VU', 'EW', 'EX']
        threatened_count = len(df[df['IUCN_Status'].isin(threatened_categories)])
        threatened_percentage = (threatened_count / len(df)) * 100
        print(f"Threatened species (VU/EN/CR): {threatened_count} ({threatened_percentage:.1f}%)")

def main():
    print("SPECIES CONSERVATION AND INVASIVENESS ASSESSMENT")
    print("=" * 60)
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("csv").mkdir(exist_ok=True)
    collector = SpeciesDataCollector()
    # Step 1: Prepare bifurcation shortlist
    short_df = collector.prepare_bifurcation_shortlist()
    if short_df is None or len(short_df) == 0:
        print("No species to process. Exiting.")
        return
    # Step 2: Build conservation database (GBIF only)
    collector.build_conservation_database(short_df)
    # Step 3: Build invasive database
    collector.build_invasive_database(short_df)
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
