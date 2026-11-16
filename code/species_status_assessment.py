import os, time, random, csv, requests, warnings, pickle
import numpy as np
import pandas as pd
import urllib.parse
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from main_dl import apply_synonym_mapping, clean_taxonomic_groups, load_combined_data, get_cached_taxonomic_group, generate_name_variants, BANDING_CODE_TO_NAME
warnings.filterwarnings('ignore')
random.seed(1234)
np.random.seed(1234)

def load_taxonomic_cache_map():
    cache_file = 'csv/taxonomic_cache.csv'
    if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
        iucn_file = 'csv/iucn_species.csv'
        if os.path.exists(iucn_file) and os.path.getsize(iucn_file) > 0:
            iucn_df = pd.read_csv(iucn_file)
            if 'species' in iucn_df.columns and 'taxonomic_group' in iucn_df.columns:
                new_cache = iucn_df[['species', 'taxonomic_group']].copy()
                new_cache['species'] = new_cache['species'].astype(str).str.strip()
                new_cache['taxonomic_group'] = new_cache['taxonomic_group'].fillna('').astype(str).str.strip()
                new_cache = new_cache[(new_cache['species'] != '') & (new_cache['taxonomic_group'] != '')].drop_duplicates(subset=['species']).reset_index(drop=True)
                os.makedirs('csv', exist_ok=True)
                new_cache.to_csv(cache_file, index=False)
                print(f"Created {cache_file} from {iucn_file} with {len(new_cache)} species.")
                return dict(zip(new_cache['species'], new_cache['taxonomic_group']))
        return {}
    df = pd.read_csv(cache_file)
    if 'species' not in df.columns or 'taxonomic_group' not in df.columns:
        return {}
    return dict(zip(df['species'].astype(str).str.strip(), df['taxonomic_group'].astype(str).str.strip()))

class SpeciesDataCollector:
    generate_name_variants = staticmethod(generate_name_variants)
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
        self.invasive_species_df = None
        self.iucn_species_df = None
        self.invasive_index = None
    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(['GET','POST','PUT','DELETE','HEAD','OPTIONS']))
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    def get_cedar_creek_species_from_main_dl(self):
        datasets_dict, _, _, _ = load_combined_data(min_records=1, prefilter=False)
        if 'CedarCreek' not in datasets_dict:
            return []
        cc = datasets_dict['CedarCreek']
        if len(cc) == 0 or 'SPECIES_ID' not in cc.columns:
            return []
        return cc['SPECIES_ID'].dropna().astype(str).unique().tolist()
    def load_invasive_sources(self):
        cache_file = "csv/invasive_index.pkl"
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            try:
                with open(cache_file, 'rb') as f:
                    self.invasive_index = pickle.load(f)
                print(f"Loaded invasive index from {cache_file} with {len(self.invasive_index)} unique name variants.")
                return
            except Exception as e:
                print(f"Warning: Could not load cached invasive index: {str(e)}")
        print("Building invasive species index from source files...")
        index = {}
        giatar_file = "data/GIATAR/dataset/species lists/invasive_all_source.csv"
        if os.path.exists(giatar_file):
            try:
                df = pd.read_csv(giatar_file)
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
                                    cat = ''
                                    if 'EICAT Category' in df.columns and val in df['Species'].values:
                                        cat_row = df[df['Species'] == val].iloc[0]
                                        cat = cat_row.get('EICAT Category', '')
                                    index[v] = (os.path.basename(f), cat if cat else 'Listed in GISD')
                except Exception as e:
                    print(f"Error processing {f}: {e}")
        cedar_species = self.get_cedar_creek_species_from_main_dl()
        for val in cedar_species:
            variants = self.generate_name_variants(val)
            for v in variants:
                if v not in index:
                    index[v] = ('CedarCreek_via_main_dl', 'Listed')
        self.invasive_index = index
        try:
            os.makedirs("csv", exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(index, f)
            print(f"Cached invasive index to {cache_file}.")
        except Exception as e:
            print(f"Warning: Could not cache invasive index: {str(e)}")
        print(f"Invasive index built with {len(index)} unique name variants.")
    def get_invasive_status_from_index(self, species_name):
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
        gbif_base = "https://api.gbif.org/v1"
        candidates = self.generate_name_variants(species_name)
        if not candidates:
            return "Not Assessed", "GBIF"
        best_fallback = None
        for name in candidates:
            try:
                match_url = f"{gbif_base}/species/match"
                mresp = self.session.get(match_url, params={"name": name}, timeout=15)
                if mresp.status_code != 200:
                    continue
                mjson = mresp.json()
                match_type = (mjson.get("matchType") or "").upper()
                if match_type in ("NONE", "HIGHERRANK"):
                    continue
                usage_key = mjson.get("usageKey") or mjson.get("speciesKey")
                if not usage_key:
                    continue
                cat_url = f"{gbif_base}/species/{usage_key}/iucnRedListCategory"
                crep = self.session.get(cat_url, timeout=15)
                if crep.status_code == 200:
                    try:
                        cjson = crep.json()
                    except Exception:
                        cjson = None
                    cat = None
                    if isinstance(cjson, dict):
                        cat = cjson.get("category") or cjson.get("iucnRedListCategory")
                        if not cat and isinstance(cjson.get("results"), list) and cjson["results"]:
                            first = cjson["results"][0]
                            if isinstance(first, dict):
                                cat = first.get("category") or first.get("iucnRedListCategory")
                    elif isinstance(cjson, list) and cjson and isinstance(cjson[0], dict):
                        cat = cjson[0].get("category") or cjson[0].get("iucnRedListCategory")
                    if cat:
                        cat_str = str(cat).strip()
                        if cat_str.upper() not in ("NOT_EVALUATED", "NE"):
                            return cat_str, "GBIF"
                spec_url = f"{gbif_base}/species/{usage_key}"
                sresp = self.session.get(spec_url, timeout=15)
                if sresp.status_code == 200:
                    sjson = sresp.json()
                    iucn_cat = sjson.get("iucnRedListCategory") or sjson.get("threatenedStatus") or ""
                    if iucn_cat:
                        iucn_str = str(iucn_cat).strip()
                        if iucn_str.upper() not in ("NOT_EVALUATED", "NE"):
                            return iucn_str, "GBIF"
                if best_fallback is None and match_type in ("EXACT", "FUZZY"):
                    best_fallback = "Not Assessed"
            except Exception:
                continue
        result_status = best_fallback or "Not Assessed"
        if result_status == "Not Assessed" and species_name.upper() in BANDING_CODE_TO_NAME:
            mapped = BANDING_CODE_TO_NAME[species_name.upper()]
            if mapped and mapped != species_name:
                return self.get_iucn_status_gbif(mapped)
        return result_status, "GBIF"
    def get_iucn_status(self, species_name):
        code_to_full = {"LC": "LEAST_CONCERN", "NT": "NEAR_THREATENED", "VU": "VULNERABLE", "EN": "ENDANGERED", "CR": "CRITICALLY_ENDANGERED", "EW": "EXTINCT_IN_THE_WILD", "EX": "EXTINCT", "DD": "DATA_DEFICIENT", "NE": "Not Assessed", "NA": "Not Assessed"}
        candidates = self.generate_name_variants(species_name)
        if not candidates:
            return "Not Assessed"
        if self.iucn_token:
            headers = {"Authorization": f"Bearer {self.iucn_token}", "accept": "application/json"}
            for name in candidates:
                parts = name.split()
                if len(parts) < 2:
                    continue
                genus, species_epithet = parts[0], parts[1]
                try:
                    resp = self.session.get(f"{self.iucn_base_url}/taxa/scientific_name", params={"genus_name": genus, "species_name": species_epithet}, headers=headers, timeout=15)
                except requests.exceptions.RequestException:
                    continue
                if resp.status_code == 401:
                    return "IUCN_TOKEN_INVALID"
                if resp.status_code != 200:
                    continue
                try:
                    data = resp.json()
                except Exception:
                    continue
                taxon = data.get("taxon", {})
                sis_id = taxon.get("sis_id")
                if not sis_id:
                    species_taxa = taxon.get("species_taxa", [])
                    if species_taxa:
                        sis_id = species_taxa[0].get("sis_id")
                if not sis_id:
                    continue
                try:
                    sis_resp = self.session.get(f"{self.iucn_base_url}/taxa/sis/{sis_id}", headers=headers, timeout=15)
                except requests.exceptions.RequestException:
                    continue
                if sis_resp.status_code != 200:
                    continue
                try:
                    sis_data = sis_resp.json()
                except Exception:
                    continue
                assessments = sis_data.get("assessments", [])
                if not assessments:
                    continue
                latest = None
                for a in assessments:
                    if a.get("latest") is True:
                        latest = a
                        break
                if latest is None:
                    latest = assessments[0]
                code = latest.get("red_list_category_code")
                if code:
                    code_str = str(code).strip().upper()
                    if code_str:
                        return code_to_full.get(code_str, code_str)
        if species_name.upper() in BANDING_CODE_TO_NAME:
            mapped = BANDING_CODE_TO_NAME[species_name.upper()]
            if mapped and mapped != species_name:
                return self.get_iucn_status(mapped)
        return "Not Assessed"
    def prepare_bifurcation_shortlist(self):
        print("Preparing bifurcation shortlist...")
        short_file = "csv/bifurcation_results_short.csv"
        if not os.path.exists(short_file):
            print(f"Error: {short_file} not found.")
            return None
        short_df = pd.read_csv(short_file)
        print(f"Loaded {len(short_df)} species from {short_file}")
        if 'transition_year' not in short_df.columns:
            print("Error: 'transition_year' column not found in bifurcation_results_short.csv")
            return None
        return short_df
    def build_conservation_database(self, species_df):
        print("Building IUCN conservation database from IUCN and GBIF APIs...")
        out_file = "csv/iucn_species.csv"
        tax_map = load_taxonomic_cache_map()
        os.makedirs("csv", exist_ok=True)
        processed = set()
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            existing = pd.read_csv(out_file)
            if 'species' in existing.columns:
                processed = set(existing['species'].astype(str).str.strip())
                print(f"Found existing {out_file} with {len(processed)} species.")
            for col in ['species', 'taxonomic_group', 'iucn_status', 'source']:
                if col not in existing.columns:
                    existing[col] = ''
            existing['source'] = existing['source'].astype(str).replace({'GBIF_online': 'GBIF', 'GBIF_fallback': 'GBIF'})
            mask = existing['taxonomic_group'].isna() | (existing['taxonomic_group'].astype(str).str.strip() == '')
            existing.loc[mask, 'taxonomic_group'] = existing.loc[mask, 'species'].astype(str).str.strip().map(lambda sp: tax_map.get(sp) or get_cached_taxonomic_group(sp))
            existing = existing[['species', 'taxonomic_group', 'iucn_status', 'source']]
            existing.to_csv(out_file, index=False)
        mode = 'a' if os.path.exists(out_file) else 'w'
        with open(out_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['species', 'taxonomic_group', 'iucn_status', 'source'])
            if mode == 'w':
                writer.writeheader()
            total = len(species_df)
            for i, row in species_df.iterrows():
                species = str(row['species']).strip()
                if species in processed:
                    continue
                tax_group = tax_map.get(species) or get_cached_taxonomic_group(species)
                print(f"IUCN lookup {i+1}/{total}: {species}")
                status = self.get_iucn_status(species)
                source = "IUCN"
                if status in ("Not Assessed", "IUCN_TOKEN_INVALID"):
                    status, source = self.get_iucn_status_gbif(species)
                writer.writerow({'species': species, 'taxonomic_group': tax_group, 'iucn_status': status, 'source': source})
                f.flush()
                processed.add(species)
                time.sleep(random.uniform(0.5, 1.2))
        print(f"Conservation database saved to {out_file}")
    def build_invasive_database(self, species_df):
        print("Building invasive species database from source files...")
        if self.invasive_index is None:
            self.load_invasive_sources()
        out_file = "csv/invasive_species.csv"
        tax_map = load_taxonomic_cache_map()
        os.makedirs("csv", exist_ok=True)
        processed = set()
        mode = 'w'
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            try:
                existing = pd.read_csv(out_file)
                if 'species' in existing.columns and 'invasive_status' in existing.columns:
                    processed = set(existing['species'].astype(str).str.strip())
                    mode = 'a'
                    print(f"Resuming {out_file} with {len(processed)} species.")
                else:
                    print(f"{out_file} has wrong schema; will be overwritten.")
            except Exception:
                pass
        with open(out_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['species', 'taxonomic_group', 'invasive_status', 'source'])
            if mode == 'w':
                writer.writeheader()
            total = len(species_df)
            for i, row in species_df.iterrows():
                species = str(row['species']).strip()
                if species in processed:
                    continue
                tax_group = tax_map.get(species) or get_cached_taxonomic_group(species)
                print(f"Invasive check {i+1}/{total}: {species}")
                status, source = self.get_invasive_status_from_index(species)
                if status is None:
                    status = "Not Listed"
                    source = ""
                writer.writerow({'species': species, 'taxonomic_group': tax_group, 'invasive_status': status, 'source': source})
                f.flush()
                processed.add(species)
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
        out_file = "csv/invasive_species.csv"
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            try:
                existing = pd.read_csv(out_file)
                if 'species' in existing.columns and 'taxonomic_group' in existing.columns and 'invasive_status' in existing.columns and 'source' in existing.columns:
                    print(f"{out_file} already has correct schema; skipping rebuild.")
                    return
            except Exception:
                pass
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
    def get_species_from_bifurcation_results(self):
        files = ["csv/bifurcation_results_dl_ews.csv", "csv/bifurcation_results_dev.csv"]
        species = set()
        for f in files:
            if os.path.exists(f) and os.path.getsize(f) > 0:
                df = pd.read_csv(f)
                if "species" in df.columns:
                    species.update(df["species"].dropna().astype(str).str.strip().tolist())
        return sorted(species)
    def create_iucn_species_csv(self):
        print("Creating combined IUCN species database...")
        tax_map = load_taxonomic_cache_map()
        iucn_file = "csv/iucn_species.csv"
        existing_df = pd.DataFrame(columns=["species", "taxonomic_group", "iucn_status", "source"])
        if os.path.exists(iucn_file) and os.path.getsize(iucn_file) > 0:
            try:
                existing_df = pd.read_csv(iucn_file)
                if 'species' not in existing_df.columns:
                    existing_df = pd.DataFrame(columns=["species", "taxonomic_group", "iucn_status", "source"])
                else:
                    for col in ["species", "taxonomic_group", "iucn_status", "source"]:
                        if col not in existing_df.columns:
                            existing_df[col] = ''
                    existing_df = existing_df[["species", "taxonomic_group", "iucn_status", "source"]]
                    existing_df['species'] = existing_df['species'].astype(str).str.strip()
                    existing_df['source'] = existing_df['source'].astype(str).replace({'GBIF_online': 'GBIF', 'GBIF_fallback': 'GBIF'})
                    mask = existing_df['taxonomic_group'].isna() | (existing_df['taxonomic_group'].astype(str).str.strip() == '')
                    if mask.any():
                        existing_df.loc[mask, 'taxonomic_group'] = existing_df.loc[mask, 'species'].map(lambda sp: tax_map.get(sp) or get_cached_taxonomic_group(sp))
                    existing_df.to_csv(iucn_file, index=False)
                    print(f"Loaded existing {len(existing_df)} species from {iucn_file}")
            except Exception as e:
                print(f"Warning: Could not read existing {iucn_file}: {str(e)}")
        existing_species = set(existing_df["species"].astype(str).str.strip())
        target_species = pd.DataFrame({"species": self.get_species_from_bifurcation_results()})
        if target_species.empty:
            print("No species collected from bifurcation result files.")
            return
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
        unprocessed_species = []
        for idx, row in target_species.iterrows():
            species_name = str(row["species"]).strip()
            if species_name and species_name not in processed_species and species_name not in existing_species:
                unprocessed_species.append((idx, row))
        print(f"Species already in file: {len(existing_species)}")
        print(f"Species in progress file: {len(processed_species)}")
        print(f"Will query IUCN for up to {len(unprocessed_species)} unprocessed species.")
        total = len(unprocessed_species)
        processed = 0
        for idx, row in unprocessed_species:
            species_name = str(row["species"]).strip()
            if not species_name:
                all_iucn_data.append({"species": species_name, "iucn_status": "Not Assessed", "source": "GBIF"})
                continue
            print(f"IUCN lookup {processed+1}/{total}: {species_name}")
            try:
                status = self.get_iucn_status(species_name)
                source = "IUCN"
                if status in ("Not Assessed", "IUCN_TOKEN_INVALID"):
                    status, source = self.get_iucn_status_gbif(species_name)
                all_iucn_data.append({"species": species_name, "iucn_status": status, "source": source})
            except Exception as e:
                print(f" Warning: IUCN lookup failed for '{species_name}': {str(e)}")
                all_iucn_data.append({"species": species_name, "iucn_status": f"GBIF Error: {str(e)}", "source": "GBIF"})
            processed += 1
            if processed % 50 == 0:
                os.makedirs("csv", exist_ok=True)
                progress_df = pd.DataFrame(all_iucn_data)
                progress_df.to_csv(progress_file, index=False)
                print(f" Progress saved: {progress_file} ({len(progress_df)} records)")
            time.sleep(random.uniform(0.5, 1.2))
        if all_iucn_data:
            new_df = pd.DataFrame(all_iucn_data)
            new_df = new_df.drop_duplicates(subset=["species"])
            new_df['taxonomic_group'] = new_df['species'].astype(str).str.strip().map(lambda sp: tax_map.get(sp) or get_cached_taxonomic_group(sp))
            new_df['source'] = new_df['source'].astype(str).replace({'GBIF_online': 'GBIF', 'GBIF_fallback': 'GBIF'})
            new_df = new_df[["species", "taxonomic_group", "iucn_status", "source"]]
            combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["species"], keep="last")
            combined.to_csv(iucn_file, index=False)
            print(f" IUCN species database saved: {iucn_file} ({len(combined)} records)")
            self.iucn_species_df = combined
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                    print(f"Removed progress file: {progress_file}")
                except Exception as e:
                    print(f"Warning: Could not remove progress file {progress_file}: {str(e)}")
            return
        print("No new IUCN data collected; existing file preserved.")

def recheck_not_assessed_species(short_df):
    iucn_file = "csv/iucn_species.csv"
    if not os.path.exists(iucn_file):
        print("iucn_species.csv not found. Skipping recheck.")
        return
    iucn_df = pd.read_csv(iucn_file)
    short_species = set(short_df['species'].astype(str).str.strip())
    mask = (iucn_df['iucn_status'].astype(str).str.strip() == "Not Assessed") & (iucn_df['species'].astype(str).str.strip().isin(short_species))
    to_recheck = iucn_df[mask].copy()
    if len(to_recheck) == 0:
        print("No 'Not Assessed' species from short_df found to recheck.")
        return
    print(f"Rechecking {len(to_recheck)} species...")
    collector = SpeciesDataCollector()
    updated = 0
    for idx, row in to_recheck.iterrows():
        species = str(row['species']).strip()
        variants = collector.generate_name_variants(species)
        found_status = None
        for variant in variants:
            status = collector.get_iucn_status(variant)
            if status and status != "Not Assessed" and not status.startswith("Error") and not status.startswith("Request") and not status.startswith("GBIF API Error"):
                found_status = status
                break
        if found_status is not None:
            iucn_df.at[idx, 'iucn_status'] = found_status
            updated += 1
            print(f"Updated {species} -> {found_status}")
        else:
            print(f"No better status found for {species}")
    if updated > 0:
        iucn_df.to_csv(iucn_file, index=False)
        print(f"Recheck complete. Updated {updated} records in {iucn_file}")
    else:
        print("No updates.")

def setup_environment():
    Path("data").mkdir(exist_ok=True)
    Path("csv").mkdir(exist_ok=True)
    print("SPECIES CONSERVATION AND INVASIVENESS ASSESSMENT")
    print("=" * 60)

def build_taxonomic_cache_from_large(large_file='csv/bifurcation_results_large.csv'):
    df = pd.read_csv(large_file)
    if 'species' not in df.columns:
        raise ValueError("bifurcation_results_large.csv missing species column")
    species = df['species'].dropna().astype(str).str.strip().unique()
    tax_map = load_taxonomic_cache_map()
    for sp in species:
        if sp and sp not in tax_map:
            get_cached_taxonomic_group(sp)
    return len(species)

def run_assessment(collector, short_df, recheck_not_assessed):
    if short_df is None or len(short_df) == 0:
        print("No species to process. Exiting.")
        return
    collector.build_conservation_database(short_df)
    collector.build_invasive_database(short_df)
    if recheck_not_assessed:
        recheck_not_assessed_species(short_df)
    print("\nAll tasks completed successfully.")

def main():
    RECHECK_NOT_ASSESSED = False
    setup_environment()
    build_taxonomic_cache_from_large()
    collector = SpeciesDataCollector()
    collector.load_combined_databases()
    short_df = collector.prepare_bifurcation_shortlist()
    run_assessment(collector, short_df, RECHECK_NOT_ASSESSED)

if __name__ == "__main__":
    main()
