#!/usr/bin/env python
# coding: utf-8

import os
import gc
import re
import sys
import time
import shutil
import logging
import ewstools
import requests
import traceback
import absl.logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from scipy.stats import linregress
from joblib import Parallel, delayed, Memory
from functools import partial
from tqdm import tqdm
import warnings

def setup_warning_suppression(worker_process=True): # worker_process: If True, applies additional suppression for parallel workers
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Suppress TensorFlow oneDNN warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # General warnings
    warnings.filterwarnings('ignore')
    # TensorFlow specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
    # TensorFlow SavedModel warnings
    warnings.filterwarnings("ignore", message="SavedModel saved prior to TF 2.5")
    warnings.filterwarnings("ignore", message=".*SavedModel.*")
    # TensorFlow oneDNN and GPU warnings
    warnings.filterwarnings("ignore", message=".*oneDNN.*")
    warnings.filterwarnings("ignore", message=".*oneAPI Deep Neural Network Library.*")
    warnings.filterwarnings("ignore", message=".*Could not load dynamic library.*")
    warnings.filterwarnings("ignore", message=".*libcudart.*")
    warnings.filterwarnings("ignore", message=".*libnvinfer.*")
    warnings.filterwarnings("ignore", message=".*TF-TRT Warning.*")
    # ewstools warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="ewstools")
    warnings.filterwarnings("ignore", category=UserWarning, module="ewstools")
    # NumPy warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.fromnumeric")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core._methods")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
    # Other library warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="ruptures")
    # Additional worker-specific suppressions
    if worker_process:
        # Even more aggressive suppression for parallel workers
        warnings.filterwarnings("ignore", category=Warning)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        # Suppress any remaining TensorFlow initialization messages
        try:
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
    # Configure TensorFlow logger (only if TensorFlow is available)
    try:
        # TensorFlow 2.11.1 compatible logging setup
        tf_logger = tf.get_logger()
        tf_logger.setLevel('ERROR')
        # For TensorFlow 2.11.1, use tf.compat.v1 for additional control if needed
        try:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except:
            pass
        # Disable eager execution warnings (if using TF 2.x with compat v1)
        try:
            tf.compat.v1.disable_eager_execution()
        except:
            pass
        # Set TensorFlow to only log errors
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass  # TensorFlow not available in this context
    # Suppress specific TensorFlow info messages by redirecting stdout/stderr temporarily
    if not worker_process:  # Only do this in main process to avoid issues with parallel workers
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        try:
            # Redirect during critical TensorFlow operations
            sys.stderr = open(os.devnull, 'w')
            sys.stdout = open(os.devnull, 'w')
            # Re-import TensorFlow with suppressed output
            tf.get_logger().setLevel('ERROR')
            # Restore stdout/stderr
            sys.stderr = original_stderr
            sys.stdout = original_stdout
        except:
            # Restore in case of error
            sys.stderr = original_stderr
            sys.stdout = original_stdout

# Safe division function to handle zero mean case
def safe_divide(data):
    mean_abs = abs(data).mean()
    if mean_abs == 0:
        return np.zeros_like(data)
    return data / mean_abs

def patched_apply_classifier(self, classifier, tmin, tmax, name="c1", verbose=1):
    input_len = classifier.layers[0].input_shape[1]
    if "residuals" in self.state.columns:
        series = self.state[(self.state.index >= tmin) & (self.state.index < tmax)]["residuals"]
    else:
        series = self.state[(self.state.index >= tmin) & (self.state.index < tmax)]["state"]
    data = series.values
    if len(data) > input_len:
        print("ERROR: Length of time series segment is too long for the classifier.")
        return
    # Use safe division instead of direct division
    data_norm = safe_divide(data)
    num_zeros = input_len - len(data_norm)
    input_data = np.concatenate((np.zeros(num_zeros), data_norm)).reshape(1, -1, 1)
    dl_pred = classifier.predict(input_data, verbose=verbose)[0]
    dict_dl_pred = {i: val for (i, val) in zip(np.arange(len(dl_pred)), dl_pred)}
    dict_dl_pred["time"] = series.index[-1]
    dict_dl_pred["classifier"] = name
    df_dl_pred = pd.DataFrame(dict_dl_pred, index=[len(self.dl_preds)])
    self.dl_preds = pd.concat([self.dl_preds, df_dl_pred], ignore_index=True)

import ewstools.core
ewstools.core.normalize_data = safe_divide
ewstools.core.TimeSeries.apply_classifier = patched_apply_classifier

# Configuration constants
API_DELAY = 0.3  # Seconds between GBIF API calls
MAX_RECURSION_DEPTH = 10  # Max levels to search up taxonomy tree
CACHE_ENABLED = True
memory = Memory("./.gbif_cache", verbose=0)

# Taxonomic mapping constants
SYNONYM_MAPPING = {
    'Archiannelida': 'Canalipalpata',
    'Chlamys': 'Zygochlamys',
    'Navicula': 'Diatom',
    'Diatom': 'Bacillariophyceae',
    'Diatoms': 'Bacillariophyceae',
    'Pennate diatom': 'Bacillariophyceae',
    'Pennate diatoms': 'Bacillariophyceae',
    'Discoid diatoms': 'Bacillariophyceae',
    'Downy woodpecker': 'Dryobates pubescens',
    'Unknown plant': 'Tracheophyta',
    'Dinoflagellate': 'Dinoflagellata',
    'Quadricoccus': 'Quatrionicoccus',
    'Antennaria sp': 'Antennariella',
    'Antennaria_sp': 'Antennariella',
    'Pecten': 'Bivalvia',
    'Dinoflagellata': 'Myzozoa',
    'Pyrocystis': 'Myzozoa',
    'Valonia': 'Valonia C.Agardh, 1823',
    'Cantharid': 'Insecta',
    'Flagellate': 'Pancryptista',
    'Flagellates': 'Pancryptista',
    'Syrphid': 'Insecta',
    'RODO': 'Rhodophyta',
    'Unknown teleost': 'Actinopterygii',
    'Diaptomid': 'Copepoda',
    'Kelp holdfast': 'Ochrophyta',
    'Orange encrusting sponge': 'Spirastrella cunctatrix',
    'Pisces': 'Actinopterygii',
    'Echinoderm': 'Echinodermata',
    'Copepodites': 'Copepoda',
    'Coccolithophorid': 'Coccolithophores',
    'Coccolithophore': 'Coccolithophores',
    'Flowering plants': 'Tracheophyta',
    'Moss': 'Bryophyta',
    'Unidentified chaetognaths': 'Chaetognatha',
#    'Acourtia': 'Acourtia D.Don',
    'Disintegrated fish-unidentifable': 'Actinopterygii',
    'Unspecifiable fish': 'Actinopterygii',
    'Unknown sponge': 'Porifera',
    'Real Weeds': 'Tracheophyta',
    'Filamentous red algae': 'Rhodophyta',
    'Bothus': 'Pleuronectiformes',
    'Hyattella': 'Porifera',
    'Argiope': 'Arachnida',
    'Unidentified mysid': 'Malacostraca',
    'Coelenterata': 'Cnidaria',
    'Microdictyon': 'Onychophora',
    'Young laminariales': 'Chromista',
    'Solenicola setigera': 'Chromista',
    'Caniformia': 'Mammalia',
    'RSFL': 'Aves',
    'ECDO': 'Aves',
    'YWAR': 'Aves',
    'ROSAACIC': 'Tracheophyta',
    'Unidentified Fish': 'Actinopterygii',
    'Unidentified medusae': 'Cnidaria',
    'Unidentified rotifer': 'Rotifera',
    'Sponqes undetermined': 'Porifera',
    'Spongia': 'Porifera',
    'Unknown tunicate': 'Tunicata',
    '': '',
    '': '',
}

BROAD_TAXON_MAP = {
    'Annelids': 'Annelida',
    'Birds': 'Aves',
    'Fish': 'Actinopterygii',
    'Mammals': 'Mammalia',
    'Marine mammals': 'Mammalia',
    'Molluscs': 'Mollusca',
    'Plants': 'Tracheophyta',
    'Bacteria': 'Bacteria',
    'Fungi': 'Fungi',
    'Tunicata': 'Tunicata',
}

AVES_TAXA = {'Uria', 'Sterna', 'Columba', 'Cygnus', 'Passer'} # , 'Chloris', 'Chlorurus', 
ACTINOPTERYGII_TAXA = {'Osmeriformes', 'Siluriformes', 'Perciformes', 'Tetraodontiformes', 'Scorpaeniformes', 'Myctophiformes', 'Cyprinodontiformes', 'Aulopiformes', 'Cypriniformes', 'Gadiformes', 'Lampriformes', 'Ophidiiformes', 'Characiformes', 'Gobiesociformes', 'Notacanthiformes', 'Pleuronectiformes', 'Beryciformes', 'Stomiiformes', 'Esociformes', 'Atheriniformes', 'Lophiiformes', 'Clupeiformes', 'Salmoniformes', 'Gasterosteiformes', 'Syngnathiformes', 'Anguilliformes', 'Zeiformes', 'Batrachoidiformes', 'Polymixiiformes', 'Beloniformes', 'Mugiliformes', 'Clupeiformes', 'Amiiformes', 'Cyprinidae', 'Aprion', 'Oncorhynchus', 'Galaxias', 'Scarus', 'Ammodytes', 'Teleostei', 'Percopsiformes', 'Kuhlia rupestris', 'Kuhlia', 'Prionotus', 'Leptocephali', 'Gibbonsia', 'Centropogon australis', 'Liparis', 'Chromis', 'Apogon', 'Naso', 'Acipenser oxyrinchus', 'Eurypharynx pelecanoides', 'Gonorynchiformes', 'Osteoglossiformes', 'Gymnotiformes', 'Saccopharyngiformes', 'Stephanoberyciformes', 'Elopiformes', 'Synbranchiformes', 'Cetomimiformes', 'Lepisosteiformes', 'Polypteriformes', 'Acipenseriformes'}
ELASMOBRANCHII_TAXA = {'Carcharhiniformes', 'Lamniformes', 'Orectolobiformes', 'Heterodontiformes', 'Hexanchiformes', 'Squaliformes', 'Squatiniformes', 'Pristiophoriformes', 'Rajiformes', 'Torpediniformes', 'Myliobatiformes', 'Rhinopristiformes', 'Raja'}
HOLOCEPHALI_TAXA = {'Chimaeriformes'}
CHONDRICHTHYES_TAXA = ELASMOBRANCHII_TAXA | HOLOCEPHALI_TAXA
HEMICHORDATA_TAXA = {'Enteropneusta'}
TUNICATA_TAXA = {'Ascidiacea', 'Thaliacea', 'Polycarpa', 'Appendicularia', 'Oikopleura', 'Oikopleuridae', 'Fritillaridae', 'Boltenia', 'Ascidiella', 'Copelata'}
ECHINODERMATA_TAXA = {'Asterias', 'Atelostomata', 'Phrynophiurida', 'Henricia', 'Asterozoa', 'Asterina', 'Ophiothrix', 'Asteroidea', 'Echinoidea', 'Holothuroidea'}
MOLLUSCA_TAXA = {'Patella', 'Bivalvia', 'Crepidula', 'Rissoa', 'Clione', 'Physa', 'Teuthida', 'Diplodonta', 'Prosobranchia', 'Teuthoidea', 'Lithopoma', 'Gymnosomata', 'Abra', 'Ervilia nitens', 'Pandora', 'Lucina', 'Spiratella', 'Lyonsia', 'Patellogastropoda', 'Opisthobranchia', 'Galba', 'Gastropoda', 'Cephalopoda'}
POLYCHAETA_TAXA = {'Spirobranchus', 'Amphitrite', 'Terebellidae', 'Glycera', 'Nereis', 'Spio', 'Chone', 'Exogone', 'Euclymeninae', 'Euclymeninae', 'Eteone', 'Pectinaria', 'Pholoe'}
SIPUNCULA_TAXA = {'Sipunculidea', 'Aspidosiphoniformes'}
ANNELIDA_TAXA = POLYCHAETA_TAXA | SIPUNCULA_TAXA | {'Halicryptus spinulosus', 'Hirudinea', 'Polychaeta', 'Sipuncula'}
NEMERTEA_TAXA = {'Nemertinea', 'Lineidae', 'Tubulanus', 'Rhynchocoela'}
PHORONIDA_TAXA = {'Phoronis', 'Phoronida'}
PLATYHELMINTHES_TAXA = {'Turbellaria', 'Girardia'}
INSECTA_TAXA = {'Enallagma', 'Homarus', 'Conocephalus', 'Rhyacophila', 'Tethea', 'Silphini', 'Colymbetes', 'Hydropsyche', 'Elachista', 'Sialis', 'Colias', 'Athetini', 'Leptocera', 'Calopteryx', 'Gerris', 'Lestes', 'Zygoptera', 'Tanypodinae'} # 'Acourtia', 
HEXAPODA_TAXA = INSECTA_TAXA | {'Collembola', 'Insecta'}
BRANCHIOPODA_TAXA = {'Diplostraca', 'Daphnia', 'Cladocera', 'Bosmina', 'Chydorus', 'Diaphanosoma', 'Evadne', 'Alona', 'Laevicaudata'} # , 'Sida'
MALACOSTRACA_TAXA = {'Isopoda', 'Homarus', 'Cumella', 'Hyas', 'Dendrobranchiata', 'Sabinea septemcarinata', 'Sabinea', 'Apseudes', 'Parthenope', 'Caprella', 'Galathea','Callianassa', 'Parthenope longispinus', 'Hyale', 'Sergia', 'Gammarus salinus', 'Nematoscelis', 'Parthenope hoplonotus', 'Calappa'}
COPEPODA_TAXA = {'Oithona', 'Oncaea', 'Calanus', 'Cyclops', 'Poecilostomatoida', 'Clytemnestra', 'Labidocera'}
MAXILLOPODA_TAXA= COPEPODA_TAXA | {'Maxillopoda', 'Copepoda'}
THECOSTRACA_TAXA = {'Austrominius', 'Balanus', 'Cirripedia', 'Cirripedia cypris'}
OLIGOSTRACA_TAXA = {'Ostracoda', 'Sessilia', 'Cirripedia'}
ARACHNIDA_TAXA = {'Pycnogonida', 'Acari', 'Pantopoda', 'Hydrachnidae', 'Lebertia', 'Hygrobates', 'Erigone atra'}
CHELICERATA_TAXA = ARACHNIDA_TAXA | {'Merostomata', 'Arachnida'}
CNIDARIA_TAXA = {'Anthozoa', 'Hydrozoa', 'Scyphozoa', 'Sertularia', 'Obelia', 'Acropora', 'Physonect', 'Vogtia', 'Edwardsia', 'Siphonophore', 'Octocorallia', 'Ceriantharia'}
BRYOZOA_TAXA = {'Gymnolaemata', 'Stenolaemata', 'Cellaria', 'Bugula', 'Alcyonidium albidum', 'Alcyonidium', 'Margaretta'}
PORIFERA_TAXA = {'Demospongiae', 'Cliona', 'Axinella'}
ROTIFERA_TAXA = {'Rotifera', 'Eurotatoria', 'Synchaeta', 'Keratella', 'Polyarthra', 'Trichocerca', 'Asplanchna', 'Kellicottia'}
BACILLARIOPHYCEAE_TAXA = {'Nitzschia', 'Navicula', 'Amphora', 'Cerataulina','Bacillariophyceae'}
TRACHEOPHYTA_TAXA = {'Plantae', 'Asteriscus', 'Fritillaria', 'Saggitaria', 'Orthosia', 'Blepharis', 'Canthidium', 'Limonia', 'Lesquerella', 'Digitaria', 'Pectis', 'Astragalus', 'Mertensia', 'Aglaia cucullata', 'Peridiaceae'}
BRYOPHYTA_TAXA = {'Bryophyta', 'Verrucaria', 'Anthocerotophyta', 'Bryopsida'}
CHLOROPHYTA_TAXA = {'Cladophora', 'Ulva', 'Mougeotia', 'Carteria', 'Schroederia', 'Neomeris', 'Ulothrix', 'Eudorina'} # , 'Kirchneriella'
STREPTOPHYTA_TAXA = {'Charophyta'}
CHLOROPLASTIDA_TAXA = CHLOROPHYTA_TAXA | STREPTOPHYTA_TAXA | {'Prasinodermatophyta', 'Chlorophyta', 'Streptophyta'}
RHODOPHYTA_TAXA = {'Polysiphonia', 'Ceramium', 'Porphyra', 'Cryptopleura', 'Lobophora', 'Dictyota', 'Jania', 'Gracilaria', 'Amphiroa'}
CRYPTOPHYTA_TAXA = {'Leucocryptos', 'Plagioselmis', 'Hemiselmis', 'Cryptomonadales', 'Rhodomonas', 'Teleaulax', 'Cryptophyta'}
OCHROPHYTA_TAXA = BACILLARIOPHYCEAE_TAXA | {'Amphora', 'Laminaria', 'Cerataulina', 'Meuniera', 'Actinocyclus', 'Odontella', 'Grammatophora', 'Anaulus', 'Bacillariophyceae'} # , 'Rossia'
RHIZARIA_TAXA = {'Halyphysema', 'Foraminifera', 'Rotaliida', 'Globothalamea', 'Astrorhizida', 'Radiozoa', 'Ebria tripartita', 'Arthracanthida'} # 'Circus',
EUGLENOZOA_TAXA = {'Euglena', 'Phacus', 'Eutreptiella', 'Trachelomonas'}
CYANOBACTERIA_TAXA = {'Anabaena', 'Trichodesmium'}
BACTERIA_TAXA = CYANOBACTERIA_TAXA | {'Bacteria', 'Cyanobacteria'}
MYRIAPODA_TAXA = {'Spirobolida', 'Scolopendromorpha', 'Lithobiomorpha'}
PANCRYPTISTA_TAXA = {'Cryptomonas'}
FUNGI_TAXA = {'Ascomycota', 'Basidiomycota'}
MAMMALIA_TAXA = {'Pipistrellus'}
LISSAMPHIBIA_TAXA = {'Amphibia'}
CHAETOGNATHA_TAXA = {'Sagitta'}
AGNATHA_TAXA = {'Petromyzonti', 'Myxini'}
CEPHALOCHORDATA_TAXA = {'Branchiostoma'}

TAXONOMIC_ORDER = [
    'Aves', 'Testudines', 'Crocodylia', 'Squamata', 'Sphenodontia', 'Mammalia', 'Lissamphibia', 'Actinopterygii', 'Chondrichthyes', 'Agnatha', 'Cephalochordata', 'Tunicata', 'Hemichordata', 'Echinodermata', 'Chaetognatha', 'Mollusca', 'Annelida', 'Nemertea', 'Brachiopoda', 'Phoronida', 'Platyhelminthes', 'Nematoda', 'Hexapoda', 'Remipedia', 'Branchiopoda', 'Cephalocarida', 'Malacostraca', 'Maxillopoda', 'Thecostraca', 'Oligostraca', 'Myriapoda', 'Chelicerata', 'Onychophora', 'Kinorhyncha', 'Cnidaria', 'Ctenophora', 'Bryozoa', 'Porifera', 'Rotifera', 'Choanoflagellatea', 'Fungi', 'Tracheophyta', 'Bryophyta', 'Chloroplastida', 'Rhodophyta', 'Cryptophyta', 'Haptophyta', 'Ochrophyta', 'Pancryptista', 'Rhizaria', 'Myzozoa', 'Euglenozoa', 'Ciliophora', 'Bacteria', 'Other'
]

def generate_collapse_mapping():
    mapping = {}
    for group in TAXONOMIC_ORDER:
        mapping[group] = group
        set_name = group.upper().replace(' ', '_') + '_TAXA'
        taxa_set = globals().get(set_name)
        if taxa_set and isinstance(taxa_set, set):
            for taxon in taxa_set:
                mapping[taxon] = group
    for group in TAXONOMIC_ORDER:
        set_name_var = group.upper().replace(' ', '_') + '_TAXA'
        if set_name_var in globals() and isinstance(globals()[set_name_var], set):
            base = set_name_var.replace('_TAXA', '')
            title_base = base.title()
            mapping[title_base] = group
    return mapping

COLLAPSE_MAPPING = generate_collapse_mapping()
COLLAPSE_MAPPING_NORMALIZED = {k.lower(): v for k, v in COLLAPSE_MAPPING.items()}

for group in TAXONOMIC_ORDER:
    COLLAPSE_MAPPING[group] = group

def create_biotime_species_count():
    os.makedirs('csv', exist_ok=True)
    df = pd.read_csv('data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv')
    species_counts = df['valid_name'].value_counts().reset_index()
    species_counts.columns = ['species', 'count']
    species_counts.to_csv('csv/biotime_species_count.csv', index=False)
    print(f"Created biotime_species_count.csv with {len(species_counts)} unique species")

def parse_quarter(q):
    """
    Parses a Quarter entry (e.g., "2/3" or "4/1") and returns an integer 1-4.
    If q is NA, returns None.
    """
    if pd.isna(q):
        return None
    try:
        return int(q)
    except ValueError:
        parts = str(q).split('/')
        if len(parts) == 2:
            try:
                q1, q2 = map(int, parts)
                avg = (q1 + q2) / 2.0
                return int(np.clip(np.rint(avg), 1, 4))
            except ValueError:
                return None
        return None

def load_biotime_data(filepath, top_n=999999):
    os.makedirs('csv', exist_ok=True)
    species_count_file = 'csv/biotime_species_count.csv'
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
        except:
            df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    taxon_mapping = {}
    if 'TAXON' in df.columns:
        taxon_mapping = df[['VALID_NAME', 'TAXON']].dropna().set_index('VALID_NAME')['TAXON'].to_dict()
        taxon_mapping = {k.lower(): v for k, v in taxon_mapping.items()}
    required = ["YEAR", "ABUNDANCE", "VALID_NAME", "STUDY_ID", "LATITUDE", "LONGITUDE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in BioTIME data: {missing}")
    df = df.dropna(subset=["YEAR", "ABUNDANCE"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    time_val = df["YEAR"].copy()
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        mask_month = df["MONTH"].notna()
        time_val.loc[mask_month] = df.loc[mask_month, "YEAR"] + (df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
        if "DAY" in df.columns:
            df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce")
            mask_day = df["DAY"].notna()
            mask = mask_month & mask_day
            time_val.loc[mask] += (df.loc[mask, "DAY"].astype(int) - 1) / 365.0
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["VALID_NAME"]
    taxon_mapping_file = 'csv/taxon_mapping.csv'
    pd.DataFrame(list(taxon_mapping.items()), columns=['species', 'taxon']).to_csv(taxon_mapping_file, index=False)
    top_species_file = f'csv/biotime_top{top_n}.csv'
    if os.path.exists(species_count_file) and os.path.exists(top_species_file):
        print("Loading preprocessed taxonomic information...")
        species_count = pd.read_csv(species_count_file)
        try:
            top_species_df = pd.read_csv(top_species_file)
            if 'species' in top_species_df.columns:
                top_species = top_species_df['species'].tolist()
            elif 'SPECIES_ID' in top_species_df.columns:
                top_species = top_species_df['SPECIES_ID'].tolist()
            else:
                raise KeyError("No species column found")
        except (KeyError, Exception):
            print("Top species file format incorrect, reprocessing...")
            species_count = df['SPECIES_ID'].value_counts().reset_index()
            species_count.columns = ['species', 'count']
            top_species = species_count.nlargest(top_n, 'count')['species'].tolist()
            species_count.to_csv(species_count_file, index=False)
            pd.DataFrame({'species': top_species}).to_csv(top_species_file, index=False)
    else:
        print("Preprocessing taxonomic information...")
        species_count = df['SPECIES_ID'].value_counts().reset_index()
        species_count.columns = ['species', 'count']
        top_species = species_count.nlargest(top_n, 'count')['species'].tolist()
        species_count.to_csv(species_count_file, index=False)
        pd.DataFrame({'species': top_species}).to_csv(top_species_file, index=False)
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE"]], top_species, species_count

def load_rivfishtime_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df.columns = df.columns.str.strip().str.upper()
    required = ["YEAR", "SPECIES", "ABUNDANCE"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing RivFishTIME columns: {missing}")
    df = df.dropna(subset=required)
    time_val = df["YEAR"].copy()
    if "QUARTER" in df.columns:
        parsed_q = df["QUARTER"].apply(parse_quarter).fillna(2.5)
        time_val += (parsed_q - 1) / 4.0
    df["TIME"] = time_val
    df["SPECIES_ID"] = df["SPECIES"]
    df["ABUNDANCE"] = pd.to_numeric(df["ABUNDANCE"], errors='coerce')
    result_df = df.groupby(["SPECIES_ID", "TIME"], as_index=False).agg({"ABUNDANCE": "sum"})
    return result_df

def find_column(df, synonyms):
    for col in df.columns:
        col_clean = col.strip().lower()
        for syn in synonyms:
            if col_clean == syn.strip().lower():
                return col
    return None

def load_cedar_creek_data(e120_plant_path, e120_arthropod_path, e244_path):
    cedar_creek_dfs = []
    print("Loading E120 Plant Biomass data...")
    try:
        e120_plant_df = pd.read_csv(e120_plant_path)
        year_col = find_column(e120_plant_df, ['year', 'YEAR', 'Year'])
        biomass_col = find_column(e120_plant_df, ['biomass (g/m2)', 'BIOMASS (G/M2)', 'Biomass (g/m2)', 'biomass', 'BIOMASS', 'Biomass', 'mass', 'MASS', 'Mass'])
        species_col = find_column(e120_plant_df, ['species', 'SPECIES', 'Species'])
        missing = []
        if not year_col: missing.append('YEAR')
        if not biomass_col: missing.append('BIOMASS (G/M2)')
        if not species_col: missing.append('SPECIES')
        if missing:
            print(f"Missing columns in E120 plant data: {missing}")
        else:
            e120_plant_df.rename(columns={year_col: 'YEAR', biomass_col: 'BIOMASS (G/M2)', species_col: 'SPECIES'}, inplace=True)
            e120_plant_df = e120_plant_df.dropna(subset=["YEAR", "BIOMASS (G/M2)"])
            e120_plant_df["YEAR"] = pd.to_numeric(e120_plant_df["YEAR"], errors="coerce")
            e120_plant_df["SPECIES"] = e120_plant_df["SPECIES"].str.strip()
            e120_plant_df["SPECIES"] = e120_plant_df["SPECIES"].replace({
                "Miscellaneous*": "Other", "Unsorted*": "Other", "Green matter*": "Other", "Achillea millefolium(lanulosa)": "Achillea millefolium", "Miscellaneous litter": "Other", "Miscellaneous Litter": "Other"
            })
            time_val = e120_plant_df["YEAR"].copy()
            if "MONTH" in e120_plant_df.columns:
                e120_plant_df["MONTH"] = pd.to_numeric(e120_plant_df["MONTH"], errors="coerce")
                mask_month = e120_plant_df["MONTH"].notna()
                time_val.loc[mask_month] = e120_plant_df.loc[mask_month, "YEAR"] + (e120_plant_df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
            e120_plant_df["TIME"] = time_val
            e120_plant_df["SPECIES_ID"] = e120_plant_df["SPECIES"]
            e120_plant_df["ABUNDANCE"] = pd.to_numeric(e120_plant_df["BIOMASS (G/M2)"], errors='coerce')
            cedar_creek_dfs.append(e120_plant_df[["SPECIES_ID", "TIME", "ABUNDANCE"]])
            print(f"E120 Plant Biomass: {len(e120_plant_df)} records")
    except Exception as e:
        print(f"Error loading E120 Plant Biomass: {e}")
    print("Loading E120 Arthropod data...")
    try:
        e120_arthropod_df = pd.read_csv(e120_arthropod_path, sep='\t')
        year_col = find_column(e120_arthropod_df, ['year', 'YEAR', 'Year'])
        count_col = find_column(e120_arthropod_df, ['count', 'COUNT', 'Count'])
        genus_col = find_column(e120_arthropod_df, ['genus', 'GENUS', 'Genus'])
        epithet_col = find_column(e120_arthropod_df, ['specific.epithet', 'SPECIFIC.EPITHET', 'Specific.epithet', 'specificepithet', 'specific epithet'])
        missing = []
        if not year_col: missing.append('YEAR')
        if not count_col: missing.append('COUNT')
        if not genus_col: missing.append('GENUS')
        if not epithet_col: missing.append('SPECIFIC.EPITHET')
        if missing:
            print(f"Missing columns in E120 arthropod data: {missing}")
        else:
            e120_arthropod_df.rename(columns={year_col: 'YEAR', count_col: 'COUNT', genus_col: 'GENUS', epithet_col: 'SPECIFIC.EPITHET'}, inplace=True)
            e120_arthropod_df = e120_arthropod_df.dropna(subset=["YEAR", "COUNT"])
            e120_arthropod_df["YEAR"] = pd.to_numeric(e120_arthropod_df["YEAR"], errors="coerce")
            e120_arthropod_df["SPECIES"] = e120_arthropod_df["GENUS"] + " " + e120_arthropod_df["SPECIFIC.EPITHET"]
            e120_arthropod_df["SPECIES"] = e120_arthropod_df["SPECIES"].replace({"undet*": "Other"})
            time_val = e120_arthropod_df["YEAR"].copy()
            if "MONTH" in e120_arthropod_df.columns:
                e120_arthropod_df["MONTH"] = pd.to_numeric(e120_arthropod_df["MONTH"], errors="coerce")
                mask_month = e120_arthropod_df["MONTH"].notna()
                time_val.loc[mask_month] = e120_arthropod_df.loc[mask_month, "YEAR"] + (e120_arthropod_df.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
            e120_arthropod_df["TIME"] = time_val
            e120_arthropod_df["SPECIES_ID"] = e120_arthropod_df["SPECIES"]
            e120_arthropod_df["ABUNDANCE"] = pd.to_numeric(e120_arthropod_df["COUNT"], errors='coerce')
            cedar_creek_dfs.append(e120_arthropod_df[["SPECIES_ID", "TIME", "ABUNDANCE"]])
            print(f"E120 Arthropod: {len(e120_arthropod_df)} records")
    except Exception as e:
        print(f"Error loading E120 Arthropod: {e}")
    print("Loading E244 Biomass data...")
    try:
        e244_df = pd.read_csv(e244_path)
        year_col = find_column(e244_df, ['year', 'YEAR', 'Year'])
        # Expanded synonyms for mass column to catch parentheses variants
        mass_col = find_column(e244_df, [
            'mass g/m^2', 'MASS G/M^2', 'Mass g/m^2',
            'mass g/m2', 'MASS G/M2', 'Mass g/m2',
            'mass (g/m^2)', 'MASS (G/M^2)', 'Mass (g/m^2)',
            'mass (g/m2)', 'MASS (G/M2)', 'Mass (g/m2)',
            'biomass g/m^2', 'BIOMASS G/M^2', 'Biomass g/m^2',
            'biomass g/m2', 'BIOMASS G/M2', 'Biomass g/m2',
            'biomass (g/m^2)', 'BIOMASS (G/M^2)', 'Biomass (g/m^2)',
            'biomass (g/m2)', 'BIOMASS (G/M2)', 'Biomass (g/m2)',
            'mass', 'MASS', 'Mass', 'biomass', 'BIOMASS', 'Biomass'
        ])
        species_col = find_column(e244_df, ['species', 'SPECIES', 'Species'])
        missing = []
        if not year_col: missing.append('YEAR')
        if not mass_col: missing.append('MASS G/M^2)')
        if not species_col: missing.append('SPECIES')
        if missing:
            print(f"Missing columns in E244 data: {missing}")
        else:
            e244_df.rename(columns={year_col: 'YEAR', mass_col: 'MASS G/M^2)', species_col: 'SPECIES'}, inplace=True)
            e244_df = e244_df.dropna(subset=["YEAR", "MASS G/M^2)"])
            e244_df["YEAR"] = pd.to_numeric(e244_df["YEAR"], errors="coerce")
            e244_df["SPECIES"] = e244_df["SPECIES"].str.strip()
            e244_df["SPECIES"] = e244_df["SPECIES"].replace({
                "Achillea millefolium(lanulosa)": "Achillea millefolium", "Miscellaneous litter": "Other", "Miscellaneous Litter": "Other"
            })
            time_val = e244_df["YEAR"].copy()
            if "DATE" in e244_df.columns:
                try:
                    dates = pd.to_datetime(e244_df["DATE"])
                    time_val = dates.dt.year + (dates.dt.month - 1) / 12.0 + (dates.dt.day - 1) / 365.0
                except:
                    pass
            e244_df["TIME"] = time_val
            e244_df["SPECIES_ID"] = e244_df["SPECIES"]
            e244_df["ABUNDANCE"] = pd.to_numeric(e244_df["MASS G/M^2)"], errors='coerce')
            cedar_creek_dfs.append(e244_df[["SPECIES_ID", "TIME", "ABUNDANCE"]])
            print(f"E244 Biomass: {len(e244_df)} records")
    except Exception as e:
        print(f"Error loading E244 Biomass: {e}")
    if cedar_creek_dfs:
        combined_df = pd.concat(cedar_creek_dfs, ignore_index=True)
        print(f"Combined Cedar Creek data: {len(combined_df)} total records")
        return combined_df
    else:
        print("No Cedar Creek data could be loaded")
        return pd.DataFrame()

def create_species_count_file(dataset_name, data, min_records=100):
    os.makedirs('csv', exist_ok=True)
    species_counts = data['SPECIES_ID'].value_counts().reset_index()
    species_counts.columns = ['species', 'count']
    species_min_n = species_counts[species_counts['count'] >= min_records]
    count_file = f'csv/{dataset_name}_species_count.csv'
    min_n_file = f'csv/{dataset_name}_species_min_n.csv'
    species_counts.to_csv(count_file, index=False)
    species_min_n[['species']].to_csv(min_n_file, index=False)
    print(f"{dataset_name}: {len(species_min_n)} species with >= {min_records} records")
    return species_min_n['species'].tolist()

def load_combined_data(biotime_path, rivfishtime_path=None, cedar_creek_paths=None, min_records=100, use_existing_biotime=True):
    """
    Load and combine data from BioTIME, RivFishTIME, and Cedar Creek datasets.
    Only includes species with at least min_records data points from each dataset.
    """
    print("Loading combined datasets...")
    # Load BioTIME data
    if use_existing_biotime and os.path.exists('csv/biotime_species_count.csv'):
        print("Using existing BioTIME preprocessing...")
        biotime_data, _, _ = load_biotime_data(biotime_path, top_n=999999)  # Get all species
    else:
        biotime_data, _, _ = load_biotime_data(biotime_path, top_n=999999)
    biotime_data['dataset'] = 'BioTIME' # Add dataset column for BioTIME
    biotime_min_n = create_species_count_file('biotime', biotime_data, min_records)
    biotime_species_set = set(biotime_min_n)
    print(f"BioTIME: {len(biotime_data)} records, {len(biotime_min_n)} species with >= {min_records} records")
    combined_data = biotime_data[biotime_data['SPECIES_ID'].isin(biotime_min_n)].copy()
    all_species_set = set(biotime_min_n)
    # Load RivFishTIME data if provided
    if rivfishtime_path and os.path.exists(rivfishtime_path):
        print("Loading RivFishTIME data...")
        rivfishtime_data = load_rivfishtime_data(rivfishtime_path)
        if not rivfishtime_data.empty:
            rivfishtime_data['dataset'] = 'RivFishTIME' # Add dataset column for RivFishTIME
            rivfishtime_min_n = create_species_count_file('rivfishtime', rivfishtime_data, min_records)
            rivfishtime_species_set = set(rivfishtime_min_n)
            unique_rivfishtime_species = rivfishtime_species_set - biotime_species_set
            rivfishtime_data_filtered = rivfishtime_data[rivfishtime_data['SPECIES_ID'].isin(unique_rivfishtime_species)]
            rivfishtime_data_filtered['STUDY_ID'] = 'RivFishTIME'
            rivfishtime_data_filtered['LATITUDE'] = None
            rivfishtime_data_filtered['LONGITUDE'] = None
            print(f"RivFishTIME: {len(rivfishtime_data_filtered)} records, {len(unique_rivfishtime_species)} unique species with >= {min_records} records (after removing {len(rivfishtime_species_set - unique_rivfishtime_species)} overlaps with BioTIME)")
            combined_data = pd.concat([combined_data, rivfishtime_data_filtered], ignore_index=True)
            all_species_set.update(unique_rivfishtime_species)
    else:
        print("RivFishTIME path not provided or file not found, skipping...")
    # Load Cedar Creek data if provided
    if cedar_creek_paths is not None:
        print("Loading Cedar Creek data...")
        e120_plant_path = cedar_creek_paths.get('e120_plant')
        e120_arthropod_path = cedar_creek_paths.get('e120_arthropod')
        e244_path = cedar_creek_paths.get('e244')
        cedar_creek_data = load_cedar_creek_data(e120_plant_path, e120_arthropod_path, e244_path)
        if not cedar_creek_data.empty:
            cedar_creek_data['dataset'] = 'CedarCreek' # Add dataset column for Cedar Creek
            cedar_creek_min_n = create_species_count_file('cedar_creek', cedar_creek_data, min_records)
            cedar_creek_species_set = set(cedar_creek_min_n)
            cedar_creek_data['STUDY_ID'] = 'CedarCreek'
            cedar_creek_data['LATITUDE'] = None
            cedar_creek_data['LONGITUDE'] = None
            print(f"Cedar Creek: {len(cedar_creek_data)} records, {len(cedar_creek_min_n)} species with >= {min_records} records")
            combined_data = pd.concat([combined_data, cedar_creek_data], ignore_index=True)
            all_species_set.update(cedar_creek_species_set)
    else:
        print("Cedar Creek paths not provided, skipping...")
    all_species_list = list(all_species_set) # Create combined species list with min_records
    combined_species_file = f'csv/combined_species_min{min_records}.csv' # Save combined species list
    pd.DataFrame({'species': all_species_list}).to_csv(combined_species_file, index=False)
    print(f"Combined dataset: {len(combined_data)} total records, {len(all_species_list)} total species with >= {min_records} records")
    return combined_data, all_species_list, None

def collapse_taxonomic_group(group):
    if group in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[group]
    return COLLAPSE_MAPPING.get(group, 'Other')

def get_cached_taxonomic_group(species_name, taxon_mapping=None, force_recompute=False):
    cache_file = 'csv/taxonomic_cache.csv'
    os.makedirs('csv', exist_ok=True)
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return 'Other'
    species_str = str(species_name).strip()
    if 'miscellaneous' in species_str.lower() or 'litter' in species_str.lower() or 'unknown' in species_str.lower():
        return 'Other'
    if taxon_mapping is None:
        taxon_mapping = {}
        if os.path.exists('csv/taxon_mapping.csv') and os.path.getsize('csv/taxon_mapping.csv') > 0:
            try:
                taxon_df = pd.read_csv('csv/taxon_mapping.csv')
                taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
            except Exception as e:
                print(f"Error reading taxon_mapping.csv: {e}")
                taxon_mapping = {}
    if not force_recompute and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            cache_df = pd.read_csv(cache_file)
            if not cache_df.empty and 'species' in cache_df.columns and 'taxonomic_group' in cache_df.columns:
                cached = cache_df[cache_df['species'] == species_name]
                if not cached.empty:
                    cached_group = cached.iloc[0]['taxonomic_group']
                    if cached_group != 'Other':
                        return cached_group
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as e:
            print(f"Warning: Cache file is corrupted, recreating it. Error: {e}")
            cache_df = pd.DataFrame(columns=['species', 'taxonomic_group'])
            cache_df.to_csv(cache_file, index=False)
    species_lower = species_name.lower()
    if species_lower in taxon_mapping:
        taxon_value = taxon_mapping[species_lower]
        taxonomic_group = collapse_taxonomic_group(taxon_value)
        if taxonomic_group != 'Other':
            try:
                if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
                    try:
                        cache_df = pd.read_csv(cache_file)
                        cache_df = cache_df[cache_df['species'] != species_name]
                        new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        cache_df.to_csv(cache_file, index=False)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError):
                        cache_df = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                        cache_df.to_csv(cache_file, index=False)
                else:
                    cache_df = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                    cache_df.to_csv(cache_file, index=False)
                return taxonomic_group
            except Exception as e:
                print(f"Error saving to cache for {species_name}: {e}")
    taxonomic_group = classify_taxonomic_group(species_name, taxon_mapping)
    try:
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            try:
                cache_df = pd.read_csv(cache_file)
                cache_df = cache_df[cache_df['species'] != species_name]
                new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                cache_df.to_csv(cache_file, index=False)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                cache_df = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                cache_df.to_csv(cache_file, index=False)
        else:
            cache_df = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
            cache_df.to_csv(cache_file, index=False)
    except Exception as e:
        print(f"Error saving to cache for {species_name}: {e}")
    return taxonomic_group

def apply_synonym_mapping(name):
    if not name or name == 'Other': # Handle empty names
        return 'Other'
    if name in SYNONYM_MAPPING: # First try exact match
        return SYNONYM_MAPPING[name]
    for key in SYNONYM_MAPPING: # Handle multi-word common names
        if name.lower() == key.lower():
            return SYNONYM_MAPPING[key]
    # Handle "Miscellaneous litter" and similar cases
    if 'miscellaneous' in name.lower() or 'litter' in name.lower() or 'unknown' in name.lower():
        return 'Other'
    genus = name.split()[0] if name.split() else name # Then try genus-level mapping
    if genus in SYNONYM_MAPPING:
        return SYNONYM_MAPPING[genus] + ' ' + ' '.join(name.split()[1:])
    return name

def clean_taxonomic_groups(name):
    if pd.isnull(name) or not isinstance(name, str) or name.strip() == '':
        return 'Other'
    # Handle "Miscellaneous litter" and similar cases
    cleaned = name.strip()
    if 'miscellaneous' in cleaned.lower() or 'litter' in cleaned.lower() or 'unknown' in cleaned.lower():
        return 'Other'
    cleaned = name.replace('_', ' ').strip()
    cleaned = re.sub(r'\s*sp(\.|\d+).*$', '', cleaned, flags=re.IGNORECASE) # Remove sp. designations and numbers
    cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned) # Remove author/year in parentheses
    cleaned = re.sub(r'[\d_]+$', '', cleaned).strip() # Remove trailing numbers and special characters
    if cleaned == '': # If empty after cleaning, return 'Other'
        return 'Other'
    return cleaned

@memory.cache
def get_taxonomic_info(scientific_name, depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return None
    base_name = apply_synonym_mapping(scientific_name)
    clean_name = clean_taxonomic_groups(base_name)
    if not clean_name or not clean_name.strip(): # Handle empty clean_name
        return None
    search_strategies = [] # Try different search strategies
    clean_words = clean_name.split() # Check if clean_name has words before splitting
    if len(clean_words) > 1:
        search_strategies.append({'name': clean_name, 'rank': 'SPECIES'})
    search_strategies.append({'name': clean_name})
    if len(clean_words) > 0: # Only use genus if we have at least one word
        genus = clean_words[0]
        search_strategies.append({'name': genus, 'rank': 'GENUS'})
        search_strategies.append({'name': genus, 'rank': 'FAMILY'})
        search_strategies.append({'name': genus, 'rank': 'ORDER'})
        search_strategies.append({'name': genus, 'rank': 'CLASS'})
        search_strategies.append({'name': genus, 'rank': 'PHYLUM'})
    search_strategies.append({'name': clean_name, 'rank': None})
    for params in search_strategies: # Try all search strategies with retry mechanism
        for attempt in range(3):  # Retry up to 3 times
            try:
                resp = requests.get(
                    "https://api.gbif.org/v1/species/match", params={'name': params['name'], 'rank': params.get('rank')}, timeout=30  # Increased timeout
                )
                data = resp.json()
                if data.get('matchType') in ['EXACT', 'FUZZY', 'HIGHERRANK'] and data.get('confidence', 0) >= 80: # Accept various match types with sufficient confidence
                    result = {
                        'order': data.get('order'), 'class': data.get('class'), 'subclass': data.get('subclass'), 'subphylum': data.get('subphylum'), 'phylum': data.get('phylum'), 'kingdom': data.get('kingdom'), }
                    higher_map = data.get('higherClassificationMap', {}) # Fill in missing ranks from higher classification
                    for rank in ['phylum', 'class', 'order', 'subphylum', 'family', 'kingdom']:
                        if not result.get(rank) and rank in higher_map:
                            result[rank] = higher_map[rank]
                    return result
                if data and 'higherClassificationMap' in data: # If we have higher classification, try parents recursively
                    for rank in ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']:
                        if rank in data.get('higherClassificationMap', {}):
                            parent_name = data['higherClassificationMap'][rank]
                            if parent_name and parent_name != scientific_name:
                                result = get_taxonomic_info(parent_name, depth+1)
                                if result:
                                    return result
                time.sleep(API_DELAY)
                break  # Exit retry loop if request was successful
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < 2:  # Not the last attempt
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Connection error, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to connect to GBIF API after 3 attempts: {str(e)}")
                    return None
            except Exception as e:
                print(f"Error querying GBIF for {params['name']}: {str(e)}")
                time.sleep(API_DELAY)
                break  # Exit retry loop for non-connection errors
    if len(clean_words) > 1 and len(clean_words) > 0: # Fallback to genus search if multi-word name and we have a genus
        return get_taxonomic_info(clean_words[0], depth+1)
    return None

def assign_taxonomic_group(tax_info, mapped_name):
    if mapped_name in TAXONOMIC_ORDER:
        return mapped_name
    if not tax_info:
        if mapped_name in COLLAPSE_MAPPING:
            return COLLAPSE_MAPPING[mapped_name]
        for part in mapped_name.split():
            if part in COLLAPSE_MAPPING:
                return COLLAPSE_MAPPING[part]
        return 'Other'
    if tax_info.get('class') and tax_info['class'] in COLLAPSE_MAPPING: # Check class first (most specific)
        return COLLAPSE_MAPPING[tax_info['class']]
    for rank in ['subclass', 'order', 'subphylum', 'phylum', 'kingdom']: # Check other taxonomic ranks
        taxon = tax_info.get(rank)
        if taxon and taxon in COLLAPSE_MAPPING:
            return COLLAPSE_MAPPING[taxon]
    for part in mapped_name.split(): # Check if any part of the name matches
        if part in COLLAPSE_MAPPING:
            return COLLAPSE_MAPPING[part]
    return 'Other'

def update_taxonomic_cache_from_other():
    cache_file = 'csv/taxonomic_cache.csv'
    if not os.path.exists(cache_file):
        print("No taxonomic cache found, skipping update.")
        return
    try:
        cache_df = pd.read_csv(cache_file)
        if 'taxonomic_group' not in cache_df.columns or 'species' not in cache_df.columns:
            print("Cache file missing required columns, skipping update.")
            return
        other_mask = cache_df['taxonomic_group'] == 'Other'
        if not other_mask.any():
            print("No 'Other' species to update.")
            return
        other_species = cache_df.loc[other_mask, 'species'].tolist()
        print(f"Found {len(other_species)} species with 'Other' classification. Updating...")
        taxon_mapping = {}
        if os.path.exists('csv/taxon_mapping.csv'):
            taxon_df = pd.read_csv('csv/taxon_mapping.csv')
            taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
        updated_count = 0
        for species in tqdm(other_species, desc="Updating 'Other' species"):
            new_group = get_cached_taxonomic_group(species, taxon_mapping=taxon_mapping, force_recompute=True)
            if new_group != 'Other':
                updated_count += 1
        print(f"Taxonomic cache updated: {updated_count} species reclassified.")
    except Exception as e:
        print(f"Error updating taxonomic cache: {e}")

def classify_taxonomic_group(species_name, taxon_mapping=None):
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return 'Other'
    species_str = str(species_name).strip()
    if 'miscellaneous' in species_str.lower() or 'litter' in species_str.lower() or 'unknown' in species_str.lower():
        return 'Other'
    clean_name = clean_taxonomic_groups(species_name)
    mapped_name = apply_synonym_mapping(clean_name)
    if mapped_name in TAXONOMIC_ORDER:
        return mapped_name
    if taxon_mapping is not None:
        species_lower = species_name.lower()
        if species_lower in taxon_mapping:
            broad_taxon = taxon_mapping[species_lower]
            if broad_taxon in BROAD_TAXON_MAP:
                mapped_group = BROAD_TAXON_MAP[broad_taxon]
                if mapped_group in TAXONOMIC_ORDER:
                    return mapped_group
    if mapped_name in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[mapped_name]
    for part in mapped_name.split():
        if part in COLLAPSE_MAPPING:
            return COLLAPSE_MAPPING[part]
    tax_info = get_taxonomic_info(clean_name)
    if not tax_info and mapped_name != clean_name:
        tax_info = get_taxonomic_info(mapped_name)
    taxonomic_group = assign_taxonomic_group(tax_info, mapped_name)
    if taxonomic_group == 'Other':
        print(f"Species '{species_name}' classified as 'Other'. Taxonomic info: {tax_info}")
    return taxonomic_group

def set_plot_style():
    plt.rcParams.update({
        'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 18, 'axes.titlesize': 20, 'font.size': 20, 'lines.linewidth': 1.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 1.5
    })

def plot_histogram(data, filename):
    set_plot_style()
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    total = data.sum() # Calculate percentages
    percentages = (data / total) * 100
    sorted_models = sorted(data.index)
    cmap = plt.get_cmap('Set1')
    color_map = {model: cmap(i % 10) for i, model in enumerate(sorted_models)}
    colors = [color_map[model] for model in data.index]
    bars = ax.bar(data.index, percentages, color=colors, alpha=0.65)
    for bar, count in zip(bars, data.values): # Add absolute counts inside bars
        height = bar.get_height()
        if height > 0:  # Only label bars with visible height
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + height / 2
            ax.text(x_pos, y_pos, f'{int(count)}', ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    plt.xlabel(None)
    plt.ylabel('Percentage of Total (%)')
    x_labels = [str(label) for label in data.index] # Split x-labels into two lines to prevent overlapping
    split_labels = []
    for label in x_labels:
        words = label.split()
        if len(words) > 1:
            mid = len(words) // 2 # Split into approximately equal halves
            first_line = ' '.join(words[:mid])
            second_line = ' '.join(words[mid:])
            split_labels.append(first_line + '\n' + second_line)
        else:
            split_labels.append(label)
    plt.xticks(range(len(x_labels)), split_labels, ha='center')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_summary_figure(results_df):
    if 'tipping_point_type' not in results_df.columns or 'taxonomic_group' not in results_df.columns:
        print("Warning: Could not create summary figure - missing required columns")
        return
    plot_df = results_df.copy()
    # Convert 'error', 'null', and cases with no transition year to 'no bifurcation' for visualization only
    mask = (plot_df['tipping_point_type'].isin(['error', 'null'])) | (plot_df['transition_year'].isna())
    plot_df.loc[mask, 'tipping_point_type'] = 'no bifurcation'
    tipping_point_order = ['no bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    tipping_point_colors = {
        'no bifurcation': '#999999',  # grey
        'fold bifurcation': '#1f77b4',  # blue
        'transcritical bifurcation': '#ff7f0e',  # orange
        'Hopf bifurcation': '#9467bd'  # purple
    }
    summary = plot_df.groupby(['taxonomic_group', 'tipping_point_type']).size().unstack(fill_value=0)
    existing_columns = [col for col in tipping_point_order if col in summary.columns]
    summary = summary[existing_columns]
    summary = summary.reindex(TAXONOMIC_ORDER, fill_value=0)
    summary = summary.loc[summary.sum(axis=1) > 0]  # Remove groups with no data
    set_plot_style()
    plt.figure(figsize=(24, 10))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    row_totals = summary.sum(axis=1)
    bottom = np.zeros(len(summary))
    bars = []
    for i, col in enumerate(summary.columns):
        values = summary[col].values
        percentages = (values / row_totals.replace(0, np.nan)) * 100 # Calculate percentage relative to group total
        percentages = percentages.fillna(0).values
        color = tipping_point_colors.get(col, '#1f77b4')  # default to blue if not found
        bar = ax.bar(summary.index, percentages, bottom=bottom, color=color, alpha=0.65, label=col)
        bars.append(bar)
        bottom += percentages
    for i, bar_collection in enumerate(bars):
        for j, rect in enumerate(bar_collection):
            height = rect.get_height()
            if height > 0:
                raw_value = summary.iloc[j, i]
                int_value = int(raw_value)
                x_pos = rect.get_x() + rect.get_width() / 2
                y_pos = rect.get_y() + height / 2
                text_color = 'black'
                ax.text(x_pos, y_pos, f'{int_value}', ha='center', va='center', color=text_color, fontsize=14, fontweight='bold')
    plt.xlabel(None)
    plt.ylabel('Percentage within Group (%)')
    plt.xlim(-0.5, len(summary) - 0.5)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Tipping Point Type', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(False)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/tipping_point_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def adjust_series_size(series_df, max_size=1500): # Adjust time series to maximum size of 1500 points
    abundance = series_df['ABUNDANCE'].values
    years = series_df['YEAR'].values
    if len(abundance) < max_size:
        padded_abundance = np.zeros(max_size)
        padded_abundance[:len(abundance)] = abundance
        padded_years = np.zeros(max_size)
        padded_years[:len(years)] = years
        if len(years) > 0:
            padded_years[len(years):] = years[-1]
        return pd.DataFrame({'YEAR': padded_years, 'ABUNDANCE': padded_abundance})
    if len(abundance) == max_size:
        return pd.DataFrame({'YEAR': years, 'ABUNDANCE': abundance})
    non_zero_mask = abundance != 0
    non_zero_abundance = abundance[non_zero_mask]
    non_zero_years = years[non_zero_mask]
    if len(non_zero_abundance) <= max_size:
        if len(non_zero_abundance) < max_size:
            padded_abundance = np.zeros(max_size)
            padded_abundance[:len(non_zero_abundance)] = non_zero_abundance
            padded_years = np.zeros(max_size)
            padded_years[:len(non_zero_years)] = non_zero_years
            padded_years[len(non_zero_years):] = non_zero_years[-1] if len(non_zero_years) > 0 else 0
            return pd.DataFrame({'YEAR': padded_years, 'ABUNDANCE': padded_abundance})
        return pd.DataFrame({'YEAR': non_zero_years, 'ABUNDANCE': non_zero_abundance})
    step = len(non_zero_abundance) / max_size
    indices = [int(i * step) for i in range(max_size)]
    uniform_abundance = non_zero_abundance[indices]
    uniform_years = non_zero_years[indices]
    return pd.DataFrame({'YEAR': uniform_years, 'ABUNDANCE': uniform_abundance})

def score_change_points(change_points, data, method):
    if not change_points:
        return -np.inf
    n_points = len(change_points)
    if n_points == 1: # Perfect score for exactly one change point
        return 1000
    score = 0 # Calculate quality score based on mean differences
    score -= (n_points - 1) * 100 # Penalize having too many change points
    for cp in change_points: # Reward large mean differences around change points
        if 10 <= cp <= len(data) - 10:
            before_mean = np.mean(data[cp-10:cp])
            after_mean = np.mean(data[cp:cp+10])
            mean_diff = abs(after_mean - before_mean)
            score += mean_diff * 50
    # Method-specific bonuses/penalties
    if method == "Pelt":
        score += 5
    elif method == "Window":
        score += 10
    return score

def find_transition(series_df, species, variable="ABUNDANCE", plot_fig=False):
    points = series_df[variable].values
    years = series_df['YEAR'].values
    methods_results = []
    if len(points) < 10: # constant series detection
        transition_index = len(points) // 2
        transition_year = series_df['YEAR'].iloc[transition_index]
        print(f"Short series detected for {species}, using midpoint: {transition_year}")
        return transition_index
    if np.std(points) < 1e-10: # Check for constant or near-constant series
        transition_index = len(points) // 2
        transition_year = series_df['YEAR'].iloc[transition_index]
        print(f"Constant series detected for {species}, using midpoint: {transition_year}")
        return transition_index
    mean_val = np.mean(points) # Check for relative range (avoid division by zero)
    if mean_val > 0:
        relative_range = (np.max(points) - np.min(points)) / mean_val
        if relative_range < 0.01:  # 1% threshold for near-constant
            transition_index = len(points) // 2
            transition_year = series_df['YEAR'].iloc[transition_index]
            print(f"Near-constant series detected for {species}, using midpoint: {transition_year}")
            return transition_index
    # Try to find the perfect scenario: exactly one change point (excluding the last point)
    best_transition_index = None
    best_method = None
    best_params = None
    best_score = -np.inf
    # Method 1: Pelt with various parameters
    pelt_models = ["rbf", "l2", "linear", "normal", "ar"]
    penalties = [1, 2, 3, 5, 10]  # Different penalty values
    for model in pelt_models:
        for pen in penalties:
            try:
                algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(points)
                penalty_value = pen * np.log(len(points)) * np.var(points)
                bkps = algo.predict(pen=penalty_value)
                # Filter valid breakpoints (excluding boundaries and final point)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="Pelt") # Score this result
                if score > best_score: # Check if this is our best result so far
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "Pelt"
                    best_params = {"model": model, "penalty": pen}
                if len(valid_bkps) == 1: # If we found exactly one change point, we might stop here
                    print(f"Perfect single change point found with Pelt ({model}, pen={pen}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"Pelt_{model}_pen{pen}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    # Method 2: Window with various parameters
    window_sizes = [20, 30, 40, 50, 60]
    n_bkps_range = [1, 2, 3, 4, 5]  # Try different numbers of breakpoints
    for window_size in window_sizes:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Window(width=window_size, model="l2").fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="Window") # Score this result
                if score > best_score: # Check if this is our best result so far
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "Window"
                    best_params = {"width": window_size, "n_bkps": n_bkps}
                if len(valid_bkps) == 1: # If we found exactly one change point
                    print(f"Perfect single change point found with Window (width={window_size}, n_bkps={n_bkps}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"Window_width{window_size}_nbps{n_bkps}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    # Method 3: Binary Segmentation with various parameters
    binseg_models = ["l2", "rbf", "linear", "normal", "ar"]
    for model in binseg_models:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Binseg(model=model, min_size=5).fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="BinSeg") # Score this result
                if score > best_score: # Check if this is our best result so far
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "BinSeg"
                    best_params = {"model": model, "n_bkps": n_bkps}
                if len(valid_bkps) == 1: # If we found exactly one change point
                    print(f"Perfect single change point found with BinSeg ({model}, n_bkps={n_bkps}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"BinSeg_{model}_nbps{n_bkps}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    # If we found a best transition index from our search (even if not perfect single)
    if best_transition_index is not None:
        transition_index = best_transition_index
        selected_method = f"{best_method}_{best_params}"
        transition_year = series_df['YEAR'].iloc[transition_index]
        print(f"Best available transition for {species}: {transition_year} (Method: {selected_method}, Score: {best_score:.2g})")
        return transition_index
    # Final fallback: if nothing works, use the most significant change point from the entire search
    # We'll re-run all methods and pick the change point with the largest mean difference
    print(f"No optimal change point found for {species}, using most significant change point")
    all_change_points = []
    # Collect all valid change points from all methods
    for method_name, bkps_list, params in methods_results:
        for bkp in bkps_list:
            if 10 <= bkp <= len(points) - 10:
                before_mean = np.mean(points[bkp-10:bkp])
                after_mean = np.mean(points[bkp:bkp+10])
                mean_diff = abs(after_mean - before_mean)
                all_change_points.append((bkp, mean_diff, method_name))
    if all_change_points:
        # Select the change point with the largest mean difference
        all_change_points.sort(key=lambda x: x[1], reverse=True)
        transition_index = all_change_points[0][0]
        selected_method = f"MostSignificant_{all_change_points[0][2]}"
    else: # Absolute last resort: midpoint
        transition_index = len(points) // 2
        selected_method = "MidpointFallback"
    transition_index = max(5, min(transition_index, len(points) - 5)) # Ensure transition index is within bounds
    transition_year = series_df['YEAR'].iloc[transition_index]
    print(f"Final transition for {species}: {transition_year} (Method: {selected_method})")
    return transition_index

def plot_dl_predictions(ts, species_name, eval_index, output_dir='png'):
    try:
        fig = ts.make_plotly(ens_avg=True, eval_index=eval_index)
        fig.add_vline(x=eval_index, line_dash="dash", line_color="grey")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_image(os.path.join(output_dir, f"{species_name}_dl_predictions.png"))
    except Exception as e:
        print(f"Error plotting DL predictions for {species_name}: {str(e)}")

def plot_dl_predictions_pdf(ts, species_name, eval_index, output_dir='pdf'):
    try:
        fig = ts.make_plotly(ens_avg=True, eval_index=eval_index)
        fig.add_vline(x=eval_index, line_dash="dash", line_color="grey")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_image(os.path.join(output_dir, f"{species_name}_dl_predictions.pdf"))
        print(f"PDF plot saved for {species_name}")
    except Exception as e:
        print(f"Error plotting DL predictions PDF for {species_name}: {str(e)}")

def backup_existing_results():
    results_file = 'csv/bifurcation_results.csv'
    if os.path.exists(results_file):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = f'csv/bifurcation_results_backup_{timestamp}.csv'
        shutil.copy2(results_file, backup_file)
        print(f"Created backup: {backup_file}")
        return backup_file
    return None

def robust_detrending(ts, series_length, method_preference='Lowess'):
    methods_to_try = []
    # Always try Lowess first with adaptive parameters, prioritizing span=0.2
    if method_preference == 'Lowess':
        methods_to_try = [
            ('Lowess', {'span': 0.2}), ('Lowess', {'span': min(0.3, max(0.1, 50/series_length))}), ('Lowess', {'span': 0.15}), ('Lowess', {'span': 0.25}), ]
    else:
        methods_to_try = [
            ('Gaussian', {'bandwidth': 0.2}), ('Gaussian', {'bandwidth': min(0.3, max(0.1, 50/series_length))}), ('Gaussian', {'bandwidth': 0.15}), ('Gaussian', {'bandwidth': 0.25}), ]
    # Add the other method as fallback only after exhausting primary method
    if method_preference == 'Lowess':
        methods_to_try.extend([
            ('Gaussian', {'bandwidth': 0.2}), ('Gaussian', {'bandwidth': min(0.3, max(0.1, 50/series_length))}), ])
    else:
        methods_to_try.extend([
            ('Lowess', {'span': 0.2}), ('Lowess', {'span': min(0.3, max(0.1, 50/series_length))}), ])
    last_error = None
    for method, params in methods_to_try:
        try:
            if method == 'Lowess':
                ts.detrend(method='Lowess', span=params['span'])
            elif method == 'Gaussian':
                ts.detrend(method='Gaussian', bandwidth=params['bandwidth'])
            # Enhanced overfitting detection
            if 'smoothing' in ts.state.columns:
                smoothing_vals = ts.state['smoothing'].values
                original_vals = ts.state['state'].values
                # Calculate R^2 between smoothed and original
                valid_mask = ~np.isnan(smoothing_vals) & ~np.isnan(original_vals)
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(original_vals[valid_mask], smoothing_vals[valid_mask])[0,1]
                    # Calculate variance reduction
                    orig_variance = np.var(original_vals[valid_mask])
                    resid_variance = np.var(original_vals[valid_mask] - smoothing_vals[valid_mask])
                    # Overfitting detection: if correlation is too high AND residual variance is too low
                    if abs(corr) > 0.95 and resid_variance < (0.05 * orig_variance):
                        print(f"Overfitting detected with {method} {params}, trying next method...")
                        # Reset the detrending
                        ts.state['smoothing'] = np.nan
                        ts.state['residuals'] = ts.state['state'].copy()
                        continue
                    # Also check if smoothing is too wiggly (too many inflection points)
                    if len(smoothing_vals[valid_mask]) > 20:
                        diff1 = np.diff(smoothing_vals[valid_mask])
                        diff2 = np.diff(diff1)
                        inflection_count = np.sum(np.abs(diff2) > np.std(diff2) * 2)
                        inflection_ratio = inflection_count / len(diff2)
                        if inflection_ratio > 0.3:  # Too many inflection points
                            print(f"Overly wiggly smoothing with {method} {params}, trying next method...")
                            ts.state['smoothing'] = np.nan
                            ts.state['residuals'] = ts.state['state'].copy()
                            continue
                print(f"Successfully detrended using {method} with params {params}")
                return True
        except Exception as e:
            last_error = e
            if 'smoothing' in ts.state.columns: # Reset detrending on error
                ts.state['smoothing'] = np.nan
                ts.state['residuals'] = ts.state['state'].copy()
            continue
    print(f"All detrending methods failed. Last error: {last_error}")
    return False

def process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None):
    try:
        setup_warning_suppression(worker_process=True)
        taxonomic_group = get_cached_taxonomic_group(species_name)
        dataset = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
        series = species_data[["TIME", "ABUNDANCE"]].copy()
        series.columns = ["YEAR", "ABUNDANCE"]
        original_length = len(series)
        raw_abundance = species_data["ABUNDANCE"]
        if original_length <= 500:
            series_adjusted = adjust_series_size(series, max_size=500)
            classifier_paths = classifier_paths_500
            max_size = 500
        else:
            series_adjusted = adjust_series_size(series, max_size=1500)
            classifier_paths = classifier_paths_1500
            max_size = 1500
        transition_idx = find_transition(series_adjusted, species_name)
        if transition_idx >= original_length: # Cap transition index to original data boundaries
            transition_idx = original_length - 1
        change_points = [transition_idx] if transition_idx is not None else []
        ts = ewstools.TimeSeries(series_adjusted['ABUNDANCE'].values, transition=transition_idx)
        ts.time_values = series_adjusted['YEAR'].values
        ts.state['year'] = series_adjusted['YEAR'].values
        detrend_success = robust_detrending(ts, original_length, method_preference='Lowess')
        if not detrend_success:
            raise ValueError("Failed to detrend time series")
        if 'smoothing' not in ts.state.columns: # Calculate smoothing average in last 1/5 of time series (ignoring NaNs)
            raise ValueError("Smoothing vector not found - cannot calculate smoothing average")
        smoothing_vals = ts.state['smoothing'].values
        valid_indices = ~np.isnan(smoothing_vals) # Get last 1/5 of time series, ignoring NaNs
        valid_vals = smoothing_vals[valid_indices]
        if len(valid_vals) > 0:
            last_fifth_idx = max(0, len(valid_vals) - len(valid_vals) // 5)
            smoothing_avg = np.mean(valid_vals[last_fifth_idx:])
        else:
            smoothing_avg = None
        if smoothing_avg is not None and abs(smoothing_avg - 1.0) < 1.0e-5: # Check if smoothing average is close to 1
            save_results(species_name, 'no bifurcation', None, [], smoothing_avg, raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, raw_abundance.max(), 0.0,  # Zero probability for no bifurcation
                        taxonomic_group, dataset)  # Pass dataset
            return {
                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no bifurcation', 'raw_tipping_point_type': 'no bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
            }
        # Check if we have a valid transition point
        if transition_idx is None or transition_idx < 0 or transition_idx >= len(series_adjusted):
            save_results(species_name, 'no bifurcation', None, [], smoothing_avg, raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, raw_abundance.max(), 0.0, taxonomic_group, dataset)  # Pass dataset
            return {
                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no bifurcation', 'raw_tipping_point_type': 'no bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
            }
        eval_index = int(0.9 * transition_idx) # Calculate evaluation index as 90% of transition index
        if eval_index < 0:
            eval_index = 0
        ts.clear_dl_preds() # Clear previous DL predictions
        for classifier_path in classifier_paths: # Apply ensemble of classifiers with memory management
            classifier = load_model(classifier_path)
            ts.apply_classifier_inc(classifier, inc=10, verbose=0, name=os.path.basename(classifier_path))
            tf.keras.backend.clear_session() # Clear TensorFlow session after each classifier to free memory
            gc.collect()
        best_probability = 0.0 # Use prediction at transition point and average all classifiers
        raw_bifurcation_type = 'no bifurcation'
        mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])  # Default to no bifurcation
        if len(ts.dl_preds) > 0:
            transition_time = series_adjusted['YEAR'].iloc[transition_idx] # Get the time value at the transition point
            valid_preds = ts.dl_preds[ts.dl_preds['time'] <= transition_time] # Find predictions at or before the transition point
            if len(valid_preds) > 0:
                transition_pred = valid_preds.iloc[-1] # Use the prediction closest to the transition point
                transition_time_used = transition_pred['time']
                ensemble_preds_at_time = ts.dl_preds[ts.dl_preds['time'] == transition_time_used] # Average probabilities across all ensemble members at this time point
                numeric_cols = [0, 1, 2, 3]
                if len(ensemble_preds_at_time) > 1:
                    mean_probs = ensemble_preds_at_time[numeric_cols].astype(float).mean() # Average across all classifiers in the ensemble
                    print(f"Using ensemble average from {len(ensemble_preds_at_time)} classifiers at transition time {transition_time_used}")
                else:
                    mean_probs = transition_pred[numeric_cols].astype(float)
                    print(f"Using single classifier prediction at transition time {transition_time_used}")
                best_probability = mean_probs.max() # Get the best probability and corresponding class
                pred_class = mean_probs.idxmax()
                raw_bifurcation_type = bifurcation_map.get(pred_class, 'no bifurcation')
                print(f"Raw classification for {species_name}: {raw_bifurcation_type} (probability: {best_probability:.2g})")
                final_bifurcation_type = raw_bifurcation_type # Apply threshold to determine final bifurcation type
                print(f"Classification for {species_name}: {final_bifurcation_type} (probability: {best_probability:.2g})")
            else: # No predictions before transition - use no bifurcation
                mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
                raw_bifurcation_type = 'no bifurcation'
                final_bifurcation_type = 'no bifurcation'
                best_probability = 1.0  # Probability for no bifurcation
                print(f"No valid predictions before transition for {species_name}, classifying as no bifurcation")
        else:
            mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
            raw_bifurcation_type = 'no bifurcation'
            final_bifurcation_type = 'no bifurcation'
            best_probability = 1.0  # Probability for no bifurcation
            print(f"No DL predictions generated for {species_name}, classifying as no bifurcation")
        transition_year = series['YEAR'].iloc[min(transition_idx, original_length-1)]
        # Final validation - override to no bifurcation if conditions aren't met
        if (transition_idx is None or
            transition_year is None or
            pd.isna(transition_year) or
            not change_points or
            final_bifurcation_type in ['null', 'error'] or
            len(ts.dl_preds) == 0):
            final_bifurcation_type = 'no bifurcation'
            transition_year = None
            change_points = []
            best_probability = 0.0 # Reset probability to 0 if we're forcing no bifurcation
        if species_to_plot is not None and (len(species_to_plot) == 0 or species_name in species_to_plot):
            plot_dl_predictions(ts, species_name, eval_index)
        pdf_species = ['Anaxyrus sp7', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
        if species_name in pdf_species:
            plot_dl_predictions_pdf(ts, species_name, eval_index)
        save_results(species_name, final_bifurcation_type, transition_year, change_points, smoothing_avg, raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, raw_abundance.max(), best_probability, taxonomic_group, dataset)  # Pass dataset
        return {
            'species': species_name, 'dataset': dataset, 'tipping_point_type': final_bifurcation_type, 'raw_tipping_point_type': raw_bifurcation_type, 'ews_metrics': mean_probs.to_dict(), 'change_points': change_points, 'taxonomic_group': taxonomic_group, 'transition_year': transition_year, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': best_probability
        }
    except Exception as e:
        print(f"Error processing {species_name}: {str(e)}")
        traceback.print_exc()
        tf.keras.backend.clear_session() # Clear session even on error
        gc.collect()
        taxonomic_group_error = get_cached_taxonomic_group(species_name) # Get taxonomic group for error case too
        dataset_error = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown' # Get dataset for error case
        try: # Save error result as no bifurcation with zero probability
            save_results(species_name, 'no bifurcation', None, [], None, 0, 0, 0.0, taxonomic_group_error, dataset_error)
        except Exception as save_error:
            print(f"Failed to save error result for {species_name}: {str(save_error)}")
        return {
            'species': species_name, 'dataset': dataset_error, 'tipping_point_type': 'no bifurcation', 'raw_tipping_point_type': 'no bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0
        }
    finally:
        tf.keras.backend.clear_session() # Clear session at the end of processing each species
        gc.collect()

def read_results_with_fixed_columns(results_file):
    try:
        results_df = pd.read_csv(results_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            results_df = pd.read_csv(results_file, encoding='latin-1')
        except:
            results_df = pd.read_csv(results_file)
    column_mapping = {}
    for col in results_df.columns:
        if '+AF8-' in col:
            fixed_col = col.replace('+AF8-', '_')
            column_mapping[col] = fixed_col
        else:
            column_mapping[col] = col
    if column_mapping != dict(zip(results_df.columns, results_df.columns)):
        results_df = results_df.rename(columns=column_mapping)
        print(f"Fixed column names: {column_mapping}")
    return results_df

def update_taxonomic_groups_from_mapping():
    try:
        results_df = read_results_with_fixed_columns('csv/bifurcation_results.csv')
        taxon_mapping_df = pd.read_csv('csv/taxonomic_cache.csv', encoding='utf-8')
        taxon_mapping_dict = dict(zip(
            taxon_mapping_df['species'].astype(str).str.lower(), taxon_mapping_df['taxonomic_group']
        ))
        updated_count = 0
        for idx, row in results_df.iterrows():
            species_str = str(row['species'])
            species_lower = species_str.lower()
            if species_lower in taxon_mapping_dict:
                taxon_value = taxon_mapping_dict[species_lower]
                collapsed = collapse_taxonomic_group(taxon_value)
                if collapsed != 'Other':
                    results_df.at[idx, 'taxonomic_group'] = collapsed
                    updated_count += 1
        results_df.to_csv('csv/bifurcation_results.csv', index=False, encoding='utf-8')
        print(f"Taxonomic groups updated for {updated_count} species from taxonomic_cache.csv")
    except Exception as e:
        print(f"Error updating taxonomic groups: {str(e)}")

def create_master_map(species_list, n_jobs):
    master_map = pd.DataFrame({'species': species_list})
    # Assign each species to a core using round-robin
    master_map['assigned_core'] = [i % n_jobs for i in range(len(species_list))]
    master_map['status'] = 'pending'  # pending, processing, completed
    master_map['start_time'] = None
    master_map['completion_time'] = None
    os.makedirs('csv', exist_ok=True)
    master_map.to_csv('csv/master_map.csv', index=False)
    print(f"Created master map with {len(species_list)} species distributed across {n_jobs} cores")
    return master_map

def get_remaining_species_count():
    try:
        master_map = pd.read_csv('csv/master_map.csv')
        pending_count = len(master_map[master_map['status'] == 'pending'])
        processing_count = len(master_map[master_map['status'] == 'processing'])
        completed_count = len(master_map[master_map['status'] == 'completed'])
        return pending_count, processing_count, completed_count
    except:
        return 0, 0, 0

def save_results(species_name, bifurcation_type, transition_year, change_points, smoothing_avg, current_abundance, max_abundance, best_probability, taxonomic_group, dataset):
    os.makedirs('csv', exist_ok=True)
    results_file = 'csv/bifurcation_results.csv'
    # Check for all cases that should be considered "no bifurcation"
    if (bifurcation_type in ['error', 'null'] or
        transition_year is None or
        pd.isna(transition_year) or
        not change_points or  # Empty list
        len(change_points) == 0 or  # No change points
        (smoothing_avg is not None and abs(smoothing_avg - 1.0) < 1.0e-5)):  # Smoothing average close to 1
        bifurcation_type = 'no bifurcation'
        transition_year = None
        change_points = []
        best_probability = 0.0  # Reset probability when forcing no bifurcation
    new_result = pd.DataFrame({
        'species': [species_name], 'dataset': [dataset], 'taxonomic_group': [taxonomic_group], 'tipping_point_type': [bifurcation_type], 'transition_year': [transition_year], 'change_points': [str(change_points)], 'smoothing_avg': [f"{smoothing_avg:.2g}" if smoothing_avg is not None else "NaN"], 'current_abundance': [current_abundance], 'max_abundance': [max_abundance], 'best_probability': [best_probability]
    })
    try:
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            results_df = read_results_with_fixed_columns(results_file)
            if species_name in results_df['species'].values:
                print(f"WARNING: {species_name} already exists in results. Skipping to prevent data loss.")
                return  # Don't save anything - preserve existing data
            results_df = pd.concat([results_df, new_result], ignore_index=True)
        else:
            results_df = new_result
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"Results saved for {species_name} (Dataset: {dataset}, Taxonomic group: {taxonomic_group}, Probability: {best_probability:.2g})")
    except Exception as e:
        print(f"Error saving results for {species_name}: {str(e)}")
        if not os.path.exists(results_file):
            new_result.to_csv(results_file, index=False, encoding='utf-8')

def save_batch_results(batch_results):
    results_file = 'csv/bifurcation_results.csv'
    if not batch_results:
        return
    try:
        batch_df = pd.DataFrame(batch_results)
        # Ensure all required columns are present
        required_columns = ['species', 'taxonomic_group', 'tipping_point_type', 'transition_year',    'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
        for col in required_columns:
            if col not in batch_df.columns:
                if col == 'best_probability':
                    batch_df[col] = 0.0  # Default probability
                else:
                    batch_df[col] = None
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            existing_df = pd.read_csv(results_file)
            # Remove any existing entries for these species (shouldn't happen, but just in case)
            existing_df = existing_df[~existing_df['species'].isin(batch_df['species'])]
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
        else:
            combined_df = batch_df
        combined_df.to_csv(results_file, index=False)
        print(f"Batch results saved for {len(batch_results)} species (redundant safety save)")
    except Exception as e:
        print(f"Error saving batch results: {e}")
        # If batch saving fails, it's ok because individual species were already saved
        print("Note: Individual species were already saved via save_results()")

def mark_batch_completed(core_id, species_names):
    master_map_file = 'csv/master_map.csv'
    for attempt in range(3):
        try:
            master_map = pd.read_csv(master_map_file)
            master_map.loc[master_map['species'].isin(species_names), 'status'] = 'completed'
            master_map.loc[master_map['species'].isin(species_names), 'completion_time'] = time.time()
            master_map.to_csv(master_map_file, index=False)
            print(f"Core {core_id} completed batch: {species_names}")
            break
        except Exception as e:
            print(f"Core {core_id} attempt {attempt + 1} failed to mark complete: {e}")
            time.sleep(0.1)

def get_next_batch_for_core(core_id, batch_size=1):
    master_map_file = 'csv/master_map.csv'
    for attempt in range(3):  # Retry mechanism for file locking
        try:
            master_map = pd.read_csv(master_map_file) # Read the current master map
            available_species = master_map[ # Find pending species assigned to this core
                (master_map['assigned_core'] == core_id) &
                (master_map['status'] == 'pending')
            ]
            if len(available_species) == 0:
                return None  # No more work for this core
            batch = available_species.head(batch_size) # Take the next batch
            species_names = batch['species'].tolist()
            # Update status to 'processing' for these species
            master_map.loc[master_map['species'].isin(species_names), 'status'] = 'processing'
            master_map.loc[master_map['species'].isin(species_names), 'start_time'] = time.time()
            master_map.to_csv(master_map_file, index=False) # Write back the updated master map
            print(f"Core {core_id} claimed batch: {species_names}")
            return species_names
        except Exception as e:
            print(f"Core {core_id} attempt {attempt + 1} failed to get batch: {e}")
            time.sleep(0.1)  # Brief pause before retry
    return None  # All retries failed

def core_worker(core_id, full_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, batch_size=1):
    setup_warning_suppression(worker_process=True)
    processed_count = 0
    while True:
        species_batch = get_next_batch_for_core(core_id, batch_size)
        if species_batch is None:
            print(f"Core {core_id}: No more species to process. Completed {processed_count} species.")
            break
        batch_results = []
        for species_name in species_batch:
            try:
                print(f"Core {core_id} processing: {species_name}")
                species_data = full_data[full_data['SPECIES_ID'] == species_name]
                if len(species_data) == 0:
                    print(f"Core {core_id}: No data found for {species_name}")
                    taxonomic_group_missing = get_cached_taxonomic_group(species_name)
                    save_results(species_name, 'no bifurcation', None, [], None, 0, 0, 0.0, taxonomic_group_missing)
                    mark_batch_completed(core_id, [species_name])
                    continue
                result = process_species(
                    species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot
                )
                batch_results.append(result)
                processed_count += 1
                # Each species is already saved individually by process_species->save_results
                # So we don't need to wait for the whole batch
                print(f"Core {core_id} completed and saved: {species_name}")
            except Exception as e:
                print(f"Core {core_id} error processing {species_name}: {e}")
                taxonomic_group_error = get_cached_taxonomic_group(species_name) # Get taxonomic group for error case
                try: # Save error result immediately
                    save_results(species_name, 'no bifurcation', None, [], None, 0, 0, 0.0, taxonomic_group_error)
                except Exception as save_error:
                    print(f"Core {core_id} failed to save error result for {species_name}: {save_error}")
                error_result = { # Create error result for batch tracking
                    'species': species_name, 'tipping_point_type': 'no bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0
                }
                batch_results.append(error_result)
                processed_count += 1
        mark_batch_completed(core_id, species_batch)
        if batch_results:
            save_batch_results(batch_results)
            print(f"Core {core_id} saved batch of {len(batch_results)} species")
    return processed_count

def run_analysis_pipeline(min_records=100, force_reclassification=False, species_to_plot=None, n_jobs=-1, batch_size=1, specific_species_list=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    classifier_base = os.path.join(script_dir, '..', 'saved_classifiers', 'bury_pnas_21')
    root_path_500 = os.path.join(classifier_base, 'len500')
    root_path_1500 = os.path.join(classifier_base, 'len1500')
    classifier_names_500 = [name for name in os.listdir(root_path_500) if name.endswith('.pkl')]
    classifier_paths_500 = [os.path.join(root_path_500, name) for name in classifier_names_500[:20]]
    classifier_names_1500 = [name for name in os.listdir(root_path_1500) if name.endswith('.pkl')]
    classifier_paths_1500 = [os.path.join(root_path_1500, name) for name in classifier_names_1500[:20]]
    bifurcation_map = {
        0: 'fold bifurcation', 1: 'Hopf bifurcation', 2: 'transcritical bifurcation', 3: 'no bifurcation'
    }
    biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
    rivfishtime_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
    cedar_creek_paths = {
        'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
        'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
        'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')
    }
    full_data, top_species, species_count = load_combined_data(
        biotime_path, rivfishtime_path, cedar_creek_paths, min_records=min_records, use_existing_biotime=True
    )
    if specific_species_list is not None:
        species_to_process = list(specific_species_list)
        print(f"Processing specific list of {len(species_to_process)} species")
    else:
        species_to_process = top_species
    results_file = 'csv/bifurcation_results.csv'
    processed_species = set()
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        try:
            existing_results = read_results_with_fixed_columns(results_file)
            processed_species = set(existing_results['species'].unique())
            print(f"Found {len(processed_species)} already processed species in total")
        except Exception as e:
            print(f"Error reading existing results: {e}")
            traceback.print_exc()
    if force_reclassification:
        species_to_process_final = species_to_process
        print(f"Force reclassification: processing all {len(species_to_process_final)} target species")
    else:
        species_to_process_final = [s for s in species_to_process if s not in processed_species]
        print(f"Species remaining to process: {len(species_to_process_final)} out of {len(species_to_process)} target species")
    if len(species_to_process_final) == 0:
        print("All target species already processed.")
        try:
            results_df = read_results_with_fixed_columns(results_file)
            create_summary_figure(results_df)
        except Exception as e:
            print(f"Error creating summary figure: {e}")
        return pd.DataFrame()
    if force_reclassification:
        print("Clearing taxonomic cache for species to be reprocessed...")
        cache_file = 'csv/taxonomic_cache.csv'
        if os.path.exists(cache_file):
            cache_df = pd.read_csv(cache_file, encoding='utf-8')
            cache_df = cache_df[~cache_df['species'].isin(species_to_process_final)]
            cache_df.to_csv(cache_file, index=False, encoding='utf-8')
        taxon_mapping = {}
        if os.path.exists('csv/taxon_mapping.csv'):
            taxon_df = pd.read_csv('csv/taxon_mapping.csv', encoding='utf-8')
            taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
        for species in tqdm(species_to_process_final, desc="Reclassifying species"):
            get_cached_taxonomic_group(species, taxon_mapping)
    process_data = full_data[full_data['SPECIES_ID'].isin(species_to_process_final)]
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    print(f"Using {n_jobs} cores with batch size {batch_size}")
    master_map = create_master_map(species_to_process_final, n_jobs)
    core_worker_partial = partial(
        core_worker, full_data=process_data, classifier_paths_500=classifier_paths_500, classifier_paths_1500=classifier_paths_1500, bifurcation_map=bifurcation_map, species_to_plot=species_to_plot, batch_size=batch_size
    )
    print(f"Starting parallel processing with {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(core_worker_partial)(core_id)
        for core_id in range(n_jobs)
    )
    total_processed = sum(results)
    print(f"Parallel processing complete. Total species processed: {total_processed}")
    try:
        results_df = read_results_with_fixed_columns(results_file)
        create_summary_figure(results_df)
        pending, processing, completed = get_remaining_species_count()
        print(f"Final status - Pending: {pending}, Processing: {processing}, Completed: {completed}")
    except Exception as e:
        print(f"Error in final processing: {e}")
    return results_df

def refresh_taxonomic_cache_from_results(results_file='csv/bifurcation_results.csv'):
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        print("No existing results file, skipping taxonomic cache refresh.")
        return
    try:
        results_df = read_results_with_fixed_columns(results_file)
        if 'species' not in results_df.columns:
            print("Results file missing 'species' column, cannot refresh cache.")
            return
        unique_species = results_df['species'].unique()
        print(f"Refreshing taxonomic cache for {len(unique_species)} species from results...")
        taxon_mapping = {}
        if os.path.exists('csv/taxon_mapping.csv') and os.path.getsize('csv/taxon_mapping.csv') > 0:
            taxon_df = pd.read_csv('csv/taxon_mapping.csv', encoding='utf-8')
            taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
        for species in tqdm(unique_species, desc="Updating cache"):
            get_cached_taxonomic_group(species, taxon_mapping=taxon_mapping, force_recompute=False)
        print("Taxonomic cache refresh completed.")
    except Exception as e:
        print(f"Error refreshing taxonomic cache: {e}")

def main():
    setup_warning_suppression()
    backup_existing_results()
    update_taxonomic_cache_from_other()
    refresh_taxonomic_cache_from_results()
    print("Starting EWS analysis pipeline...")
    min_records = 20
    force_reclassification = False
    results_file = 'csv/bifurcation_results.csv'
    bifurcation_results_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
    if bifurcation_results_exists:
        results_df = read_results_with_fixed_columns(results_file)
        combined_species_file = f'csv/combined_species_min{min_records}.csv'
        if os.path.exists(combined_species_file):
            combined_species_df = pd.read_csv(combined_species_file, encoding='utf-8')
            if 'species' in combined_species_df.columns:
                combined_species_list = combined_species_df['species'].tolist()
            else:
                raise KeyError("No species column found in combined species file")
            processed_species = set(results_df['species'].unique())
            combined_species_set = set(combined_species_list)
            processed_combined_species = processed_species & combined_species_set
            missing_species = combined_species_set - processed_combined_species
            print(f"Total species with >= {min_records} records: {len(combined_species_set)}")
            print(f"Already processed species: {len(processed_combined_species)}")
            print(f"Missing species: {len(missing_species)}")
            if len(missing_species) == 0:
                print(f"All species with >= {min_records} records already processed. Generating figures...")
                create_summary_figure(results_df)
                overall_counts = results_df['tipping_point_type'].value_counts()
                plot_histogram(overall_counts, 'img/overall_distribution.pdf')
                print("Figures generated from existing results!")
                return results_df
            else:
                print(f"Found {len(missing_species)} species missing from results. Running analysis for missing species only...")
                if force_reclassification:
                    species_list_to_process = None
                    print("Force reclassification enabled: will reprocess ALL species.")
                else:
                    species_list_to_process = list(missing_species)
        else:
            print("Combined species file not found. Running full analysis...")
            species_list_to_process = None
    else:
        print("No existing results found. Running full analysis...")
        species_list_to_process = None
    plot_dl_figs = True
    specific_species_to_plot = []
    if plot_dl_figs:
        species_to_plot = specific_species_to_plot
    else:
        species_to_plot = None
    results = run_analysis_pipeline(
        min_records=min_records, force_reclassification=force_reclassification, species_to_plot=species_to_plot, n_jobs=-1, batch_size=1, specific_species_list=species_list_to_process
    )
    update_taxonomic_groups_from_mapping()
    try:
        results_df = read_results_with_fixed_columns(results_file)
        create_summary_figure(results_df)
        overall_counts = results_df['tipping_point_type'].value_counts()
        plot_histogram(overall_counts, 'img/overall_distribution.pdf')
    except Exception as e:
        print(f"Error in final processing: {e}")
    print("Tipping points analysis complete!")
    return results_df

if __name__ == "__main__":
    results = main()
