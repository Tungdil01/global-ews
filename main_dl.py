#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import gc, re, sys, time, shutil, logging, requests, traceback, pickle, threading, json, tempfile
import ewstools
import absl.logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from joblib import Parallel, delayed, Memory
from functools import partial
from tqdm import tqdm
import warnings
cache_lock = threading.Lock()

def setup_warning_suppression(worker_process=True):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
    warnings.filterwarnings("ignore", message="SavedModel saved prior to TF 2.5")
    warnings.filterwarnings("ignore", message=".*SavedModel.*")
    warnings.filterwarnings("ignore", message=".*oneDNN.*")
    warnings.filterwarnings("ignore", message=".*oneAPI Deep Neural Network Library.*")
    warnings.filterwarnings("ignore", message=".*Could not load dynamic library.*")
    warnings.filterwarnings("ignore", message=".*libcudart.*")
    warnings.filterwarnings("ignore", message=".*libnvinfer.*")
    warnings.filterwarnings("ignore", message=".*TF-TRT Warning.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="ewstools")
    warnings.filterwarnings("ignore", category=UserWarning, module="ewstools")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.fromnumeric")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core._methods")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="ruptures")
    if worker_process:
        warnings.filterwarnings("ignore", category=Warning)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        try:
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
    try:
        tf_logger = tf.get_logger()
        tf_logger.setLevel('ERROR')
        try:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except:
            pass
        try:
            tf.compat.v1.disable_eager_execution()
        except:
            pass
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass

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

API_DELAY = 0.3
MAX_RECURSION_DEPTH = 10
CACHE_ENABLED = True
memory = Memory("./.gbif_cache", verbose=0)

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
    'Young laminariales': 'Ochrophyta',
    'Solenicola setigera': 'Ochrophyta',
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
    'Unknown Insect sp327': 'Insecta',
    'Unknown Insect sp328': 'Insecta',
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

AVES_TAXA = {'Uria', 'Sterna', 'Columba', 'Cygnus', 'Passer'}
ACTINOPTERYGII_TAXA = {'Osmeriformes', 'Siluriformes', 'Perciformes', 'Tetraodontiformes', 'Scorpaeniformes', 'Myctophiformes', 'Cyprinodontiformes', 'Aulopiformes', 'Cypriniformes', 'Gadiformes', 'Lampriformes', 'Ophidiiformes', 'Characiformes', 'Gobiesociformes', 'Notacanthiformes', 'Pleuronectiformes', 'Beryciformes', 'Stomiiformes', 'Esociformes', 'Atheriniformes', 'Lophiiformes', 'Clupeiformes', 'Salmoniformes', 'Gasterosteiformes', 'Syngnathiformes', 'Anguilliformes', 'Zeiformes', 'Batrachoidiformes', 'Polymixiiformes', 'Beloniformes', 'Mugiliformes', 'Clupeiformes', 'Amiiformes', 'Cyprinidae', 'Aprion', 'Oncorhynchus', 'Galaxias', 'Scarus', 'Ammodytes', 'Teleostei', 'Percopsiformes', 'Kuhlia rupestris', 'Kuhlia', 'Prionotus', 'Leptocephali', 'Gibbonsia', 'Centropogon australis', 'Liparis', 'Chromis', 'Apogon', 'Naso', 'Acipenser oxyrinchus', 'Eurypharynx pelecanoides', 'Gonorynchiformes', 'Osteoglossiformes', 'Gymnotiformes', 'Saccopharyngiformes', 'Stephanoberyciformes', 'Elopiformes', 'Synbranchiformes', 'Cetomimiformes', 'Lepisosteiformes', 'Polypteriformes', 'Acipenseriformes', 'Pagrus'}
ELASMOBRANCHII_TAXA = {'Carcharhiniformes', 'Lamniformes', 'Orectolobiformes', 'Heterodontiformes', 'Hexanchiformes', 'Squaliformes', 'Squatiniformes', 'Pristiophoriformes', 'Rajiformes', 'Torpediniformes', 'Myliobatiformes', 'Rhinopristiformes', 'Raja'}
HOLOCEPHALI_TAXA = {'Chimaeriformes'}
CHONDRICHTHYES_TAXA = ELASMOBRANCHII_TAXA | HOLOCEPHALI_TAXA | {'Selachimorpha'}
HEMICHORDATA_TAXA = {'Enteropneusta'}
TUNICATA_TAXA = {'Ascidiacea', 'Thaliacea', 'Polycarpa', 'Appendicularia', 'Oikopleura', 'Oikopleuridae', 'Fritillaridae', 'Boltenia', 'Ascidiella', 'Copelata', 'Diplosoma'}
ECHINODERMATA_TAXA = {'Asterias', 'Atelostomata', 'Phrynophiurida', 'Henricia', 'Asterozoa', 'Asterina', 'Ophiothrix', 'Asteroidea', 'Echinoidea', 'Holothuroidea', 'Holothuria', 'Cucumaria', 'Elpidia', 'Echinus', 'Ophioderma', 'Ophiuroids unidentified', 'Pentagonaster', 'Salmacis', 'Salmacis sp312', 'Salmacis sp313', 'Stereoderma', 'Stelleroidea', 'Thyone', 'Holothuria sp2'}
MOLLUSCA_TAXA = {'Patella', 'Bivalvia', 'Crepidula', 'Rissoa', 'Clione', 'Physa', 'Teuthida', 'Diplodonta', 'Prosobranchia', 'Teuthoidea', 'Lithopoma', 'Gymnosomata', 'Abra', 'Ervilia nitens', 'Pandora', 'Lucina', 'Spiratella', 'Lyonsia', 'Patellogastropoda', 'Opisthobranchia', 'Galba', 'Gastropoda', 'Cephalopoda', 'Aplacophora', 'Rossia', 'Strombiformis'}
POLYCHAETA_TAXA = {'Spirobranchus', 'Amphitrite', 'Terebellidae', 'Glycera', 'Nereis', 'Spio', 'Chone', 'Exogone', 'Euclymeninae', 'Euclymeninae', 'Eteone', 'Pectinaria', 'Pholoe', 'Harmothoinae', 'Megalomma', 'Monticellina', 'Monticellina sp', 'Aphroditacea'}
SIPUNCULA_TAXA = {'Sipunculidea', 'Aspidosiphoniformes'}
ANNELIDA_TAXA = POLYCHAETA_TAXA | SIPUNCULA_TAXA | {'Halicryptus spinulosus', 'Hirudinea', 'Polychaeta', 'Sipuncula'}
NEMERTEA_TAXA = {'Nemertinea', 'Lineidae', 'Tubulanus', 'Rhynchocoela', 'Nemertina', 'Tetrastemma sp'}
PHORONIDA_TAXA = {'Phoronis', 'Phoronida'}
PLATYHELMINTHES_TAXA = {'Turbellaria', 'Girardia'}
INSECTA_TAXA = {'Enallagma', 'Homarus', 'Conocephalus', 'Rhyacophila', 'Tethea', 'Silphini', 'Colymbetes', 'Hydropsyche', 'Elachista', 'Sialis', 'Colias', 'Athetini', 'Leptocera', 'Calopteryx', 'Gerris', 'Lestes', 'Zygoptera', 'Tanypodinae', 'Gomphus', 'Acilius', 'Agabinae', 'Aleocharinae', 'Anax', 'Capnia', 'Chironomini', 'Dolichopus undet', 'Ephemera', 'Ephemerella', 'Forcipomyiinae sp3', 'Hydrobius', 'Macronema sp', 'Oxybelus undet', 'Paederinae sp', 'Pedicidae', 'Perilampus undet', 'Plagiognathus undet', 'Potamophylax', 'Scaphisomini', 'Scolomyia', 'Scolytidae', 'Sericostoma', 'Sphaeridiinae', 'Stenonema', 'Stylurus', 'Symplecta undet', 'Tanytarsini', 'Urophora undet', 'Limnophila'}
HEXAPODA_TAXA = INSECTA_TAXA | {'Collembola', 'Insecta'}
BRANCHIOPODA_TAXA = {'Diplostraca', 'Daphnia', 'Cladocera', 'Bosmina', 'Chydorus', 'Diaphanosoma', 'Evadne', 'Alona', 'Laevicaudata', 'Holopedium', 'Leptodora', 'Eurycercus sp', 'Sida'}
MALACOSTRACA_TAXA = {'Isopoda', 'Homarus', 'Cumella', 'Hyas', 'Dendrobranchiata', 'Sabinea septemcarinata', 'Sabinea', 'Apseudes', 'Parthenope', 'Caprella', 'Galathea','Callianassa', 'Parthenope longispinus', 'Hyale', 'Sergia', 'Gammarus salinus', 'Nematoscelis', 'Parthenope hoplonotus', 'Calappa', 'Hippolyte', 'Hyperia', 'Jaera', 'Leptocheirus', 'Leucothoe', 'Maera', 'Anomura', 'Atylus', 'Corophioidea', 'Dulichia', 'Eurydice', 'Neolepas', 'Oxycephalus', 'Palinurid phyllosoma', 'Pamphithoidea', 'Penaeoidea', 'Phrosina', 'Pilumnus', 'Pontogeneia sp', 'Porcellana', 'Sicyonia', 'Sphaeroma', 'Achaeus sp5993', 'Actaea tuberculosa', 'Leucosiid zoea'}
COPEPODA_TAXA = {'Oithona', 'Oncaea', 'Calanus', 'Cyclops', 'Poecilostomatoida', 'Clytemnestra', 'Labidocera', 'Heterorhabdus', 'Kelleria', 'Lubbockia sp', 'Mecynocera', 'Porcellidium'}
MAXILLOPODA_TAXA= COPEPODA_TAXA | {'Maxillopoda', 'Copepoda'}
THECOSTRACA_TAXA = {'Austrominius', 'Balanus', 'Cirripedia', 'Cirripedia cypris', 'Neolepas'}
OLIGOSTRACA_TAXA = {'Ostracoda', 'Sessilia', 'Cirripedia', 'Sarsiella'}
ARACHNIDA_TAXA = {'Pycnogonida', 'Acari', 'Pantopoda', 'Hydrachnidae', 'Lebertia', 'Hygrobates', 'Erigone atra', 'Monostyliferoidea', 'Phidippus undet', 'Torrenticola'}
CHELICERATA_TAXA = ARACHNIDA_TAXA | {'Merostomata', 'Arachnida'}
CNIDARIA_TAXA = {'Anthozoa', 'Hydrozoa', 'Scyphozoa', 'Sertularia', 'Obelia', 'Acropora', 'Physonect', 'Vogtia', 'Edwardsia', 'Siphonophore', 'Octocorallia', 'Ceriantharia', 'Hexacorallia', 'Hydroidolina', 'Leptoria', 'Leptoseris', 'Lucernaria', 'Alcyonacea', 'Bolocera', 'Calycophoran', 'Coryne', 'Diphasia', 'Diphyes', 'Echinopora', 'Ellisella', 'Ellisella sp2', 'Mopsella sp1', 'Mopsella sp2', 'Physonectae', 'Prayid sp', 'Primnoa', 'Sarcophyton', 'Siphonophora', 'Steroenepthya', 'Stylophora', 'Turbinaria', 'Virgularia sp1'}
BRYOZOA_TAXA = {'Gymnolaemata', 'Stenolaemata', 'Cellaria', 'Bugula', 'Alcyonidium albidum', 'Alcyonidium', 'Margaretta', 'Didymozoum'}
PORIFERA_TAXA = {'Demospongiae', 'Cliona', 'Axinella', 'Leuconia', 'Clathrina sp1', 'Clathrina sp3', 'Pseudoceratina sp6', 'Raspailia sp2', 'Raspailia sp23', 'Tethya sp17', 'Tethya sp2', 'Tethya sp3', 'Tethya sp4'}
ROTIFERA_TAXA = {'Rotifera', 'Eurotatoria', 'Synchaeta', 'Keratella', 'Polyarthra', 'Trichocerca', 'Asplanchna', 'Kellicottia'}
BACILLARIOPHYCEAE_TAXA = {'Nitzschia', 'Navicula', 'Amphora', 'Cerataulina','Bacillariophyceae'}
TRACHEOPHYTA_TAXA = {'Plantae', 'Asteriscus', 'Fritillaria', 'Saggitaria', 'Orthosia', 'Blepharis', 'Canthidium', 'Limonia', 'Lesquerella', 'Digitaria', 'Pectis', 'Astragalus', 'Mertensia', 'Aglaia cucullata', 'Peridiaceae', 'Hibbertia', 'Acourtia', 'Byblis', 'Sagittaria', 'Viola CHE1', 'Weeds', 'Limnophila'}
BRYOPHYTA_TAXA = {'Bryophyta', 'Verrucaria', 'Anthocerotophyta', 'Bryopsida'}
CHLOROPHYTA_TAXA = {'Cladophora', 'Ulva', 'Mougeotia', 'Carteria', 'Schroederia', 'Neomeris', 'Ulothrix', 'Eudorina'}
STREPTOPHYTA_TAXA = {'Charophyta'}
CHLOROPLASTIDA_TAXA = CHLOROPHYTA_TAXA | STREPTOPHYTA_TAXA | {'Prasinodermatophyta', 'Chlorophyta', 'Streptophyta'}
RHODOPHYTA_TAXA = {'Polysiphonia', 'Ceramium', 'Porphyra', 'Cryptopleura', 'Lobophora', 'Dictyota', 'Jania', 'Gracilaria', 'Amphiroa', 'Martensia'}
CRYPTOPHYTA_TAXA = {'Leucocryptos', 'Plagioselmis', 'Hemiselmis', 'Cryptomonadales', 'Rhodomonas', 'Teleaulax', 'Cryptophyta'}
OCHROPHYTA_TAXA = BACILLARIOPHYCEAE_TAXA | {'Amphora', 'Laminaria', 'Cerataulina', 'Meuniera', 'Actinocyclus', 'Odontella', 'Grammatophora', 'Anaulus', 'Bacillariophyceae', 'Cylindrial diatom', 'Cymbella', 'Desmarestia', 'Dictyopteris sp2', 'Dictyosiphon', 'Fallacia', 'Pennales', 'Pinnularia', 'Plagiotropis', 'Pseudosolenia', 'Raphidophytes', 'Rhabdonema', 'Solenicola setigera', 'Striatella', 'Vaucheria', 'Young laminariales', 'Rossia'}
RHIZARIA_TAXA = {'Halyphysema', 'Foraminifera', 'Rotaliida', 'Globothalamea', 'Astrorhizida', 'Radiozoa', 'Ebria tripartita', 'Arthracanthida', 'Difflugia', 'Sagenoarium', 'Trisulcus undet'}
EUGLENOZOA_TAXA = {'Euglena', 'Phacus', 'Eutreptiella', 'Trachelomonas'}
CYANOBACTERIA_TAXA = {'Anabaena', 'Trichodesmium'}
BACTERIA_TAXA = CYANOBACTERIA_TAXA | {'Bacteria', 'Cyanobacteria'}
MYRIAPODA_TAXA = {'Spirobolida', 'Scolopendromorpha', 'Lithobiomorpha', 'Julida sp'}
PANCRYPTISTA_TAXA = {'Cryptomonas'}
FUNGI_TAXA = {'Ascomycota', 'Basidiomycota'}
MAMMALIA_TAXA = {'Pipistrellus'}
LISSAMPHIBIA_TAXA = {'Amphibia'}
CHAETOGNATHA_TAXA = {'Sagitta'}
AGNATHA_TAXA = {'Petromyzonti', 'Myxini'}
CEPHALOCHORDATA_TAXA = {'Branchiostoma'}
GASTROTRICHA_TAXA = {'Gastrotricha'}
PRIAPULIDA_TAXA = {'Priapulus caudatus'}
ENTOPROCTA_TAXA = {'Entoprocta sp1', 'Pedicellina cernua'}
NEMATOMORPHA_TAXA = {'Nematomorpha'}
BRACHIOPODA_TAXA = {'Lingula'}
HAPTOPHYTA_TAXA = {'Coccolithophorid sp1', 'Coccolithophorid sp2', 'Coccolithophorid sp4', 'Coccolithophorid sp5', 'Coronosphaera', 'Coronosphaera sp', 'Coccolithophorid', 'Coccolithophore'}
MYZOZOA_TAXA = {'Dinomastigota', 'Karenia', 'Thoracosphaera'}
CILIOPHORA_TAXA = {'Favella'}
CHOANOFLAGELLATEA_TAXA = {'Bicosta'}
CHLOROPLASTIDA_TAXA = CHLOROPHYTA_TAXA | STREPTOPHYTA_TAXA | {'Prasinodermatophyta', 'Chlorophyta', 'Streptophyta', 'Coccomyxa sp', 'Didymocystis', 'Sphaerocystis'}

TAXONOMIC_ORDER = [
    'Aves', 'Testudines', 'Crocodylia', 'Squamata', 'Sphenodontia', 'Mammalia', 'Lissamphibia', 'Actinopterygii', 'Chondrichthyes', 'Agnatha', 'Cephalochordata', 'Tunicata', 'Hemichordata', 'Echinodermata', 'Chaetognatha', 'Mollusca', 'Annelida', 'Nemertea', 'Brachiopoda', 'Phoronida', 'Platyhelminthes', 'Nematoda', 'Hexapoda', 'Remipedia', 'Branchiopoda', 'Cephalocarida', 'Malacostraca', 'Maxillopoda', 'Thecostraca', 'Oligostraca', 'Myriapoda', 'Chelicerata', 'Onychophora', 'Kinorhyncha', 'Cnidaria', 'Ctenophora', 'Bryozoa', 'Porifera', 'Rotifera', 'Choanoflagellatea', 'Fungi', 'Tracheophyta', 'Bryophyta', 'Chloroplastida', 'Rhodophyta', 'Cryptophyta', 'Haptophyta', 'Ochrophyta', 'Pancryptista', 'Rhizaria', 'Myzozoa', 'Euglenozoa', 'Ciliophora', 'Bacteria', 'Gastrotricha', 'Priapulida', 'Entoprocta', 'Nematomorpha', 'Other'
]

EXCLUDED_GROUPS = {
    'Bacteria', 'Cyanobacteria', 'Euglenozoa', 'Myzozoa', 'Rhizaria', 'Ciliophora',
    'Cryptophyta', 'Haptophyta', 'Ochrophyta', 'Chloroplastida', 'Rhodophyta',
    'Pancryptista', 'Choanoflagellatea', 'Fungi', 'Archaea', 'Viruses', 'Virus',
}

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

def parse_quarter(q):
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

def load_biotime_data(filepath, min_time=1900):
    os.makedirs('csv', exist_ok=True)
    encodings = ['latin-1', 'ISO-8859-1', None]
    used_encoding = None
    header = None
    for enc in encodings:
        try:
            if enc is None:
                header = pd.read_csv(filepath, nrows=0)
            else:
                header = pd.read_csv(filepath, encoding=enc, nrows=0)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if header is None:
        raise UnicodeDecodeError("Could not determine encoding for BioTIME file")
    header.columns = header.columns.str.strip().str.upper()
    required = ["YEAR", "ABUNDANCE", "VALID_NAME", "STUDY_ID", "LATITUDE", "LONGITUDE"]
    missing = [col for col in required if col not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns in BioTIME data: {missing}")
    chunks = []
    chunk_iter = pd.read_csv(filepath, encoding=used_encoding, chunksize=100000)
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.strip().str.upper()
        chunk = chunk.dropna(subset=["YEAR", "ABUNDANCE"])
        chunk["YEAR"] = pd.to_numeric(chunk["YEAR"], errors="coerce")
        time_val = chunk["YEAR"].copy()
        if "MONTH" in chunk.columns:
            chunk["MONTH"] = pd.to_numeric(chunk["MONTH"], errors="coerce")
            mask_month = chunk["MONTH"].notna()
            time_val.loc[mask_month] = chunk.loc[mask_month, "YEAR"] + (chunk.loc[mask_month, "MONTH"].astype(int) - 1) / 12.0
            if "DAY" in chunk.columns:
                chunk["DAY"] = pd.to_numeric(chunk["DAY"], errors="coerce")
                mask_day = chunk["DAY"].notna()
                mask = mask_month & mask_day
                time_val.loc[mask] += (chunk.loc[mask, "DAY"].astype(int) - 1) / 365.0
        chunk["TIME"] = time_val
        chunk = chunk[chunk["TIME"] >= min_time]
        chunk["SPECIES_ID"] = chunk["VALID_NAME"]
        chunk["DATASET"] = "BioTIME"
        chunk["TIMESERIES_ID"] = chunk["STUDY_ID"].astype(str)
        chunk = chunk[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]]
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)

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
    df = df.groupby(["SPECIES_ID", "TIME"], as_index=False).agg({"ABUNDANCE": "sum"})
    if "TIMESERIESID" in df.columns:
        df["TIMESERIES_ID"] = df["TIMESERIESID"].astype(str)
    else:
        df["TIMESERIES_ID"] = "RivFishTIME"
    df["DATASET"] = "RivFishTIME"
    df["STUDY_ID"] = "RivFishTIME"
    df["LATITUDE"] = None
    df["LONGITUDE"] = None
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]]

def load_cedar_creek_data(e120_plant_path, e120_arthropod_path, e244_path):
    cedar_dfs = []
    try:
        e120_plant_df = pd.read_csv(e120_plant_path)
        year_col = find_column(e120_plant_df, ['year', 'YEAR', 'Year'])
        biomass_col = find_column(e120_plant_df, ['biomass (g/m2)', 'BIOMASS (G/M2)', 'Biomass (g/m2)', 'biomass', 'BIOMASS', 'Biomass', 'mass', 'MASS', 'Mass'])
        species_col = find_column(e120_plant_df, ['species', 'SPECIES', 'Species'])
        if year_col and biomass_col and species_col:
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
            e120_plant_df["DATASET"] = "CedarCreek"
            e120_plant_df["TIMESERIES_ID"] = "CedarCreek_e120_plant"
            e120_plant_df["STUDY_ID"] = "CedarCreek_e120_plant"
            e120_plant_df["LATITUDE"] = None
            e120_plant_df["LONGITUDE"] = None
            cedar_dfs.append(e120_plant_df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]])
    except Exception as e:
        print(f"Error loading E120 Plant Biomass: {e}")
    try:
        e120_arthropod_df = pd.read_csv(e120_arthropod_path, sep='\t')
        year_col = find_column(e120_arthropod_df, ['year', 'YEAR', 'Year'])
        count_col = find_column(e120_arthropod_df, ['count', 'COUNT', 'Count'])
        genus_col = find_column(e120_arthropod_df, ['genus', 'GENUS', 'Genus'])
        epithet_col = find_column(e120_arthropod_df, ['specific.epithet', 'SPECIFIC.EPITHET', 'Specific.epithet', 'specificepithet', 'specific epithet'])
        if year_col and count_col and genus_col and epithet_col:
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
            e120_arthropod_df["DATASET"] = "CedarCreek"
            e120_arthropod_df["TIMESERIES_ID"] = "CedarCreek_e120_arthropod"
            e120_arthropod_df["STUDY_ID"] = "CedarCreek_e120_arthropod"
            e120_arthropod_df["LATITUDE"] = None
            e120_arthropod_df["LONGITUDE"] = None
            cedar_dfs.append(e120_arthropod_df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]])
    except Exception as e:
        print(f"Error loading E120 Arthropod: {e}")
    try:
        e244_df = pd.read_csv(e244_path)
        year_col = find_column(e244_df, ['year', 'YEAR', 'Year'])
        mass_col = 'Mass g/m^2)'
        species_col = find_column(e244_df, ['species', 'SPECIES', 'Species'])
        if year_col and (mass_col in e244_df.columns) and species_col:
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
            e244_df["DATASET"] = "CedarCreek"
            e244_df["TIMESERIES_ID"] = "CedarCreek_e244"
            e244_df["STUDY_ID"] = "CedarCreek_e244"
            e244_df["LATITUDE"] = None
            e244_df["LONGITUDE"] = None
            cedar_dfs.append(e244_df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]])
    except Exception as e:
        print(f"Error loading E244 Biomass: {e}")
    if cedar_dfs:
        combined_df = pd.concat(cedar_dfs, ignore_index=True)
        print(f"Combined Cedar Creek data: {len(combined_df)} records")
        return combined_df
    else:
        print("No Cedar Creek data could be loaded")
        return pd.DataFrame()

def load_gbif_data(filepath):
    df = pd.read_csv(filepath, sep='\t')
    required = ['species', 'year', 'abundance']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in GBIF data: {missing}")
    df = df.rename(columns={'species': 'SPECIES_ID', 'year': 'TIME', 'abundance': 'ABUNDANCE'})
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['ABUNDANCE'] = pd.to_numeric(df['ABUNDANCE'], errors='coerce')
    df = df.dropna(subset=['TIME', 'ABUNDANCE'])
    df['STUDY_ID'] = 'GBIF'
    df['LATITUDE'] = None
    df['LONGITUDE'] = None
    df['DATASET'] = 'GBIF'
    df['TIMESERIES_ID'] = 'GBIF'
    for col in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = None
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID", "kingdom", "phylum", "class", "order", "family", "genus"]]

def load_gpdd_data(gpdd_final_path):
    df = pd.read_csv(gpdd_final_path)
    df = df.dropna(subset=['Species', 'Year', 'Abundance'])
    df['SPECIES_ID'] = df['Species'].str.strip()
    df['TIME'] = df['Year'].astype(float)
    df['ABUNDANCE'] = pd.to_numeric(df['Abundance'], errors='coerce')
    df = df.dropna(subset=['ABUNDANCE'])
    df['STUDY_ID'] = 'GPDD'
    df['DATASET'] = 'GPDD'
    lat = df['Latitude'].fillna(0).astype(float).round(2)
    lon = df['Longitude'].fillna(0).astype(float).round(2)
    df['TIMESERIES_ID'] = 'GPDD_' + lat.astype(str) + '_' + lon.astype(str)
    df['LATITUDE'] = df['Latitude']
    df['LONGITUDE'] = df['Longitude']
    tax_cols = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
    for col in tax_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = None
    return df[["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"] + [c for c in tax_cols if c in df.columns]]

def find_column(df, synonyms):
    for col in df.columns:
        col_clean = col.strip().lower()
        for syn in synonyms:
            if col_clean == syn.strip().lower():
                return col
    return None

def create_species_count_file(dataset_name, data, min_records=20):
    os.makedirs('csv', exist_ok=True)
    species_counts = data['SPECIES_ID'].value_counts().reset_index()
    species_counts.columns = ['species', 'count']
    species_min_n = species_counts[species_counts['count'] >= min_records]
    count_file = f'csv/{dataset_name}_species_count.csv'
    min_n_file = f'csv/{dataset_name}_species_min_n.csv'
    species_counts.to_csv(count_file, index=False)
    species_min_n[['species']].to_csv(min_n_file, index=False)
    return species_min_n['species'].tolist()

def build_data_taxonomy_from_dataframe(df):
    tax_dict = {}
    rank_order = ['class', 'order', 'phylum', 'kingdom']
    for sp, group in df.groupby('SPECIES_ID'):
        row = group.iloc[0]
        found = None
        for rank in rank_order:
            if rank in row and pd.notna(row[rank]) and str(row[rank]).strip() != '':
                val = str(row[rank]).strip()
                collapsed = collapse_taxonomic_group(val)
                if collapsed != 'Other':
                    found = collapsed
                    break
        if found is None:
            if 'genus' in row and pd.notna(row['genus']) and str(row['genus']).strip() != '':
                genus = str(row['genus']).strip()
                collapsed = collapse_taxonomic_group(genus)
                if collapsed != 'Other':
                    found = collapsed
        if found is not None:
            tax_dict[sp] = found
        else:
            tax_dict[sp] = 'Other'
    return tax_dict

def process_one_species_filter(species_name, full_data=None, taxon_mapping=None, data_taxonomy=None, genus_taxonomy=None, dataset_filter=None, failure_log_func=None, compute_quality=True, filter_func=None, species_data=None):
    if filter_func is None:
        raise ValueError("filter_func must be provided")
    if pd.isna(species_name) or species_name.strip() == '':
        return None
    if species_data is not None:
        sp_data = species_data.copy()
    elif full_data is not None:
        sp_data = full_data[full_data['SPECIES_ID'] == species_name].copy()
    else:
        raise ValueError("Either full_data or species_data must be provided")
    if len(sp_data) == 0:
        return None
    if dataset_filter is not None:
        sp_data = sp_data[sp_data['dataset'] == dataset_filter]
        if len(sp_data) == 0:
            return None
    sp_data = sp_data.sort_values('TIME')
    best_quality = -np.inf
    best_ts_id = None
    best_data = None
    failure_reason = None
    for ts_id, ts_group in sorted(sp_data.groupby('TIMESERIES_ID'), key=lambda x: str(x[0])):
        annual = ts_group.groupby(np.floor(ts_group['TIME']).astype(int))['ABUNDANCE'].mean().reset_index()
        annual.columns = ['YEAR', 'ABUNDANCE']
        annual['ABUNDANCE'] = annual['ABUNDANCE'].astype(float)
        annual = annual.dropna(subset=['ABUNDANCE'])
        passed, reason = filter_func(annual)
        if not passed:
            if failure_reason is None:
                failure_reason = reason
            continue
        if compute_quality:
            quality = compute_quality_score(annual, reliability=3)
            if best_ts_id is None or quality > best_quality or (abs(quality - best_quality) < 1e-12 and str(ts_id) < best_ts_id):
                best_quality = quality
                best_ts_id = str(ts_id)
                best_data = ts_group
        else:
            if best_data is None:
                best_data = ts_group
                break
    if best_data is None:
        if failure_reason is None:
            failure_reason = 'no_valid_timeseries'
        if failure_log_func is not None:
            failure_log_func(species_name, dataset_filter if dataset_filter is not None else 'Unknown', failure_reason)
        return None
    tax_group = get_cached_taxonomic_group(species_name, taxon_mapping=taxon_mapping, data_taxonomy=data_taxonomy,    force_recompute=False, genus_taxonomy=genus_taxonomy)
    if tax_group in EXCLUDED_GROUPS:
        if failure_log_func is not None:
            failure_log_func(species_name, dataset_filter if dataset_filter is not None else 'Unknown', 'taxonomic_excluded')
        return None
    return (species_name, best_data)

def build_genus_taxonomy(data_taxonomy):
    genus_tax = {}
    for sp, group in data_taxonomy.items():
        if group == 'Other':
            continue
        parts = sp.split()
        if len(parts) > 0:
            genus = parts[0]
            if genus not in genus_tax:
                genus_tax[genus] = group
    return genus_tax

def build_species_whitelist(biotime_path, min_year=1900, min_records=20, chunksize=100000):
    species_min_time = {}
    for chunk in pd.read_csv(biotime_path, chunksize=chunksize, encoding='latin-1'):
        chunk.columns = chunk.columns.str.strip().str.upper()
        if "YEAR" not in chunk.columns or "VALID_NAME" not in chunk.columns:
            continue
        chunk = chunk.dropna(subset=["YEAR", "VALID_NAME"])
        chunk["YEAR"] = pd.to_numeric(chunk["YEAR"], errors="coerce")
        time_val = chunk["YEAR"].copy()
        if "MONTH" in chunk.columns:
            chunk["MONTH"] = pd.to_numeric(chunk["MONTH"], errors="coerce")
            mask_month = chunk["MONTH"].notna()
            time_val.loc[mask_month] = chunk.loc[mask_month, "YEAR"] + (chunk.loc[mask_month, "MONTH"] - 1) / 12.0
        valid = time_val >= min_year
        chunk = chunk[valid]
        for _, row in chunk.iterrows():
            sp = row["VALID_NAME"]
            year = row["YEAR"]
            if sp not in species_min_time:
                species_min_time[sp] = {"min": year, "max": year, "count": 1}
            else:
                species_min_time[sp]["min"] = min(species_min_time[sp]["min"], year)
                species_min_time[sp]["max"] = max(species_min_time[sp]["max"], year)
                species_min_time[sp]["count"] += 1
    whitelist = [sp for sp, info in species_min_time.items() if info["count"] >= min_records]
    return whitelist

def load_combined_data(biotime_path, rivfishtime_path=None, cedar_creek_paths=None, gbif_path=None, gpdd_path=None, min_records=20, use_existing_biotime=True, force_reload=False, prefilter=True, filter_func=None, min_time=1900, whitelist=None):
    if prefilter and filter_func is None:
        raise ValueError("filter_func must be provided when prefilter=True")
    cache_dir = 'csv'
    os.makedirs(cache_dir, exist_ok=True)
    filtered_data_path = os.path.join(cache_dir, 'filtered_full_data.csv')
    filtered_species_path = os.path.join(cache_dir, 'filtered_species_list.csv')
    data_taxonomy_path = os.path.join(cache_dir, 'data_taxonomy.pkl')
    genus_taxonomy_path = os.path.join(cache_dir, 'genus_taxonomy.pkl')
    if prefilter and not force_reload and os.path.exists(filtered_data_path) and os.path.exists(filtered_species_path) and os.path.exists(data_taxonomy_path) and os.path.exists(genus_taxonomy_path):
        print("Loading cached filtered data...")
        full_data = pd.read_csv(filtered_data_path)
        species_list = pd.read_csv(filtered_species_path)['species'].tolist()
        with open(data_taxonomy_path, 'rb') as f:
            data_taxonomy = pickle.load(f)
        with open(genus_taxonomy_path, 'rb') as f:
            genus_taxonomy = pickle.load(f)
        return full_data, species_list, data_taxonomy, genus_taxonomy
    if whitelist is None:
        post1900_file = 'csv/species_post1900.csv'
        if os.path.exists(post1900_file) and os.path.getsize(post1900_file) > 0:
            whitelist_df = pd.read_csv(post1900_file)
            if 'species' in whitelist_df.columns:
                whitelist = set(whitelist_df['species'].tolist())
                print(f"Loaded whitelist of {len(whitelist)} species from {post1900_file}")
    print("Loading all datasets for filtering...")
    all_dfs = []
    if use_existing_biotime and os.path.exists('csv/biotime_species_count.csv'):
        print("Using existing BioTIME preprocessing...")
        biotime_data = load_biotime_data(biotime_path, min_time=min_time)
    else:
        biotime_data = load_biotime_data(biotime_path, min_time=min_time)
    if not biotime_data.empty:
        if whitelist is not None:
            biotime_data = biotime_data[biotime_data['SPECIES_ID'].isin(whitelist)]
        all_dfs.append(biotime_data)
    if rivfishtime_path and os.path.exists(rivfishtime_path):
        rivfishtime_data = load_rivfishtime_data(rivfishtime_path)
        if not rivfishtime_data.empty:
            if whitelist is not None:
                rivfishtime_data = rivfishtime_data[rivfishtime_data['SPECIES_ID'].isin(whitelist)]
            all_dfs.append(rivfishtime_data)
    if cedar_creek_paths is not None:
        cedar_creek_data = load_cedar_creek_data(cedar_creek_paths.get('e120_plant'), cedar_creek_paths.get('e120_arthropod'), cedar_creek_paths.get('e244'))
        if not cedar_creek_data.empty:
            if whitelist is not None:
                cedar_creek_data = cedar_creek_data[cedar_creek_data['SPECIES_ID'].isin(whitelist)]
            all_dfs.append(cedar_creek_data)
    if gbif_path and os.path.exists(gbif_path):
        try:
            gbif_data = load_gbif_data(gbif_path)
            if not gbif_data.empty:
                if whitelist is not None:
                    gbif_data = gbif_data[gbif_data['SPECIES_ID'].isin(whitelist)]
                all_dfs.append(gbif_data)
        except Exception as e:
            print(f"Warning: GBIF data loading failed: {e}")
    if gpdd_path and os.path.exists(gpdd_path):
        try:
            gpdd_data = load_gpdd_data(gpdd_path)
            if not gpdd_data.empty:
                if whitelist is not None:
                    gpdd_data = gpdd_data[gpdd_data['SPECIES_ID'].isin(whitelist)]
                all_dfs.append(gpdd_data)
        except Exception as e:
            print(f"Warning: GPDD data loading failed: {e}")
    if not all_dfs:
        raise ValueError("No data loaded from any source.")
    full_data = pd.concat(all_dfs, ignore_index=True)
    print(f"Total raw records: {len(full_data)}")
    species_list_raw = full_data['SPECIES_ID'].unique()
    print(f"Unique species before taxonomic filtering: {len(species_list_raw)}")
    tax_dfs = []
    if gbif_path and os.path.exists(gbif_path):
        try:
            gbif_df = load_gbif_data(gbif_path)
            if not gbif_df.empty:
                tax_dfs.append(gbif_df)
        except: pass
    if gpdd_path and os.path.exists(gpdd_path):
        try:
            gpdd_df = load_gpdd_data(gpdd_path)
            if not gpdd_df.empty:
                tax_dfs.append(gpdd_df)
        except: pass
    if tax_dfs:
        combined_tax_df = pd.concat(tax_dfs, ignore_index=True)
        data_taxonomy = build_data_taxonomy_from_dataframe(combined_tax_df)
    else:
        data_taxonomy = {}
    genus_taxonomy = build_genus_taxonomy(data_taxonomy)
    print(f"Built data_taxonomy for {len(data_taxonomy)} species from GPDD/GBIF.")
    print(f"Built genus_taxonomy for {len(genus_taxonomy)} genera.")
    if not prefilter:
        return full_data, list(species_list_raw), data_taxonomy, genus_taxonomy
    taxon_mapping = {}
    if os.path.exists('csv/taxon_mapping.csv') and os.path.getsize('csv/taxon_mapping.csv') > 0:
        try:
            taxon_df = pd.read_csv('csv/taxon_mapping.csv')
            taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
        except: pass
    # Pre‑group the full_data by SPECIES_ID for fast lookup
    species_groups = {sp: group for sp, group in full_data.groupby('SPECIES_ID')}
    checkpoint_file = os.path.join(cache_dir, 'filter_checkpoint.pkl')
    kept_species = []
    processed_species = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            kept_species = checkpoint.get('kept_species', [])
            processed_species = checkpoint.get('processed_species', set())
        print(f"Loaded checkpoint: {len(kept_species)} species kept, {len(processed_species)} species processed.")
    species_to_process = [sp for sp in species_list_raw if sp not in processed_species]
    print(f"Filtering species (microbial + Sibly/Brook) in parallel with checkpointing...")
    batch_size = 1000
    total_batches = (len(species_to_process) + batch_size - 1) // batch_size
    temp_data_path = os.path.join(cache_dir, 'temp_filtered_data.csv')
    first_write = True
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(species_to_process))
        batch = species_to_process[start:end]
        if not batch:
            continue
        print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch)} species)...")
        # For each species, get the pre‑computed group data
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(process_one_species_filter)(
                sp, taxon_mapping=taxon_mapping, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy, filter_func=filter_func, species_data=species_groups.get(sp)
            )
            for sp in tqdm(batch, desc=f"Batch {batch_idx+1}")
        )
        for res in results:
            if res is not None:
                sp, data = res
                data.to_csv(temp_data_path, mode='a', header=first_write, index=False)
                first_write = False
                kept_species.append(sp)
        processed_species.update(batch)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'kept_species': kept_species, 'processed_species': processed_species}, f)
        print(f"Checkpoint saved after batch {batch_idx+1}: {len(kept_species)} species kept.")
    if not kept_species:
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        raise ValueError("No species passed filters")
    filtered_data = pd.read_csv(temp_data_path)
    os.remove(temp_data_path)
    print(f"Records after microbial and Sibly/Brook filtering: {len(filtered_data)}")
    print(f"Species after filters: {len(kept_species)}")
    species_min_n = create_species_count_file('combined_filtered', filtered_data, min_records)
    selected_species = set(species_min_n)
    filtered_data = filtered_data[filtered_data['SPECIES_ID'].isin(selected_species)]
    print(f"Final species count after min_records={min_records}: {len(selected_species)}")
    filtered_data.to_csv(filtered_data_path, index=False)
    pd.DataFrame({'species': list(selected_species)}).to_csv(filtered_species_path, index=False)
    with open(data_taxonomy_path, 'wb') as f:
        pickle.dump(data_taxonomy, f)
    with open(genus_taxonomy_path, 'wb') as f:
        pickle.dump(genus_taxonomy, f)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("Cached filtered data saved.")
    return filtered_data, list(selected_species), data_taxonomy, genus_taxonomy

#def apply_filters_and_select_best(df, filter_func=None):
#    if filter_func is None:
#        raise ValueError("filter_func must be provided")
#    df = df.copy()
#    if df.empty:
#        return df
#    df['YEAR'] = np.floor(df['TIME']).astype(int)
#    grouped = df.groupby(['SPECIES_ID', 'TIMESERIES_ID'])
#    selected = []
#    for (species, ts_id), group in grouped:
#        group = group.sort_values('TIME')
#        annual = group.groupby('YEAR').agg({'ABUNDANCE': 'mean'}).reset_index()
#        annual['ABUNDANCE'] = annual['ABUNDANCE'].astype(float)
#        annual = annual.dropna(subset=['ABUNDANCE'])
#        passed, _ = filter_func(annual)
#        if not passed:
#            continue
#        quality = compute_quality_score(annual, reliability=3)
#        selected.append({
#            'SPECIES_ID': species,
#            'TIMESERIES_ID': ts_id,
#            'quality': quality,
#            'data': group
#        })
#    if not selected:
#        return pd.DataFrame()
#    best_per_species = {}
#    for item in selected:
#        sp = item['SPECIES_ID']
#        if sp not in best_per_species or item['quality'] > best_per_species[sp]['quality']:
#            best_per_species[sp] = item
#    best_dfs = []
#    for sp, item in best_per_species.items():
#        best_dfs.append(item['data'])
#    if not best_dfs:
#        return pd.DataFrame()
#    return pd.concat(best_dfs, ignore_index=True)

def compute_quality_score(annual_df, reliability=3):
    timesteps = len(annual_df)
    changes = sum(annual_df['ABUNDANCE'].diff().fillna(0) != 0)
    mean_val = np.mean(annual_df['ABUNDANCE'])
    std_val = np.std(annual_df['ABUNDANCE'])
    cv = std_val / mean_val if mean_val > 0 else 0
    time_score = np.log(timesteps + changes + 1)
    var_score = np.log(cv + 1) if cv > 0 else 0
    return reliability + time_score + 0.5 * var_score

def collapse_taxonomic_group(group):
    if group in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[group]
    return COLLAPSE_MAPPING.get(group, 'Other')

def get_cached_taxonomic_group(species_name, taxon_mapping=None, data_taxonomy=None, force_recompute=False, genus_taxonomy=None):
    cache_file = 'csv/taxonomic_cache.csv'
    os.makedirs('csv', exist_ok=True)
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return 'Other'
    species_str = str(species_name).strip()
    if 'miscellaneous' in species_str.lower() or 'litter' in species_str.lower() or 'unknown' in species_str.lower():
        return 'Other'
    if data_taxonomy is not None:
        if species_str in data_taxonomy:
            group = data_taxonomy[species_str]
            if group != 'Other':
                return group
    # Use lock for thread‑safe cache file access
    with cache_lock:
        cache_df = None
        if not force_recompute and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            try:
                cache_df = pd.read_csv(cache_file)
                if not cache_df.empty and 'species' in cache_df.columns and 'taxonomic_group' in cache_df.columns:
                    cached = cache_df[cache_df['species'] == species_name]
                    if not cached.empty:
                        cached_group = cached.iloc[0]['taxonomic_group']
                        if cached_group != 'Other':
                            return cached_group
            except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                # Cache is corrupted; remove it to let recreation happen
                try:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                except OSError:
                    pass
                cache_df = None
        if cache_df is None or cache_df.empty:
            cache_df = pd.DataFrame(columns=['species', 'taxonomic_group'])
        # Check taxon_mapping
        if taxon_mapping is not None:
            species_lower = species_name.lower()
            if species_lower in taxon_mapping:
                taxon_value = taxon_mapping[species_lower]
                taxonomic_group = collapse_taxonomic_group(taxon_value)
                if taxonomic_group != 'Other':
                    cache_df = cache_df[cache_df['species'] != species_name]
                    new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                    cache_df.to_csv(cache_file, index=False)
                    return taxonomic_group
        # Check for sp. etc.
        if 'sp' in species_str.lower() or 'cf.' in species_str.lower() or 'aff.' in species_str.lower():
            parts = species_str.split()
            if len(parts) > 0:
                genus = parts[0]
                if genus_taxonomy is not None and genus in genus_taxonomy:
                    taxonomic_group = genus_taxonomy[genus]
                    if taxonomic_group != 'Other':
                        cache_df = cache_df[cache_df['species'] != species_name]
                        new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        cache_df.to_csv(cache_file, index=False)
                        return taxonomic_group
                tax_info = get_taxonomic_info(genus)
                if tax_info:
                    taxonomic_group = assign_taxonomic_group(tax_info, genus)
                    if taxonomic_group != 'Other':
                        cache_df = cache_df[cache_df['species'] != species_name]
                        new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
                        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        cache_df.to_csv(cache_file, index=False)
                        return taxonomic_group
        # Fallback to GBIF classification
        taxonomic_group = classify_taxonomic_group(species_name, taxon_mapping)
        cache_df = cache_df[cache_df['species'] != species_name]
        new_row = pd.DataFrame({'species': [species_name], 'taxonomic_group': [taxonomic_group]})
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
        cache_df.to_csv(cache_file, index=False)
        return taxonomic_group

def apply_synonym_mapping(name):
    if not name or name == 'Other':
        return 'Other'
    if name in SYNONYM_MAPPING:
        return SYNONYM_MAPPING[name]
    for key in SYNONYM_MAPPING:
        if name.lower() == key.lower():
            return SYNONYM_MAPPING[key]
    if 'miscellaneous' in name.lower() or 'litter' in name.lower() or 'unknown' in name.lower():
        return 'Other'
    genus = name.split()[0] if name.split() else name
    if genus in SYNONYM_MAPPING:
        return SYNONYM_MAPPING[genus] + ' ' + ' '.join(name.split()[1:])
    return name

def clean_taxonomic_groups(name):
    if pd.isnull(name) or not isinstance(name, str) or name.strip() == '':
        return 'Other'
    cleaned = name.strip()
    if 'miscellaneous' in cleaned.lower() or 'litter' in cleaned.lower() or 'unknown' in cleaned.lower():
        return 'Other'
    cleaned = name.replace('_', ' ').strip()
    cleaned = re.sub(r'\s*sp(\.|\d+).*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned)
    cleaned = re.sub(r'[\d_]+$', '', cleaned).strip()
    if cleaned == '':
        return 'Other'
    return cleaned

@memory.cache
def get_taxonomic_info(scientific_name, depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return None
    base_name = apply_synonym_mapping(scientific_name)
    clean_name = clean_taxonomic_groups(base_name)
    if not clean_name or not clean_name.strip():
        return None
    search_strategies = []
    clean_words = clean_name.split()
    if len(clean_words) > 1:
        search_strategies.append({'name': clean_name, 'rank': 'SPECIES'})
    search_strategies.append({'name': clean_name})
    if len(clean_words) > 0:
        genus = clean_words[0]
        search_strategies.append({'name': genus, 'rank': 'GENUS'})
        search_strategies.append({'name': genus, 'rank': 'FAMILY'})
        search_strategies.append({'name': genus, 'rank': 'ORDER'})
        search_strategies.append({'name': genus, 'rank': 'CLASS'})
        search_strategies.append({'name': genus, 'rank': 'PHYLUM'})
    search_strategies.append({'name': clean_name, 'rank': None})
    for params in search_strategies:
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.gbif.org/v1/species/match", params={'name': params['name'], 'rank': params.get('rank')}, timeout=30
                )
                data = resp.json()
                if data.get('matchType') in ['EXACT', 'FUZZY', 'HIGHERRANK'] and data.get('confidence', 0) >= 80:
                    result = {
                        'order': data.get('order'), 'class': data.get('class'), 'subclass': data.get('subclass'), 'subphylum': data.get('subphylum'), 'phylum': data.get('phylum'), 'kingdom': data.get('kingdom'), }
                    higher_map = data.get('higherClassificationMap', {})
                    for rank in ['phylum', 'class', 'order', 'subphylum', 'family', 'kingdom']:
                        if not result.get(rank) and rank in higher_map:
                            result[rank] = higher_map[rank]
                    return result
                if data and 'higherClassificationMap' in data:
                    for rank in ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']:
                        if rank in data.get('higherClassificationMap', {}):
                            parent_name = data['higherClassificationMap'][rank]
                            if parent_name and parent_name != scientific_name:
                                result = get_taxonomic_info(parent_name, depth+1)
                                if result:
                                    return result
                time.sleep(API_DELAY)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"Connection error, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to connect to GBIF API after 3 attempts: {str(e)}")
                    return None
            except Exception as e:
                print(f"Error querying GBIF for {params['name']}: {str(e)}")
                time.sleep(API_DELAY)
                break
    if len(clean_words) > 1 and len(clean_words) > 0:
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
    if tax_info.get('class') and tax_info['class'] in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[tax_info['class']]
    for rank in ['subclass', 'order', 'subphylum', 'phylum', 'kingdom']:
        taxon = tax_info.get(rank)
        if taxon and taxon in COLLAPSE_MAPPING:
            return COLLAPSE_MAPPING[taxon]
    for part in mapped_name.split():
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

def plot_histogram(data, filename):
    set_plot_style()
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    total = data.sum()
    percentages = (data / total) * 100
    sorted_models = sorted(data.index)
    cmap = plt.get_cmap('Set1')
    color_map = {model: cmap(i % 10) for i, model in enumerate(sorted_models)}
    colors = [color_map[model] for model in data.index]
    bars = ax.bar(data.index, percentages, color=colors, alpha=0.65)
    for bar, count in zip(bars, data.values):
        height = bar.get_height()
        if height > 0:
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + height / 2
            ax.text(x_pos, y_pos, f'{int(count)}', ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    plt.xlabel(None)
    plt.ylabel('Percentage of Total (%)')
    x_labels = [abbreviate_bifurcation_type(str(label)) for label in data.index]
    split_labels = []
    for label in x_labels:
        words = label.split()
        if len(words) > 1:
            mid = len(words) // 2
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

def get_results_for_figure(threshold=0.0):
    short_file = 'csv/bifurcation_results_short.csv'
    if os.path.exists(short_file) and os.path.getsize(short_file) > 0:
        print("Using bifurcation_results_short.csv for figure generation")
        df = read_results_with_fixed_columns(short_file)
    else:
        print("bifurcation_results_short.csv not found, using bifurcation_results.csv")
        df = read_results_with_fixed_columns('csv/bifurcation_results.csv')
    if 'best_probability' in df.columns:
        df = df[df['best_probability'] > threshold]
    if 'taxonomic_group' in df.columns:
        df['taxonomic_group'] = df['taxonomic_group'].apply(collapse_taxonomic_group)
    return df

def create_summary_figure(results_df):
    if 'tipping_point_type' not in results_df.columns or 'taxonomic_group' not in results_df.columns:
        print("Warning: Could not create summary figure - missing required columns")
        return
    plot_df = results_df.copy()
    tipping_point_order = ['no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    tipping_point_colors = {
        'no-bifurcation': '#999999',
        'fold bifurcation': '#1f77b4',
        'transcritical bifurcation': '#ff7f0e',
        'Hopf bifurcation': '#9467bd'
    }
    summary = plot_df.groupby(['taxonomic_group', 'tipping_point_type']).size().unstack(fill_value=0)
    existing_columns = [col for col in tipping_point_order if col in summary.columns]
    summary = summary[existing_columns]
    summary = summary.reindex(TAXONOMIC_ORDER, fill_value=0)
    summary = summary.loc[summary.sum(axis=1) > 0]
    total_row = summary.sum(axis=0)
    total_row.name = 'Total'
    summary_with_total = pd.concat([summary, total_row.to_frame().T])
    print("\nTaxonomic group summary (counts of bifurcation types):")
    print("-" * 80)
    header = "Taxonomic Group".ljust(25) + "".join([t.ljust(22) for t in ['no-bif', 'fold', 'transc', 'Hopf']]) + "Total"
    print(header)
    print("-" * 80)
    for group in summary_with_total.index:
        row = summary_with_total.loc[group]
        total = row.sum()
        counts_str = (
            str(row.get('no-bifurcation', 0)).rjust(6) + " " +
            str(row.get('fold bifurcation', 0)).rjust(6) + " " +
            str(row.get('transcritical bifurcation', 0)).rjust(6) + " " +
            str(row.get('Hopf bifurcation', 0)).rjust(6)
        )
        print(group.ljust(25) + counts_str + "   " + str(total))
    print("-" * 80)
    print(f"Total species across all groups: {summary.sum().sum()}\n")
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
        percentages = (values / row_totals.replace(0, np.nan)) * 100
        percentages = percentages.fillna(0).values
        color = tipping_point_colors.get(col, '#1f77b4')
        bar = ax.bar(summary.index, percentages, bottom=bottom, color=color, alpha=0.65, label=abbreviate_bifurcation_type(col))
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
    plt.legend(title='Bifurcation Type', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(False)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/tipping_point_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def adjust_series_size(series_df, max_size=1500):
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
    if n_points == 1:
        return 1000
    score = 0
    score -= (n_points - 1) * 100
    for cp in change_points:
        if 10 <= cp <= len(data) - 10:
            before_mean = np.mean(data[cp-10:cp])
            after_mean = np.mean(data[cp:cp+10])
            mean_diff = abs(after_mean - before_mean)
            score += mean_diff * 50
    if method == "Pelt":
        score += 5
    elif method == "Window":
        score += 10
    return score

def find_transition(series_df, species, variable="ABUNDANCE", plot_fig=False):
    points = series_df[variable].values
    years = series_df['YEAR'].values
    if len(points) < 10:
        print(f"Short series detected for {species}, skipping")
        return None
    if np.std(points) < 1e-10:
        print(f"Constant series detected for {species}, skipping")
        return None
    median_val = np.median(points)
    q25, q75 = np.percentile(points, [25, 75])
    iqr = q75 - q25
    if median_val > 0 and iqr < 0.05 * median_val:
        print(f"Near-constant series detected for {species}, skipping")
        return None
    best_transition_index = None
    best_method = None
    best_params = None
    best_score = -np.inf
    dynp_models = ["l2", "rbf", "linear", "normal", "ar"]
    for model in dynp_models:
        try:
            algo = rpt.Dynp(model=model, min_size=5, jump=1).fit(points)
            bkps = algo.predict(n_bkps=1)
            valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
            score = score_change_points(valid_bkps, points, method="Dynp")
            if score > best_score:
                best_score = score
                best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                best_method = "Dynp"
                best_params = {"model": model, "n_bkps": 1}
            if len(valid_bkps) == 1:
                print(f"Perfect single change point found with Dynp ({model}) for {species}")
                transition_index = valid_bkps[0]
                selected_method = f"Dynp_{model}_nbps1"
                transition_year = series_df['YEAR'].iloc[transition_index]
                print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                return transition_index
        except Exception as e:
            continue
    pelt_models = ["rbf", "l2", "linear", "normal", "ar"]
    penalties = [1, 2, 3, 5, 10]
    for model in pelt_models:
        for pen in penalties:
            try:
                algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(points)
                penalty_value = pen * np.log(len(points)) * np.var(points)
                bkps = algo.predict(pen=penalty_value)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="Pelt")
                if score > best_score:
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "Pelt"
                    best_params = {"model": model, "penalty": pen}
                if len(valid_bkps) == 1:
                    print(f"Perfect single change point found with Pelt ({model}, pen={pen}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"Pelt_{model}_pen{pen}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    binseg_models = ["l2", "rbf", "linear", "normal", "ar"]
    n_bkps_range = [1, 2, 3, 4, 5]
    for model in binseg_models:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Binseg(model=model, min_size=5).fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="BinSeg")
                if score > best_score:
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "BinSeg"
                    best_params = {"model": model, "n_bkps": n_bkps}
                if len(valid_bkps) == 1:
                    print(f"Perfect single change point found with BinSeg ({model}, n_bkps={n_bkps}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"BinSeg_{model}_nbps{n_bkps}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    window_sizes = [20, 30, 40, 50, 60]
    for window_size in window_sizes:
        for n_bkps in n_bkps_range:
            try:
                algo = rpt.Window(width=window_size, model="l2").fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
                valid_bkps = [b for b in bkps if 5 <= b <= (len(points) - 5) and b != len(points)]
                score = score_change_points(valid_bkps, points, method="Window")
                if score > best_score:
                    best_score = score
                    best_transition_index = valid_bkps[0] if len(valid_bkps) == 1 else None
                    best_method = "Window"
                    best_params = {"width": window_size, "n_bkps": n_bkps}
                if len(valid_bkps) == 1:
                    print(f"Perfect single change point found with Window (width={window_size}, n_bkps={n_bkps}) for {species}")
                    transition_index = valid_bkps[0]
                    selected_method = f"Window_width{window_size}_nbps{n_bkps}"
                    transition_year = series_df['YEAR'].iloc[transition_index]
                    print(f"Transition Value (year) for {species}: {transition_year} (Method: {selected_method})")
                    return transition_index
            except Exception as e:
                continue
    if best_transition_index is not None:
        transition_index = best_transition_index
        selected_method = f"{best_method}_{best_params}"
        transition_year = series_df['YEAR'].iloc[transition_index]
        print(f"Best available transition for {species}: {transition_year} (Method: {selected_method}, Score: {best_score:.2g})")
        return transition_index
    print(f"No transition found for {species}, skipping")
    return None

#def extract_pre_transition_segment(series_df, transition_index, max_size=1500):
#    abundance = series_df['ABUNDANCE'].values
#    years = series_df['YEAR'].values
#    if transition_index <= 0:
#        transition_index = len(abundance) // 2
#    pre_abundance = abundance[:transition_index]
#    pre_years = years[:transition_index]
#    if len(pre_abundance) >= max_size:
#        pre_abundance = pre_abundance[-max_size:]
#        pre_years = pre_years[-max_size:]
#    else:
#        pad_len = max_size - len(pre_abundance)
#        pre_abundance = np.concatenate([np.zeros(pad_len), pre_abundance])
#        if len(pre_years) > 0:
#            pre_years = np.concatenate([np.full(pad_len, pre_years[0]), pre_years])
#        else:
#            pre_years = np.concatenate([np.zeros(pad_len), pre_years])
#    return pd.DataFrame({'YEAR': pre_years, 'ABUNDANCE': pre_abundance})

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
    if method_preference == 'Lowess':
        methods_to_try = [
            ('Lowess', {'span': 0.2}), ('Lowess', {'span': min(0.3, max(0.1, 50/series_length))}), ('Lowess', {'span': 0.15}), ('Lowess', {'span': 0.25}), ]
    else:
        methods_to_try = [
            ('Gaussian', {'bandwidth': 0.2}), ('Gaussian', {'bandwidth': min(0.3, max(0.1, 50/series_length))}), ('Gaussian', {'bandwidth': 0.15}), ('Gaussian', {'bandwidth': 0.25}), ]
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
            if 'smoothing' in ts.state.columns:
                smoothing_vals = ts.state['smoothing'].values
                original_vals = ts.state['state'].values
                valid_mask = ~np.isnan(smoothing_vals) & ~np.isnan(original_vals)
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(original_vals[valid_mask], smoothing_vals[valid_mask])[0,1]
                    orig_variance = np.var(original_vals[valid_mask])
                    resid_variance = np.var(original_vals[valid_mask] - smoothing_vals[valid_mask])
                    if abs(corr) > 0.95 and resid_variance < (0.05 * orig_variance):
                        print(f"Overfitting detected with {method} {params}, trying next method...")
                        ts.state['smoothing'] = np.nan
                        ts.state['residuals'] = ts.state['state'].copy()
                        continue
                    if len(smoothing_vals[valid_mask]) > 20:
                        diff1 = np.diff(smoothing_vals[valid_mask])
                        diff2 = np.diff(diff1)
                        inflection_count = np.sum(np.abs(diff2) > np.std(diff2) * 2)
                        inflection_ratio = inflection_count / len(diff2)
                        if inflection_ratio > 0.3:
                            print(f"Overly wiggly smoothing with {method} {params}, trying next method...")
                            ts.state['smoothing'] = np.nan
                            ts.state['residuals'] = ts.state['state'].copy()
                            continue
                print(f"Successfully detrended using {method} with params {params}")
                return True
        except Exception as e:
            last_error = e
            if 'smoothing' in ts.state.columns:
                ts.state['smoothing'] = np.nan
                ts.state['residuals'] = ts.state['state'].copy()
            continue
    print(f"All detrending methods failed. Last error: {last_error}")
    return False

def deseason_residuals(ts, time_values):
    if 'residuals' not in ts.state.columns:
        return
    residuals = ts.state['residuals'].values.copy()
    frac = time_values - np.floor(time_values)
    unique_frac = np.unique(np.round(frac, 3))
    if len(unique_frac) == 12 and np.all(np.diff(np.sort(unique_frac)) < 0.1):
        period = 12
    else:
        return
    if len(residuals) < 2 * period:
        return
    result = sm.tsa.seasonal_decompose(residuals, model='additive', period=period, extrapolate_trend='freq')
    seasonal = result.seasonal
    if seasonal is None or np.any(np.isnan(seasonal)):
        raise ValueError("Seasonal decomposition produced NaNs")
    ts.state['residuals'] = residuals - seasonal

#def process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, data_taxonomy=None, genus_taxonomy=None, output_file='csv/bifurcation_results.csv'):
#    try:
#        setup_warning_suppression(worker_process=True)
#        taxonomic_group = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
#        dataset = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
#        series = species_data[["TIME", "ABUNDANCE"]].copy()
#        series.columns = ["YEAR", "ABUNDANCE"]
#        original_length = len(series)
#        raw_abundance = species_data["ABUNDANCE"]
#        if original_length <= 500:
#            series_adjusted = adjust_series_size(series, max_size=500)
#            classifier_paths = classifier_paths_500
#            max_size = 500
#        else:
#            series_adjusted = adjust_series_size(series, max_size=1500)
#            classifier_paths = classifier_paths_1500
#            max_size = 1500
#        transition_idx = find_transition(series_adjusted, species_name)
#        if transition_idx is None:
#            smoothing_avg = None
#            return {
#                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
#            }
#        if transition_idx >= original_length:
#            transition_idx = original_length - 1
#        change_points = [transition_idx] if transition_idx is not None else []
#        ts = ewstools.TimeSeries(series_adjusted['ABUNDANCE'].values, transition=transition_idx)
#        ts.time_values = series_adjusted['YEAR'].values
#        ts.state['year'] = series_adjusted['YEAR'].values
#        detrend_success = robust_detrending(ts, original_length, method_preference='Lowess')
#        if not detrend_success:
#            raise ValueError("Failed to detrend time series")
#        if 'smoothing' not in ts.state.columns:
#            raise ValueError("Smoothing vector not found - cannot calculate smoothing average")
#        smoothing_vals = ts.state['smoothing'].values
#        valid_indices = ~np.isnan(smoothing_vals)
#        valid_vals = smoothing_vals[valid_indices]
#        if len(valid_vals) > 0:
#            last_fifth_idx = max(0, len(valid_vals) - len(valid_vals) // 5)
#            smoothing_avg = np.mean(valid_vals[last_fifth_idx:])
#        else:
#            smoothing_avg = None
#        if smoothing_avg is not None and abs(smoothing_avg - 1.0) < 1.0e-5:
#            return {
#                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
#            }
#        if transition_idx is None or transition_idx < 0 or transition_idx >= len(series_adjusted):
#            return {
#                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
#            }
#        eval_index = int(0.9 * transition_idx)
#        if eval_index < 0:
#            eval_index = 0
#        ts.clear_dl_preds()
#        for classifier_path in classifier_paths:
#            classifier = load_model(classifier_path)
#            ts.apply_classifier_inc(classifier, inc=10, verbose=0, name=os.path.basename(classifier_path))
#            tf.keras.backend.clear_session()
#            gc.collect()
#        best_probability = 0.0
#        raw_bifurcation_type = 'no-bifurcation'
#        mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
#        if len(ts.dl_preds) > 0:
#            transition_time = series_adjusted['YEAR'].iloc[transition_idx]
#            valid_preds = ts.dl_preds[ts.dl_preds['time'] <= transition_time]
#            if len(valid_preds) > 0:
#                transition_pred = valid_preds.iloc[-1]
#                transition_time_used = transition_pred['time']
#                ensemble_preds_at_time = ts.dl_preds[ts.dl_preds['time'] == transition_time_used]
#                numeric_cols = [0, 1, 2, 3]
#                if len(ensemble_preds_at_time) > 1:
#                    mean_probs = ensemble_preds_at_time[numeric_cols].astype(float).mean()
#                else:
#                    mean_probs = transition_pred[numeric_cols].astype(float)
#                best_probability = mean_probs.max()
#                pred_class = mean_probs.idxmax()
#                raw_bifurcation_type = bifurcation_map.get(pred_class, 'no-bifurcation')
#                final_bifurcation_type = raw_bifurcation_type
#            else:
#                mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
#                raw_bifurcation_type = 'no-bifurcation'
#                final_bifurcation_type = 'no-bifurcation'
#                best_probability = 1.0
#        else:
#            mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
#            raw_bifurcation_type = 'no-bifurcation'
#            final_bifurcation_type = 'no-bifurcation'
#            best_probability = 1.0
#        transition_year = series['YEAR'].iloc[min(transition_idx, original_length-1)]
#        if (transition_idx is None or
#            transition_year is None or
#            pd.isna(transition_year) or
#            not change_points or
#            final_bifurcation_type in ['null', 'error'] or
#            len(ts.dl_preds) == 0):
#            final_bifurcation_type = 'no-bifurcation'
#            transition_year = None
#            change_points = []
#            best_probability = 0.0
#        if species_to_plot is not None and (len(species_to_plot) == 0 or species_name in species_to_plot):
#            plot_dl_predictions(ts, species_name, eval_index)
#        pdf_species = ['Anaxyrus sp7', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
#        if species_name in pdf_species:
#            plot_dl_predictions_pdf(ts, species_name, eval_index)
#        ews_metrics = {int(k): float(v) for k, v in mean_probs.to_dict().items()} if not mean_probs.empty else {}
#        return {
#            'species': species_name, 'dataset': dataset, 'tipping_point_type': final_bifurcation_type, 'raw_tipping_point_type': raw_bifurcation_type, 'ews_metrics': ews_metrics, 'change_points': change_points, 'taxonomic_group': taxonomic_group, 'transition_year': transition_year, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': best_probability
#        }
#    except Exception as e:
#        print(f"Error processing {species_name}: {str(e)}")
#        traceback.print_exc()
#        tf.keras.backend.clear_session()
#        gc.collect()
#        taxonomic_group_error = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
#        dataset_error = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
#        return {
#            'species': species_name, 'dataset': dataset_error, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0
#        }
#    finally:
#        tf.keras.backend.clear_session()
#        gc.collect()

def process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, data_taxonomy=None, genus_taxonomy=None, output_file='csv/bifurcation_results.csv'):
    try:
        setup_warning_suppression(worker_process=True)
        taxonomic_group = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
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
        if transition_idx is None:
            smoothing_avg = None
            return {
                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
            }
        if transition_idx >= original_length:
            transition_idx = original_length - 1
        change_points = [transition_idx] if transition_idx is not None else []
        ts = ewstools.TimeSeries(series_adjusted['ABUNDANCE'].values, transition=transition_idx)
        ts.time_values = series_adjusted['YEAR'].values
        ts.state['year'] = series_adjusted['YEAR'].values
        detrend_success = robust_detrending(ts, original_length, method_preference='Lowess')
        if not detrend_success:
            raise ValueError("Failed to detrend time series")
        deseason_residuals(ts, series_adjusted['YEAR'].values)
        if 'smoothing' not in ts.state.columns:
            raise ValueError("Smoothing vector not found - cannot calculate smoothing average")
        smoothing_vals = ts.state['smoothing'].values
        valid_indices = ~np.isnan(smoothing_vals)
        valid_vals = smoothing_vals[valid_indices]
        if len(valid_vals) > 0:
            last_fifth_idx = max(0, len(valid_vals) - len(valid_vals) // 5)
            smoothing_avg = np.mean(valid_vals[last_fifth_idx:])
        else:
            smoothing_avg = None
        if smoothing_avg is not None and abs(smoothing_avg - 1.0) < 1.0e-5:
            return {
                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
            }
        if transition_idx is None or transition_idx < 0 or transition_idx >= len(series_adjusted):
            return {
                'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0
            }
        eval_index = int(0.9 * transition_idx)
        if eval_index < 0:
            eval_index = 0
        ts.clear_dl_preds()
        for classifier_path in classifier_paths:
            classifier = load_model(classifier_path)
            ts.apply_classifier_inc(classifier, inc=10, verbose=0, name=os.path.basename(classifier_path))
            tf.keras.backend.clear_session()
            gc.collect()
        best_probability = 0.0
        raw_bifurcation_type = 'no-bifurcation'
        mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
        if len(ts.dl_preds) > 0:
            transition_time = series_adjusted['YEAR'].iloc[transition_idx]
            valid_preds = ts.dl_preds[ts.dl_preds['time'] <= transition_time]
            if len(valid_preds) > 0:
                transition_pred = valid_preds.iloc[-1]
                transition_time_used = transition_pred['time']
                ensemble_preds_at_time = ts.dl_preds[ts.dl_preds['time'] == transition_time_used]
                numeric_cols = [0, 1, 2, 3]
                if len(ensemble_preds_at_time) > 1:
                    mean_probs = ensemble_preds_at_time[numeric_cols].astype(float).mean()
                else:
                    mean_probs = transition_pred[numeric_cols].astype(float)
                best_probability = mean_probs.max()
                pred_class = mean_probs.idxmax()
                raw_bifurcation_type = bifurcation_map.get(pred_class, 'no-bifurcation')
                final_bifurcation_type = raw_bifurcation_type
            else:
                mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
                raw_bifurcation_type = 'no-bifurcation'
                final_bifurcation_type = 'no-bifurcation'
                best_probability = 1.0
        else:
            mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
            raw_bifurcation_type = 'no-bifurcation'
            final_bifurcation_type = 'no-bifurcation'
            best_probability = 1.0
        transition_year = series['YEAR'].iloc[min(transition_idx, original_length-1)]
        if (transition_idx is None or
            transition_year is None or
            pd.isna(transition_year) or
            not change_points or
            final_bifurcation_type in ['null', 'error'] or
            len(ts.dl_preds) == 0):
            final_bifurcation_type = 'no-bifurcation'
            transition_year = None
            change_points = []
            best_probability = 0.0
        if species_to_plot is not None and (len(species_to_plot) == 0 or species_name in species_to_plot):
            plot_dl_predictions(ts, species_name, eval_index)
        pdf_species = ['Anaxyrus sp7', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
        if species_name in pdf_species:
            plot_dl_predictions_pdf(ts, species_name, eval_index)
        ews_metrics = {int(k): float(v) for k, v in mean_probs.to_dict().items()} if not mean_probs.empty else {}
        return {
            'species': species_name, 'dataset': dataset, 'tipping_point_type': final_bifurcation_type, 'raw_tipping_point_type': raw_bifurcation_type, 'ews_metrics': ews_metrics, 'change_points': change_points, 'taxonomic_group': taxonomic_group, 'transition_year': transition_year, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': best_probability
        }
    except Exception as e:
        print(f"Error processing {species_name}: {str(e)}")
        traceback.print_exc()
        tf.keras.backend.clear_session()
        gc.collect()
        taxonomic_group_error = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
        dataset_error = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
        return {
            'species': species_name, 'dataset': dataset_error, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0
        }
    finally:
        tf.keras.backend.clear_session()
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
    master_map['assigned_core'] = [i % n_jobs for i in range(len(species_list))]
    master_map['status'] = 'pending'
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

def save_results(species_name, bifurcation_type, transition_year, change_points, smoothing_avg, current_abundance, max_abundance, best_probability, taxonomic_group, dataset, output_file='csv/bifurcation_results.csv', raw_tipping_point_type=None, ews_metrics=None):
    os.makedirs('csv', exist_ok=True)
    if (bifurcation_type in ['error', 'null'] or
        transition_year is None or
        pd.isna(transition_year) or
        not change_points or
        len(change_points) == 0 or
        (smoothing_avg is not None and abs(smoothing_avg - 1.0) < 1.0e-5)):
        bifurcation_type = 'no-bifurcation'
        transition_year = None
        change_points = []
        best_probability = 0.0
    if raw_tipping_point_type is None:
        raw_tipping_point_type = bifurcation_type
    if ews_metrics is None:
        ews_metrics = {}
    new_result = pd.DataFrame({
        'species': [species_name],
        'dataset': [dataset],
        'taxonomic_group': [taxonomic_group],
        'tipping_point_type': [bifurcation_type],
        'raw_tipping_point_type': [raw_tipping_point_type],
        'ews_metrics': [json.dumps(ews_metrics) if ews_metrics else '{}'],
        'transition_year': [transition_year],
        'change_points': [str(change_points)],
        'smoothing_avg': [f"{smoothing_avg:.2g}" if smoothing_avg is not None else "NaN"],
        'current_abundance': [current_abundance],
        'max_abundance': [max_abundance],
        'best_probability': [best_probability]
    })
    try:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            results_df = read_results_with_fixed_columns(output_file)
            if species_name in results_df['species'].values:
                print(f"WARNING: {species_name} already exists in results. Skipping to prevent data loss.")
                return
            results_df = pd.concat([results_df, new_result], ignore_index=True)
        else:
            results_df = new_result
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved for {species_name} (Dataset: {dataset}, Taxonomic group: {taxonomic_group}, Probability: {best_probability:.2g}) to {output_file}")
    except Exception as e:
        print(f"Error saving results for {species_name} to {output_file}: {str(e)}")
        if not os.path.exists(output_file):
            new_result.to_csv(output_file, index=False, encoding='utf-8')

def save_batch_results(batch_results):
    results_file = 'csv/bifurcation_results.csv'
    if not batch_results:
        return
    try:
        batch_df = pd.DataFrame(batch_results)
        required_columns = ['species', 'taxonomic_group', 'tipping_point_type', 'transition_year',    'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
        for col in required_columns:
            if col not in batch_df.columns:
                if col == 'best_probability':
                    batch_df[col] = 0.0
                else:
                    batch_df[col] = None
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            existing_df = pd.read_csv(results_file)
            existing_df = existing_df[~existing_df['species'].isin(batch_df['species'])]
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
        else:
            combined_df = batch_df
        combined_df.to_csv(results_file, index=False)
        print(f"Batch results saved for {len(batch_results)} species (redundant safety save)")
    except Exception as e:
        print(f"Error saving batch results: {e}")
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
    for attempt in range(3):
        try:
            master_map = pd.read_csv(master_map_file)
            available_species = master_map[
                (master_map['assigned_core'] == core_id) &
                (master_map['status'] == 'pending')
            ]
            if len(available_species) == 0:
                return None
            batch = available_species.head(batch_size)
            species_names = batch['species'].tolist()
            master_map.loc[master_map['species'].isin(species_names), 'status'] = 'processing'
            master_map.loc[master_map['species'].isin(species_names), 'start_time'] = time.time()
            master_map.to_csv(master_map_file, index=False)
            print(f"Core {core_id} claimed batch: {species_names}")
            return species_names
        except Exception as e:
            print(f"Core {core_id} attempt {attempt + 1} failed to get batch: {e}")
            time.sleep(0.1)
    return None

def core_worker(core_id, full_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, batch_size=1, data_taxonomy=None, genus_taxonomy=None):
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
                    taxonomic_group_missing = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
                    save_results(species_name, 'no-bifurcation', None, [], None, 0, 0, 0.0, taxonomic_group_missing, 'Unknown', raw_tipping_point_type='no-bifurcation', ews_metrics={})
                    mark_batch_completed(core_id, [species_name])
                    continue
                result = process_species(
                    species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot, data_taxonomy, genus_taxonomy
                )
                batch_results.append(result)
                processed_count += 1
                print(f"Core {core_id} completed and saved: {species_name}")
            except Exception as e:
                print(f"Core {core_id} error processing {species_name}: {e}")
                taxonomic_group_error = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
                dataset_error = 'Unknown'
                try:
                    save_results(species_name, 'no-bifurcation', None, [], None, 0, 0, 0.0, taxonomic_group_error, dataset_error, raw_tipping_point_type='no-bifurcation', ews_metrics={})
                except Exception as save_error:
                    print(f"Core {core_id} failed to save error result for {species_name}: {save_error}")
                error_result = {
                    'species': species_name, 'tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0
                }
                batch_results.append(error_result)
                processed_count += 1
        mark_batch_completed(core_id, species_batch)
        if batch_results:
            save_batch_results(batch_results)
            print(f"Core {core_id} saved batch of {len(batch_results)} species")
    return processed_count

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

def run_analysis_pipeline(min_records=20, force_reclassification=False, species_to_plot=None, n_jobs=-1, batch_size=1, specific_species_list=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    classifier_base = os.path.join(script_dir, '..', 'saved_classifiers', 'bury_pnas_21')
    root_path_500 = os.path.join(classifier_base, 'len500')
    root_path_1500 = os.path.join(classifier_base, 'len1500')
    classifier_names_500 = [name for name in os.listdir(root_path_500) if name.endswith('.pkl')]
    classifier_paths_500 = [os.path.join(root_path_500, name) for name in classifier_names_500[:20]]
    classifier_names_1500 = [name for name in os.listdir(root_path_1500) if name.endswith('.pkl')]
    classifier_paths_1500 = [os.path.join(root_path_1500, name) for name in classifier_names_1500[:20]]
    bifurcation_map = {0: 'fold bifurcation', 1: 'Hopf bifurcation', 2: 'transcritical bifurcation', 3: 'no-bifurcation'}
    biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
    rivfishtime_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
    cedar_creek_paths = {'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'), 'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'), 'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')}
    gbif_path = os.path.join(data_dir, 'gbif', 'output.csv')
    gpdd_path = os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
    full_data, top_species, data_taxonomy, genus_taxonomy = load_combined_data(biotime_path, rivfishtime_path, cedar_creek_paths, gbif_path, gpdd_path, min_records=min_records, use_existing_biotime=True)
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
            results_df = get_results_for_figure()
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
            get_cached_taxonomic_group(species, taxon_mapping, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
    process_data = full_data[full_data['SPECIES_ID'].isin(species_to_process_final)]
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    print(f"Using {n_jobs} cores with batch size {batch_size}")
    master_map = create_master_map(species_to_process_final, n_jobs)
    core_worker_partial = partial(core_worker, full_data=process_data, classifier_paths_500=classifier_paths_500, classifier_paths_1500=classifier_paths_1500, bifurcation_map=bifurcation_map, species_to_plot=species_to_plot, batch_size=batch_size, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
    print(f"Starting parallel processing with {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(delayed(core_worker_partial)(core_id) for core_id in range(n_jobs))
    total_processed = sum(results)
    print(f"Parallel processing complete. Total species processed: {total_processed}")
    try:
        results_df = get_results_for_figure()
        create_summary_figure(results_df)
        pending, processing, completed = get_remaining_species_count()
        print(f"Final status - Pending: {pending}, Processing: {processing}, Completed: {completed}")
    except Exception as e:
        print(f"Error in final processing: {e}")
    return results_df

def generate_final_figures(probability_threshold=0.7):
    try:
        results_df = get_results_for_figure(threshold=probability_threshold)
        create_summary_figure(results_df)
        overall_counts = results_df['tipping_point_type'].value_counts()
        plot_histogram(overall_counts, 'img/overall_distribution.pdf')
    except Exception as e:
        print(f"Error in final figure generation: {e}")

def get_species_list_to_process(min_records, force_reclassification):
    results_file = 'csv/bifurcation_results.csv'
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        return None, False
    results_df = read_results_with_fixed_columns(results_file)
    combined_species_file = f'csv/combined_species_min{min_records}.csv'
    if not os.path.exists(combined_species_file):
        return None, False
    combined_species_df = pd.read_csv(combined_species_file, encoding='utf-8')
    if 'species' not in combined_species_df.columns:
        raise KeyError("No species column found in combined species file")
    combined_species_list = combined_species_df['species'].tolist()
    processed_species = set(results_df['species'].unique())
    combined_species_set = set(combined_species_list)
    processed_combined_species = processed_species & combined_species_set
    missing_species = combined_species_set - processed_combined_species
    print(f"Total species with >= {min_records} records: {len(combined_species_set)}")
    print(f"Already processed species: {len(processed_combined_species)}")
    print(f"Missing species: {len(missing_species)}")
    if len(missing_species) == 0:
        print(f"All species with >= {min_records} records already processed.")
        return [], True
    else:
        print(f"Found {len(missing_species)} species missing from results. Running analysis for missing species only...")
        if force_reclassification:
            print("Force reclassification enabled: will reprocess ALL species.")
            return None, False
        else:
            return list(missing_species), False

def main():
    setup_warning_suppression()
    backup_existing_results()
    update_taxonomic_cache_from_other()
    refresh_taxonomic_cache_from_results()
    print("Starting EWS analysis pipeline...")
    min_records = 20
    force_reclassification = False
    probability_threshold = 0.7
    species_list_to_process, all_done = get_species_list_to_process(min_records, force_reclassification)
    if all_done:
        generate_final_figures(probability_threshold)
        print("Figures generated from existing results!")
        return pd.read_csv('csv/bifurcation_results.csv')
    species_to_plot = []
    results = run_analysis_pipeline(
        min_records=min_records,
        force_reclassification=force_reclassification,
        species_to_plot=species_to_plot,
        n_jobs=-1,
        batch_size=1,
        specific_species_list=species_list_to_process
    )
    update_taxonomic_groups_from_mapping()
    generate_final_figures(probability_threshold)
    print("Tipping points analysis complete!")
    return results

if __name__ == "__main__":
    results = main()
