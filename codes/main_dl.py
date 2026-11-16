#!/usr/bin/env python
# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc, re, time, logging, requests, traceback, pickle, threading, json, hashlib, warnings
import ewstools
import absl.logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
cache_lock = threading.Lock()
MIN_CP_FRAC = 0.5
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
rivfishtime_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
cedar_creek_paths = {'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'), 'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'), 'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')}
gbif_path = os.path.join(data_dir, 'gbif', 'output.csv')
gpdd_path = os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
nps_path = os.path.join(data_dir, 'nps', 'snowy_plover_abundance.csv')
lpd_path = os.path.join(data_dir, 'livingplanet', 'lpd_abundance.csv')
glba_path = os.path.join(data_dir, 'glba', 'glba_abundance.csv')
vrdata_path = os.path.join(data_dir, 'VRData', 'vrdata_abundance.csv')
tsx_path = os.path.join(data_dir, 'tsxdata', 'tsx_abundance.csv')
parrot_path = os.path.join(data_dir, 'parrot', 'parrot_abundance.csv')
lizard_path = os.path.join(data_dir, 'lizard', 'lizard_abundance.csv')
primate_path = os.path.join(data_dir, 'primate', 'primate_abundance.csv')
frog_path = os.path.join(data_dir, 'frog', 'frog_abundance.csv')
akepa_path = os.path.join(data_dir, 'hawaii', 'akepa_abundance.csv')
banding_codes_path = os.path.join(data_dir, 'banding_codes.csv')

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
    'ROSAACIC': 'Tracheophyta',
    'Unidentified Fish': 'Actinopterygii',
    'Unidentified medusae': 'Cnidaria',
    'Unidentified rotifer': 'Rotifera',
    'Sponqes undetermined': 'Porifera',
    'Spongia': 'Porifera',
    'Unknown tunicate': 'Tunicata',
    'Unknown Insect sp327': 'Insecta',
    'Unknown Insect sp328': 'Insecta',
    'Paralichthinae': 'Paralichthyidae',
    'Unknown Insect sp321': 'Insecta',
    'Unknown Insect sp331': 'Insecta',
}
BROAD_TAXON_MAP = {'Annelids': 'Annelida', 'Birds': 'Aves', 'Fish': 'Actinopterygii', 'Mammals': 'Mammalia', 'Marine mammals': 'Mammalia', 'Molluscs': 'Mollusca', 'Plants': 'Tracheophyta', 'Bacteria': 'Bacteria', 'Fungi': 'Fungi', 'Tunicata': 'Tunicata'}
AVES_TAXA = {'Uria', 'Sterna', 'Columba', 'Cygnus', 'Passer', 'Anarhynchus nivosus'}
ACTINOPTERYGII_TAXA = {'Osmeriformes', 'Siluriformes', 'Perciformes', 'Tetraodontiformes', 'Scorpaeniformes', 'Myctophiformes', 'Cyprinodontiformes', 'Aulopiformes', 'Cypriniformes', 'Gadiformes', 'Lampriformes', 'Ophidiiformes', 'Characiformes', 'Gobiesociformes', 'Notacanthiformes', 'Pleuronectiformes', 'Beryciformes', 'Stomiiformes', 'Esociformes', 'Atheriniformes', 'Lophiiformes', 'Clupeiformes', 'Salmoniformes', 'Gasterosteiformes', 'Syngnathiformes', 'Anguilliformes', 'Zeiformes', 'Batrachoidiformes', 'Polymixiiformes', 'Beloniformes', 'Mugiliformes', 'Clupeiformes', 'Amiiformes', 'Cyprinidae', 'Aprion', 'Oncorhynchus', 'Galaxias', 'Scarus', 'Ammodytes', 'Teleostei', 'Percopsiformes', 'Kuhlia rupestris', 'Kuhlia', 'Prionotus', 'Leptocephali', 'Gibbonsia', 'Centropogon australis', 'Liparis', 'Chromis', 'Apogon', 'Naso', 'Acipenser oxyrinchus', 'Eurypharynx pelecanoides', 'Gonorynchiformes', 'Osteoglossiformes', 'Gymnotiformes', 'Saccopharyngiformes', 'Stephanoberyciformes', 'Elopiformes', 'Synbranchiformes', 'Cetomimiformes', 'Lepisosteiformes', 'Polypteriformes', 'Acipenseriformes', 'Pagrus', 'Brevoortia', 'Diplectrum', 'Stephanolepis'}
ELASMOBRANCHII_TAXA = {'Carcharhiniformes', 'Lamniformes', 'Orectolobiformes', 'Heterodontiformes', 'Hexanchiformes', 'Squaliformes', 'Squatiniformes', 'Pristiophoriformes', 'Rajiformes', 'Torpediniformes', 'Myliobatiformes', 'Rhinopristiformes', 'Raja'}
HOLOCEPHALI_TAXA = {'Chimaeriformes'}
CHONDRICHTHYES_TAXA = ELASMOBRANCHII_TAXA | HOLOCEPHALI_TAXA | {'Selachimorpha'}
HEMICHORDATA_TAXA = {'Enteropneusta'}
TUNICATA_TAXA = {'Ascidiacea', 'Thaliacea', 'Polycarpa', 'Appendicularia', 'Oikopleura', 'Oikopleuridae', 'Fritillaridae', 'Boltenia', 'Ascidiella', 'Copelata', 'Diplosoma'}
ECHINODERMATA_TAXA = {'Asterias', 'Atelostomata', 'Phrynophiurida', 'Henricia', 'Asterozoa', 'Asterina', 'Ophiothrix', 'Asteroidea', 'Echinoidea', 'Holothuroidea', 'Holothuria', 'Cucumaria', 'Elpidia', 'Echinus', 'Ophioderma', 'Ophiuroids unidentified', 'Pentagonaster', 'Salmacis', 'Salmacis sp312', 'Salmacis sp313', 'Stereoderma', 'Stelleroidea', 'Thyone', 'Holothuria sp2'}
MOLLUSCA_TAXA = {'Patella', 'Bivalvia', 'Crepidula', 'Rissoa', 'Clione', 'Physa', 'Teuthida', 'Diplodonta', 'Prosobranchia', 'Teuthoidea', 'Lithopoma', 'Gymnosomata', 'Abra', 'Ervilia nitens', 'Pandora', 'Lucina', 'Spiratella', 'Lyonsia', 'Patellogastropoda', 'Opisthobranchia', 'Galba', 'Gastropoda', 'Cephalopoda', 'Aplacophora', 'Rossia', 'Strombiformis', 'Anomia', 'Diceras', 'Melanochlamys', 'Stagnicola', 'Stagnicola'}
POLYCHAETA_TAXA = {'Polychaeta', 'Spirobranchus', 'Amphitrite', 'Terebellidae', 'Glycera', 'Nereis', 'Spio', 'Chone', 'Exogone', 'Euclymeninae', 'Euclymeninae', 'Eteone', 'Pectinaria', 'Pholoe', 'Harmothoinae', 'Megalomma', 'Monticellina', 'Monticellina sp', 'Aphroditacea', 'Aphroditoidea', 'Neanthes', 'Asychis', 'Fabricia', 'Polydora'}
SIPUNCULA_TAXA = {'Sipunculidea', 'Aspidosiphoniformes'}
ANNELIDA_TAXA = POLYCHAETA_TAXA | SIPUNCULA_TAXA | {'Halicryptus spinulosus', 'Hirudinea', 'Sipuncula'}
NEMERTEA_TAXA = {'Nemertinea', 'Lineidae', 'Tubulanus', 'Rhynchocoela', 'Nemertina', 'Tetrastemma sp'}
PHORONIDA_TAXA = {'Phoronis', 'Phoronida'}
PLATYHELMINTHES_TAXA = {'Turbellaria', 'Girardia'}
INSECTA_TAXA = {'Enallagma', 'Homarus', 'Conocephalus', 'Rhyacophila', 'Tethea', 'Silphini', 'Colymbetes', 'Hydropsyche', 'Elachista', 'Sialis', 'Colias', 'Athetini', 'Leptocera', 'Calopteryx', 'Gerris', 'Lestes', 'Zygoptera', 'Tanypodinae', 'Gomphus', 'Acilius', 'Agabinae', 'Aleocharinae', 'Anax', 'Capnia', 'Chironomini', 'Dolichopus undet', 'Ephemera', 'Ephemerella', 'Forcipomyiinae sp3', 'Hydrobius', 'Macronema sp', 'Oxybelus undet', 'Paederinae sp', 'Pedicidae', 'Perilampus undet', 'Plagiognathus undet', 'Potamophylax', 'Scaphisomini', 'Scolomyia', 'Scolytidae', 'Sericostoma', 'Sphaeridiinae', 'Stenonema', 'Stylurus', 'Symplecta undet', 'Tanytarsini', 'Urophora undet', 'Limnophila', 'Lepidostoma', 'Prociphilines (genus Prociphilus)'}
HEXAPODA_TAXA = INSECTA_TAXA | {'Collembola', 'Insecta'}
BRANCHIOPODA_TAXA = {'Diplostraca', 'Daphnia', 'Cladocera', 'Bosmina', 'Chydorus', 'Diaphanosoma', 'Evadne', 'Alona', 'Laevicaudata', 'Holopedium', 'Leptodora', 'Eurycercus sp', 'Sida'}
MALACOSTRACA_TAXA = {'Isopoda', 'Homarus', 'Cumella', 'Hyas', 'Dendrobranchiata', 'Sabinea septemcarinata', 'Sabinea', 'Apseudes', 'Parthenope', 'Caprella', 'Galathea', 'Callianassa', 'Parthenope longispinus', 'Hyale', 'Sergia', 'Gammarus salinus', 'Nematoscelis', 'Parthenope hoplonotus', 'Calappa', 'Hippolyte', 'Hyperia', 'Jaera', 'Leptocheirus', 'Leucothoe', 'Maera', 'Anomura', 'Atylus', 'Corophioidea', 'Dulichia', 'Eurydice', 'Neolepas', 'Oxycephalus', 'Palinurid phyllosoma', 'Pamphithoidea', 'Penaeoidea', 'Phrosina', 'Pilumnus', 'Pontogeneia sp', 'Porcellana', 'Sicyonia', 'Sphaeroma', 'Achaeus sp5993', 'Actaea tuberculosa', 'Leucosiid zoea', 'Varuna litterata'}
COPEPODA_TAXA = {'Oithona', 'Oncaea', 'Calanus', 'Cyclops', 'Poecilostomatoida', 'Clytemnestra', 'Labidocera', 'Heterorhabdus', 'Kelleria', 'Lubbockia sp', 'Mecynocera', 'Porcellidium'}
MAXILLOPODA_TAXA = COPEPODA_TAXA | {'Maxillopoda', 'Copepoda'}
THECOSTRACA_TAXA = {'Austrominius', 'Balanus', 'Cirripedia', 'Cirripedia cypris', 'Neolepas'}
OLIGOSTRACA_TAXA = {'Ostracoda', 'Sessilia', 'Cirripedia', 'Sarsiella'}
ARACHNIDA_TAXA = {'Pycnogonida', 'Acari', 'Pantopoda', 'Hydrachnidae', 'Lebertia', 'Hygrobates', 'Erigone atra', 'Monostyliferoidea', 'Phidippus undet', 'Torrenticola'}
CHELICERATA_TAXA = ARACHNIDA_TAXA | {'Merostomata', 'Arachnida'}
CNIDARIA_TAXA = {'Anthozoa', 'Hydrozoa', 'Scyphozoa', 'Sertularia', 'Obelia', 'Acropora', 'Physonect', 'Vogtia', 'Edwardsia', 'Siphonophore', 'Octocorallia', 'Ceriantharia', 'Hexacorallia', 'Hydroidolina', 'Leptoria', 'Leptoseris', 'Lucernaria', 'Alcyonacea', 'Bolocera', 'Calycophoran', 'Coryne', 'Diphasia', 'Diphyes', 'Echinopora', 'Ellisella', 'Ellisella sp2', 'Mopsella sp1', 'Mopsella sp2', 'Physonectae', 'Prayid sp', 'Primnoa', 'Sarcophyton', 'Siphonophora', 'Steroenepthya', 'Stylophora', 'Turbinaria', 'Virgularia sp1'}
BRYOZOA_TAXA = {'Gymnolaemata', 'Stenolaemata', 'Cellaria', 'Bugula', 'Alcyonidium albidum', 'Alcyonidium', 'Margaretta', 'Didymozoum', 'Hydra'}
PORIFERA_TAXA = {'Demospongiae', 'Cliona', 'Axinella', 'Leuconia', 'Clathrina sp1', 'Clathrina sp3', 'Pseudoceratina sp6', 'Raspailia sp2', 'Raspailia sp23', 'Tethya sp17', 'Tethya sp2', 'Tethya sp3', 'Tethya sp4'}
ACANTHOCEPHALA_TAXA = {'Acanthocephala'}
ROTIFERA_TAXA = {'Rotifera', 'Eurotatoria', 'Synchaeta', 'Keratella', 'Polyarthra', 'Trichocerca', 'Asplanchna', 'Kellicottia'} | ACANTHOCEPHALA_TAXA
BACILLARIOPHYCEAE_TAXA = {'Nitzschia', 'Navicula', 'Amphora', 'Cerataulina', 'Bacillariophyceae'}
TRACHEOPHYTA_TAXA = {'Plantae', 'Asteriscus', 'Fritillaria', 'Saggitaria', 'Orthosia', 'Blepharis', 'Canthidium', 'Limonia', 'Lesquerella', 'Digitaria', 'Pectis', 'Astragalus', 'Mertensia', 'Aglaia cucullata', 'Peridiaceae', 'Hibbertia', 'Acourtia', 'Byblis', 'Sagittaria', 'Viola CHE1', 'Limnophila','Phyllodoce'}
BRYOPHYTA_TAXA = {'Bryophyta', 'Verrucaria', 'Anthocerotophyta', 'Bryopsida', 'Dicranellopsis crispa', 'Dicranella crispa'}
CHLOROPHYTA_TAXA = {'Cladophora', 'Ulva', 'Mougeotia', 'Carteria', 'Schroederia', 'Neomeris', 'Ulothrix', 'Eudorina', 'Bryopsis'}
STREPTOPHYTA_TAXA = {'Charophyta'}
CHLOROPLASTIDA_TAXA = CHLOROPHYTA_TAXA | STREPTOPHYTA_TAXA | {'Prasinodermatophyta', 'Chlorophyta', 'Streptophyta'}
RHODOPHYTA_TAXA = {'Polysiphonia', 'Ceramium', 'Porphyra', 'Cryptopleura', 'Lobophora', 'Dictyota', 'Jania', 'Gracilaria', 'Amphiroa', 'Martensia', 'Liagora', 'Dasya'}
CRYPTOPHYTA_TAXA = {'Leucocryptos', 'Plagioselmis', 'Hemiselmis', 'Cryptomonadales', 'Rhodomonas', 'Teleaulax', 'Cryptophyta'}
OCHROPHYTA_TAXA = BACILLARIOPHYCEAE_TAXA | {'Amphora', 'Laminaria', 'Cerataulina', 'Meuniera', 'Actinocyclus', 'Odontella', 'Grammatophora', 'Anaulus', 'Bacillariophyceae', 'Cylindrial diatom', 'Cymbella', 'Desmarestia', 'Dictyopteris sp2', 'Dictyosiphon', 'Fallacia', 'Pennales', 'Pinnularia', 'Plagiotropis', 'Pseudosolenia', 'Raphidophytes', 'Rhabdonema', 'Solenicola setigera', 'Striatella', 'Vaucheria', 'Young laminariales', 'Rossia'}
RHIZARIA_TAXA = {'Halyphysema', 'Foraminifera', 'Rotaliida', 'Globothalamea', 'Astrorhizida', 'Radiozoa', 'Ebria tripartita', 'Arthracanthida', 'Difflugia', 'Sagenoarium', 'Trisulcus undet'}
EUGLENOZOA_TAXA = {'Euglena', 'Phacus', 'Eutreptiella', 'Trachelomonas'}
CYANOBACTERIA_TAXA = {'Anabaena', 'Trichodesmium'}
BACTERIA_TAXA = CYANOBACTERIA_TAXA | {'Bacteria', 'Cyanobacteria'}
MYRIAPODA_TAXA = {'Spirobolida', 'Scolopendromorpha', 'Lithobiomorpha', 'Julida sp'}
PANCRYPTISTA_TAXA = {'Cryptomonas'}
FUNGI_TAXA = {'Ascomycota', 'Basidiomycota'}
MAMMALIA_TAXA = {'Pipistrellus', 'Mustela', 'Sciurus'}
LISSAMPHIBIA_TAXA = {'Amphibia', 'Boreorana sylvatica'}
CHAETOGNATHA_TAXA = {'Sagitta'}
AGNATHA_TAXA = {'Petromyzonti', 'Myxini'}
CEPHALOCHORDATA_TAXA = {'Branchiostoma'}
GASTROTRICHA_TAXA = {'Gastrotricha'}
SQUAMATA_TAXA = {'Coronella austriaca'}
PRIAPULIDA_TAXA = {'Priapulus caudatus'}
ENTOPROCTA_TAXA = {'Entoprocta sp1', 'Pedicellina cernua'}
NEMATOMORPHA_TAXA = {'Nematomorpha'}
BRACHIOPODA_TAXA = {'Lingula'}
PROTOZOA_TAXA = {'Amoeba', 'Amoebozoa'}
HAPTOPHYTA_TAXA = {'Coccolithophorid sp1', 'Coccolithophorid sp2', 'Coccolithophorid sp4', 'Coccolithophorid sp5', 'Coronosphaera', 'Coronosphaera sp', 'Coccolithophorid', 'Coccolithophore'}
MYZOZOA_TAXA = {'Dinomastigota', 'Karenia', 'Thoracosphaera'}
CILIOPHORA_TAXA = {'Favella', 'Brachionus'}
CHOANOFLAGELLATEA_TAXA = {'Bicosta'}
CHLOROPLASTIDA_TAXA = CHLOROPHYTA_TAXA | STREPTOPHYTA_TAXA | {'Prasinodermatophyta', 'Chlorophyta', 'Streptophyta', 'Coccomyxa sp', 'Didymocystis', 'Sphaerocystis'}
EXCLUDED_GROUPS = {'Bacteria', 'Cyanobacteria', 'Euglenozoa', 'Myzozoa', 'Rhizaria', 'Ciliophora', 'Cryptophyta', 'Haptophyta', 'Ochrophyta', 'Chloroplastida', 'Chlorophyta', 'Rhodophyta', 'Pancryptista', 'Choanoflagellatea', 'Fungi', 'Archaea', 'Viruses', 'Virus', 'Amoebozoa', 'Protozoa', 'Chromista'}
_TAXONOMIC_CORE = ['Aves', 'Testudines', 'Crocodylia', 'Squamata', 'Sphenodontia', 'Mammalia', 'Lissamphibia', 'Sarcopterygii', 'Actinopterygii', 'Chondrichthyes', 'Agnatha', 'Cephalochordata', 'Tunicata', 'Hemichordata', 'Echinodermata', 'Chaetognatha', 'Mollusca', 'Annelida', 'Nemertea', 'Brachiopoda', 'Phoronida', 'Platyhelminthes', 'Nematoda', 'Hexapoda', 'Remipedia', 'Branchiopoda', 'Cephalocarida', 'Malacostraca', 'Maxillopoda', 'Thecostraca', 'Oligostraca', 'Myriapoda', 'Chelicerata', 'Onychophora', 'Kinorhyncha', 'Cnidaria', 'Ctenophora', 'Bryozoa', 'Porifera', 'Rotifera', 'Choanoflagellatea', 'Fungi', 'Tracheophyta', 'Bryophyta', 'Chloroplastida', 'Rhodophyta', 'Cryptophyta', 'Haptophyta', 'Ochrophyta', 'Pancryptista', 'Rhizaria', 'Myzozoa', 'Euglenozoa', 'Ciliophora', 'Bacteria', 'Gastrotricha', 'Priapulida', 'Entoprocta', 'Nematomorpha']
TAXONOMIC_ORDER = _TAXONOMIC_CORE + sorted(EXCLUDED_GROUPS - set(_TAXONOMIC_CORE)) + ['Other']

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
for group in TAXONOMIC_ORDER:
    COLLAPSE_MAPPING[group] = group

class CSVLoader:
    def __init__(self, name, columns=None, sep=',', min_time=None, time_col='month', timeseries_from='latlon', timeseries_col=None, timeseries_fixed=None, study_id_from='name', extra_cols=(), aggregate=False, null_latlon=False):
        os.makedirs('csv', exist_ok=True)
        self.name = name
        self.columns = columns or {}
        self.sep = sep
        self.min_time = min_time
        self.time_col = time_col
        self.timeseries_from = timeseries_from
        self.timeseries_col = timeseries_col
        self.timeseries_fixed = timeseries_fixed
        self.study_id_from = study_id_from
        self.extra_cols = list(extra_cols)
        self.aggregate = aggregate
        self.null_latlon = null_latlon
    def _read(self, filepath):
        for enc in ('latin-1', 'ISO-8859-1', None):
            try:
                kw = dict(sep=self.sep)
                if enc:
                    kw['encoding'] = enc
                return pd.read_csv(filepath, **kw)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Could not read {filepath}")
    def _prep(self, df):
        df.columns = df.columns.str.strip()
        lookup = {c.lower(): c for c in df.columns}
        for std in ('species', 'abundance', 'year', 'latitude', 'longitude', 'month', 'day', 'quarter', 'study_id', 'taxonomic_group'):
            actual = self.columns.get(std, std)
            if actual.lower() in lookup:
                df = df.rename(columns={lookup[actual.lower()]: std})
        required = ['species', 'abundance', 'year']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.name}: {missing}")
        df = df.dropna(subset=required)
        df['ABUNDANCE'] = pd.to_numeric(df['abundance'], errors='coerce')
        df['YEAR'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['ABUNDANCE', 'YEAR'])
        df['SPECIES_ID'] = df['species'].astype(str).str.strip().str.replace('_', ' ')
        time_val = df['YEAR'].copy()
        if self.time_col == 'month' and 'month' in df.columns:
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
            mm = df['month'].notna()
            time_val.loc[mm] = df.loc[mm, 'YEAR'] + (df.loc[mm, 'month'].astype(int) - 1) / 12.0
            if 'day' in df.columns:
                df['day'] = pd.to_numeric(df['day'], errors='coerce')
                md = df['day'].notna()
                m = mm & md
                time_val.loc[m] += (df.loc[m, 'day'].astype(int) - 1) / 365.0
        elif self.time_col == 'quarter' and 'quarter' in df.columns:
            time_val += (df['quarter'].apply(parse_quarter).fillna(2.5) - 1) / 4.0
        df['TIME'] = time_val
        if self.min_time is not None:
            df = df[df['TIME'] >= self.min_time]
        df['STUDY_ID'] = df['study_id'].astype(str) if self.study_id_from == 'data' and 'study_id' in df.columns else self.name
        df['DATASET'] = self.name
        df['LATITUDE'] = df['latitude'] if not self.null_latlon and 'latitude' in df.columns else None
        df['LONGITUDE'] = df['longitude'] if not self.null_latlon and 'longitude' in df.columns else None
        if self.timeseries_from == 'study_id':
            df['TIMESERIES_ID'] = df['STUDY_ID'].astype(str)
        elif self.timeseries_from == 'column' and self.timeseries_col.lower() in {c.lower() for c in df.columns}:
            actual = next(c for c in df.columns if c.lower() == self.timeseries_col.lower())
            df['TIMESERIES_ID'] = df[actual].astype(str)
        elif self.timeseries_from == 'fixed':
            df['TIMESERIES_ID'] = self.timeseries_fixed or self.name
        elif self.null_latlon or 'latitude' not in df.columns or 'longitude' not in df.columns:
            df['TIMESERIES_ID'] = self.name
        else:
            lat = df['latitude'].fillna(0).astype(float).round(2)
            lon = df['longitude'].fillna(0).astype(float).round(2)
            df['TIMESERIES_ID'] = self.name + '_' + lat.astype(str) + '_' + lon.astype(str)
        out_cols = ["SPECIES_ID", "TIME", "ABUNDANCE", "STUDY_ID", "LATITUDE", "LONGITUDE", "DATASET", "TIMESERIES_ID"]
        for c in self.extra_cols:
            actual = next((a for a in df.columns if a.lower() == c.lower()), None)
            df[c] = df[actual].astype(str).str.strip() if actual else None
            out_cols.append(c)
        if self.aggregate:
            df = df.groupby(['SPECIES_ID', 'TIMESERIES_ID', 'TIME'], as_index=False).agg({'ABUNDANCE': 'sum'})
            df['STUDY_ID'] = self.name
            df['DATASET'] = self.name
            df['LATITUDE'] = None
            df['LONGITUDE'] = None
        return df[out_cols]
    def load(self, filepath):
        return self._prep(self._read(filepath))
    def load_cedar_creek_data(self, e120_plant_path, e120_arthropod_path, e244_path):
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
                e120_plant_df["SPECIES"] = e120_plant_df["SPECIES"].replace({"Miscellaneous*": "Other", "Unsorted*": "Other", "Green matter*": "Other", "Achillea millefolium(lanulosa)": "Achillea millefolium", "Miscellaneous litter": "Other", "Miscellaneous Litter": "Other"})
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
                e244_df["SPECIES"] = e244_df["SPECIES"].replace({"Achillea millefolium(lanulosa)": "Achillea millefolium", "Miscellaneous litter": "Other", "Miscellaneous Litter": "Other"})
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
def csv_entry(name, path, taxonomy='taxonomic_group', require_exists=True, **overrides):
    return dict(paths=[path], loader=CSVLoader(name, **overrides).load, args=(path,), kwargs={}, require_exists=require_exists, taxonomy=taxonomy)
DATASETS = {
'biotime': csv_entry('BioTIME', biotime_path, taxonomy=None, require_exists=False, columns={'species': 'VALID_NAME'}, min_time=0, timeseries_from='study_id', study_id_from='data'),
'rivfishtime': csv_entry('RivFishTIME', rivfishtime_path, taxonomy=None, time_col='quarter', timeseries_from='column', timeseries_col='TIMESERIESID', timeseries_fixed='RivFishTIME', aggregate=True, null_latlon=True),
'cedar_creek': dict(paths=[cedar_creek_paths['e120_plant'], cedar_creek_paths['e120_arthropod'], cedar_creek_paths['e244']], loader=CSVLoader('CedarCreek').load_cedar_creek_data, args=(cedar_creek_paths['e120_plant'], cedar_creek_paths['e120_arthropod'], cedar_creek_paths['e244']), kwargs={}, require_exists=False, taxonomy=None),
'gbif': csv_entry('GBIF', gbif_path, taxonomy='rank_cols', sep='\t', timeseries_from='fixed', timeseries_fixed='GBIF', extra_cols=['kingdom', 'phylum', 'class', 'order', 'family', 'genus'], null_latlon=True),
'gpdd': csv_entry('GPDD', gpdd_path, taxonomy='rank_cols', extra_cols=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']),
'nps': csv_entry('NPS', nps_path, extra_cols=['taxonomic_group']),
'lpd': csv_entry('LPD', lpd_path, extra_cols=['taxonomic_group']),
'glba': csv_entry('GLBA', glba_path, extra_cols=['taxonomic_group']),
'vrdata': csv_entry('VRData', vrdata_path, extra_cols=['taxonomic_group']),
'tsx': csv_entry('TSX', tsx_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
'parrot': csv_entry('Parrot', parrot_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
'lizard': csv_entry('Lizard', lizard_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
'primate': csv_entry('Primate', primate_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
'frog': csv_entry('Frog', frog_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
'akepa': csv_entry('Akepa', akepa_path, timeseries_from='column', timeseries_col='timeseries_id', extra_cols=['taxonomic_group']),
}

def find_column(df, synonyms):
    for col in df.columns:
        col_clean = col.strip().lower()
        for syn in synonyms:
            if col_clean == syn.strip().lower():
                return col
    return None

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

def _dataset_cache_key(name, args, kwargs):
    key_str = name + '|' + repr(args) + '|' + repr(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:8]

def _build_taxonomy_from_requested(requested, dataset_cache_dir, force_reload, min_time):
    tax_dfs = []
    for name, spec in DATASETS.items():
        if spec['taxonomy'] == 'rank_cols' and name in requested:
            df = _load_one_dataset(name, dataset_cache_dir, force_reload, min_time)
            if df is not None and len(df) > 0:
                tax_dfs.append(df)
    if tax_dfs:
        combined = pd.concat(tax_dfs, ignore_index=True)
        data_taxonomy = build_data_taxonomy_from_dataframe(combined)
    else:
        data_taxonomy = {}
    for name, spec in DATASETS.items():
        if spec['taxonomy'] == 'taxonomic_group' and name in requested:
            df = _load_one_dataset(name, dataset_cache_dir, force_reload, min_time)
            if df is not None and len(df) > 0 and 'taxonomic_group' in df.columns:
                for _, row in df[['SPECIES_ID', 'taxonomic_group']].drop_duplicates().iterrows():
                    sp = row['SPECIES_ID']
                    tg = row['taxonomic_group']
                    if pd.notna(tg) and str(tg).strip() != '':
                        data_taxonomy[sp] = collapse_taxonomic_group(str(tg).strip())
    return data_taxonomy

#def _requested_dataset_ids():
#    ids = set()
#    for name, spec in DATASETS.items():
#        if not spec['require_exists'] or all(p and os.path.exists(p) for p in spec['paths']):
#            ids.add(name)
#    return ids

def _requested_dataset_ids():
    ids = set()
    cache_dir = os.path.join('csv', 'datasets')
    cache_files = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
    for name, spec in DATASETS.items():
        if not spec['require_exists'] or all(p and os.path.exists(p) for p in spec['paths']):
            ids.add(name)
        elif any(f.startswith(f'{name}_') and f.endswith('.pkl') for f in cache_files):
            ids.add(name)
    return ids

def _load_dataset_with_cache(name, loader_fn, cache_dir, force_reload, min_time_filter, *args, **kwargs):
    os.makedirs(cache_dir, exist_ok=True)
    kwargs_for_key = dict(kwargs)
    kwargs_for_key.pop('min_time', None)
    key = _dataset_cache_key(name, args, kwargs_for_key)
    cache_path = os.path.join(cache_dir, f'{name}_{key}.pkl')
    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading cached {name} dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
    else:
        print(f"Loading {name} dataset from raw files...")
        df = loader_fn(*args, **kwargs)
        if df is not None and len(df) > 0:
            if 'DATASET' in df.columns and 'dataset' not in df.columns:
                df = df.rename(columns={'DATASET': 'dataset'})
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
    if df is not None and len(df) > 0:
        df = df[df['TIME'] >= min_time_filter].copy()
    return df

def _load_one_dataset(name, dataset_cache_dir, force_reload, min_time):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    spec = DATASETS[name]
    return _load_dataset_with_cache(name, spec['loader'], dataset_cache_dir, force_reload, min_time, *spec['args'], **spec['kwargs'])

def _read_species_post1900_manifest(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        raise FileNotFoundError(f"species_post1900 manifest not found or empty: {filepath}")
    df = pd.read_csv(filepath)
    required = ['species', 'dataset', 'first_year', 'last_year']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"species_post1900.csv missing required columns: {missing}")
    has_ts = 'timeseries_id' in df.columns
    manifest = {}
    for _, row in df.iterrows():
        if has_ts:
            key = (str(row['species']).strip(), str(row['dataset']).strip(), str(row['timeseries_id']).strip())
        else:
            key = (str(row['species']).strip(), str(row['dataset']).strip())
        manifest[key] = (int(row['first_year']), int(row['last_year']))
    return manifest

def _apply_species_post1900_manifest(df, manifest):
    if df is None or len(df) == 0:
        return df
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest must be a dict, got {type(manifest)}")
    if not manifest:
        return df
    sample_key = next(iter(manifest))
    use_ts = len(sample_key) == 3
    dataset_col = 'DATASET' if 'DATASET' in df.columns else 'dataset'
    rows = []
    for k, v in manifest.items():
        row = {'_species_key': k[0], '_dataset_key': k[1], '_first_year': v[0], '_last_year': v[1]}
        if use_ts:
            row['_timeseries_key'] = k[2]
        rows.append(row)
    manifest_df = pd.DataFrame(rows)
    manifest_df['_species_key'] = manifest_df['_species_key'].astype(str).str.strip()
    manifest_df['_dataset_key'] = manifest_df['_dataset_key'].astype(str).str.strip()
    df = df.copy()
    df['_species_key'] = df['SPECIES_ID'].astype(str).str.strip()
    df['_dataset_key'] = df[dataset_col].astype(str).str.strip()
    merge_keys = ['_species_key', '_dataset_key']
    if use_ts:
        df['_timeseries_key'] = df['TIMESERIES_ID'].astype(str).str.strip()
        manifest_df['_timeseries_key'] = manifest_df['_timeseries_key'].astype(str).str.strip()
        merge_keys.append('_timeseries_key')
    df = df.merge(manifest_df, on=merge_keys, how='inner')
    df = df[(df['TIME'] >= df['_first_year']) & (df['TIME'] < df['_last_year'] + 1)]
    drop_cols = ['_species_key', '_dataset_key', '_first_year', '_last_year']
    if use_ts:
        drop_cols.append('_timeseries_key')
    return df.drop(columns=drop_cols)

def split_into_maximal_valid_segments(series_df, filter_func, year_to_points):
    if len(series_df) == 0:
        return []
    series_df = series_df.sort_values('YEAR').reset_index(drop=True)
    years = series_df['YEAR'].values
    boundaries = [0] + [int(i) + 1 for i in np.where(np.diff(years) > 1)[0]] + [len(years)]
    all_segments = []
    for b in range(len(boundaries) - 1):
        start = int(boundaries[b])
        end = int(boundaries[b + 1])
        block = series_df.iloc[start:end].reset_index(drop=True)
        if len(block) == 0:
            continue
        segs = _optimal_partition(block, filter_func, year_to_points)
        all_segments.extend(segs)
    return all_segments

def _score_better(a, b):
    a_years, a_min, a_points, a_num = a
    b_years, b_min, b_points, b_num = b
    if a_years != b_years:
        return a_years > b_years
    if a_min != b_min:
        return a_min > b_min
    if a_points != b_points:
        return a_points > b_points
    return a_num < b_num

def _optimal_partition(block, filter_func, year_to_points):
    n = len(block)
    if n == 0:
        return []
    MAX_SEG_LEN = 200
    arr = block['ABUNDANCE'].values.astype(float)
    years = block['YEAR'].values.astype(int)
    S = np.concatenate([[0.0], np.cumsum(arr)])
    S2 = np.concatenate([[0.0], np.cumsum(arr * arr)])
    changes = np.zeros(n + 1, dtype=np.int64)
    for k in range(1, n):
        changes[k + 1] = changes[k] + (1 if arr[k] != arr[k - 1] else 0)
    min_4u = np.full(n, n, dtype=np.int64)
    for i in range(n):
        seen = set()
        for j in range(i, n):
            seen.add(arr[j])
            if len(seen) >= 4:
                min_4u[i] = j
                break
    max_len = min(MAX_SEG_LEN, n)
    valid = np.zeros((n, max_len), dtype=bool)
    for i in range(n):
        upper = min(max_len, n - i)
        for k in range(upper):
            j = i + k
            length = j - i + 1
            if length < 4:
                continue
            if j < min_4u[i]:
                continue
            cnt = length
            total = S[j + 1] - S[i]
            total_sq = S2[j + 1] - S2[i]
            mean = total / cnt
            var = total_sq / cnt - mean * mean
            if var <= 0.0:
                continue
            if i == 0:
                num_ch = int(changes[j + 1])
            else:
                num_ch = int(changes[j + 1] - changes[i + 1])
            if num_ch < 3:
                continue
            valid[i, k] = True
    INF = float('inf')
    f_score = [None] * (n + 1)
    f_choice = [None] * (n + 1)
    f_score[n] = (0, INF, 0, 0)
    f_choice[n] = []
    for i in range(n - 1, -1, -1):
        best_score = f_score[i + 1]
        best_choice = f_choice[i + 1]
        max_k = min(max_len, n - i)
        for k in range(max_k):
            if not valid[i, k]:
                continue
            j = i + k
            seg_len = j - i + 1
            seg_years_set = set(years[i:j + 1].tolist())
            seg_points = sum(year_to_points.get(int(y), 0) for y in seg_years_set)
            rest_score = f_score[j + 1]
            rest_total_years, rest_min, rest_points, rest_num = rest_score
            if rest_num == 0:
                new_min = seg_len
            else:
                new_min = min(seg_len, rest_min)
            new_score = (seg_len + rest_total_years, new_min, seg_points + rest_points, rest_num + 1)
            if _score_better(new_score, best_score):
                best_score = new_score
                best_choice = [(i, j)] + f_choice[j + 1]
        f_score[i] = best_score
        f_choice[i] = best_choice
    result = []
    for (i, j) in f_choice[0]:
        seg = block.iloc[i:j + 1].reset_index(drop=True)
        passed, _ = filter_func(seg[['YEAR', 'ABUNDANCE']].copy())
        if passed:
            result.append(seg)
    return result

def _segment_one_species(species_name, sp_df, filter_func, data_taxonomy, genus_taxonomy):
    if str(species_name).strip() in EXCLUDED_SPECIES_LABELS:
        return None
    tax_group = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
    if tax_group in EXCLUDED_GROUPS:
        return None
    parts = []
    for ts_id, ts_df in sp_df.groupby('TIMESERIES_ID'):
        ts_df = ts_df.copy()
        ts_df['_YEAR'] = np.floor(ts_df['TIME']).astype(int)
        annual = ts_df.groupby('_YEAR')['ABUNDANCE'].mean().reset_index()
        annual.columns = ['YEAR', 'ABUNDANCE']
        passed_whole, _ = filter_func(annual[['YEAR', 'ABUNDANCE']].copy())
        if passed_whole:
            keep = ts_df.drop(columns=['_YEAR'])
            keep['TIMESERIES_ID'] = str(ts_id)
            parts.append(keep)
            continue
        year_to_points_raw = ts_df.groupby('_YEAR').size().to_dict()
        year_to_points = {int(k): int(v) for k, v in year_to_points_raw.items()}
        segments = split_into_maximal_valid_segments(annual, filter_func, year_to_points)
        for seg_idx, seg in enumerate(segments):
            seg_years = set(seg['YEAR'].values)
            seg_data = ts_df[ts_df['_YEAR'].isin(seg_years)].copy()
            seg_data['TIMESERIES_ID'] = f"{ts_id}__seg{seg_idx + 1:02d}"
            seg_data = seg_data.drop(columns=['_YEAR'])
            parts.append(seg_data)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)

def _filter_dataset_segments(df, filter_func, data_taxonomy, genus_taxonomy):
    if 'dataset' not in df.columns:
        if 'DATASET' in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
    results = Parallel(n_jobs=-1, backend='threading')(delayed(_segment_one_species)(sp, sp_df, filter_func, data_taxonomy, genus_taxonomy) for sp, sp_df in tqdm(df.groupby('SPECIES_ID'), desc="Segmenting species"))
    kept = [r for r in results if r is not None and len(r) > 0]
    if not kept:
        return df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)

def load_combined_data(min_records=20, force_reload=False, prefilter=True, filter_func=None, min_time=1900, whitelist=None):
    if prefilter and filter_func is None:
        raise ValueError("filter_func must be provided when prefilter=True")
    cache_dir = 'csv'
    os.makedirs(cache_dir, exist_ok=True)
    dataset_cache_dir = os.path.join(cache_dir, 'datasets')
    data_taxonomy_path = os.path.join(cache_dir, 'data_taxonomy.pkl')
    genus_taxonomy_path = os.path.join(cache_dir, 'genus_taxonomy.pkl')
    filtered_datasets_dir = os.path.join(cache_dir, 'filtered_datasets')
    os.makedirs(filtered_datasets_dir, exist_ok=True)
    whitelist_hash_file = os.path.join(cache_dir, 'whitelist_hash.txt')
    if whitelist is None:
        post1900_file = os.path.join(cache_dir, 'species_post1900.csv')
        whitelist = _read_species_post1900_manifest(post1900_file)
        print(f"Loaded species_post1900 manifest with {len(whitelist)} (species, dataset) pairs from {post1900_file}")
    current_hash = hashlib.md5(str(sorted(whitelist.items())).encode() if whitelist else b'empty').hexdigest()
    cached_hash = None
    if os.path.exists(whitelist_hash_file):
        with open(whitelist_hash_file, 'r') as f:
            cached_hash = f.read().strip()
    if cached_hash != current_hash:
        print(f"Whitelist hash changed ({cached_hash} -> {current_hash}); invalidating filtered dataset cache.")
        for fname in os.listdir(filtered_datasets_dir):
            if fname.endswith('.pkl'):
                os.remove(os.path.join(filtered_datasets_dir, fname))
    requested = _requested_dataset_ids()
    if not force_reload and os.path.exists(data_taxonomy_path) and os.path.exists(genus_taxonomy_path):
        with open(data_taxonomy_path, 'rb') as f:
            data_taxonomy = pickle.load(f)
        with open(genus_taxonomy_path, 'rb') as f:
            genus_taxonomy = pickle.load(f)
    else:
        data_taxonomy = _build_taxonomy_from_requested(requested, dataset_cache_dir, force_reload, min_time)
        genus_taxonomy = build_genus_taxonomy(data_taxonomy)
        with open(data_taxonomy_path, 'wb') as f:
            pickle.dump(data_taxonomy, f)
        with open(genus_taxonomy_path, 'wb') as f:
            pickle.dump(genus_taxonomy, f)
    print(f"Built data_taxonomy for {len(data_taxonomy)} species, genus_taxonomy for {len(genus_taxonomy)} genera.")
    datasets_dict = {}
    for name in sorted(requested):
        if str(name).lower() == 'gbif':
            continue
        out_path = os.path.join(filtered_datasets_dir, f'{name}.pkl')
        if not force_reload and os.path.exists(out_path):
            print(f"Loading cached filtered {name} from {out_path}")
            with open(out_path, 'rb') as f:
                df = pickle.load(f)
        else:
            print(f"Loading and filtering {name} dataset...")
            df = _load_one_dataset(name, dataset_cache_dir, force_reload, min_time)
            if df is None or len(df) == 0:
                df = pd.DataFrame()
            else:
                df = _apply_species_post1900_manifest(df, whitelist)
                if len(df) > 0 and prefilter:
                    df = _filter_dataset_segments(df, filter_func, data_taxonomy, genus_taxonomy)
            with open(out_path, 'wb') as f:
                pickle.dump(df, f)
        if len(df) > 0:
            if 'DATASET' in df.columns and 'dataset' not in df.columns:
                df = df.rename(columns={'DATASET': 'dataset'})
            if 'STUDY_ID' in df.columns and 'study_id' not in df.columns:
                df = df.rename(columns={'STUDY_ID': 'study_id'})
            datasets_dict[name] = df
    if prefilter:
        for name in list(datasets_dict.keys()):
            df = datasets_dict[name]
            if 'dataset' not in df.columns:
                df = df.copy()
                df['dataset'] = name
            counts = df.groupby(['SPECIES_ID', 'dataset', 'TIMESERIES_ID']).size().reset_index(name='count')
            keep = counts[counts['count'] >= min_records][['SPECIES_ID', 'dataset', 'TIMESERIES_ID']]
            filtered = df.merge(keep, on=['SPECIES_ID', 'dataset', 'TIMESERIES_ID'], how='inner')
            if len(filtered) == 0:
                del datasets_dict[name]
            else:
                datasets_dict[name] = filtered
    all_species = set()
    for name, df in datasets_dict.items():
        all_species.update(df['SPECIES_ID'].unique())
    species_list = sorted(all_species)
    total = sum(len(df) for df in datasets_dict.values())
    print(f"Final: {len(datasets_dict)} datasets, {len(species_list)} species, {total} records after min_records={min_records}.")
    with open(whitelist_hash_file, 'w') as f:
        f.write(current_hash)
    return datasets_dict, species_list, data_taxonomy, genus_taxonomy

def collapse_taxonomic_group(group):
    if group in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[group]
    return COLLAPSE_MAPPING.get(group, 'Other')

def get_cached_taxonomic_group(species_name, data_taxonomy=None, force_recompute=False, genus_taxonomy=None):
    cache_file = 'csv/taxonomic_cache.csv'
    os.makedirs('csv', exist_ok=True)
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return 'Other'
    species_str = str(species_name).strip()
    if species_str in EXCLUDED_SPECIES_LABELS:
        return 'Other'
    if species_str.upper() in BANDING_CODE_TO_NAME:
        return 'Aves'
    if data_taxonomy is not None:
        if species_str in data_taxonomy:
            group = data_taxonomy[species_str]
            if group != 'Other':
                return group
    if re.search(r'\b(miscellaneous|litter|unknown)\b', species_str.lower()):
        return 'Other'
    with cache_lock:
        cache_df = None
        if not force_recompute and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            try:
                cache_df = pd.read_csv(cache_file)
                if 'species' not in cache_df.columns or 'taxonomic_group' not in cache_df.columns:
                    cache_df = None
                elif not cache_df.empty:
                    raw_len = len(cache_df)
                    cache_df['species'] = cache_df['species'].astype(str).str.strip()
                    cache_df = cache_df.drop_duplicates(subset=['species'], keep='last')
                    if len(cache_df) != raw_len:
                        cache_df.to_csv(cache_file, index=False)
                    cached = cache_df[cache_df['species'] == species_str]
                    if not cached.empty:
                        cached_group = cached.iloc[0]['taxonomic_group']
                        if cached_group != 'Other':
                            return cached_group
            except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                try:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                except OSError:
                    pass
                cache_df = None
        if cache_df is None or cache_df.empty or 'species' not in cache_df.columns or 'taxonomic_group' not in cache_df.columns:
            cache_df = pd.DataFrame(columns=['species', 'taxonomic_group'])
        if 'sp' in species_str.lower() or 'cf.' in species_str.lower() or 'aff.' in species_str.lower():
            parts = species_str.split()
            if len(parts) > 0:
                genus = parts[0]
                if genus_taxonomy is not None and genus in genus_taxonomy:
                    taxonomic_group = genus_taxonomy[genus]
                    if taxonomic_group != 'Other':
                        cache_df = cache_df[cache_df['species'] != species_str]
                        new_row = pd.DataFrame({'species': [species_str], 'taxonomic_group': [taxonomic_group]})
                        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        cache_df = cache_df.drop_duplicates(subset=['species'], keep='last')
                        cache_df.to_csv(cache_file, index=False)
                        return taxonomic_group
                tax_info = get_taxonomic_info(genus)
                if tax_info:
                    taxonomic_group = assign_taxonomic_group(tax_info, genus)
                    if taxonomic_group != 'Other':
                        cache_df = cache_df[cache_df['species'] != species_str]
                        new_row = pd.DataFrame({'species': [species_str], 'taxonomic_group': [taxonomic_group]})
                        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        cache_df = cache_df.drop_duplicates(subset=['species'], keep='last')
                        cache_df.to_csv(cache_file, index=False)
                        return taxonomic_group
        taxonomic_group = classify_taxonomic_group(species_str)
        cache_df = cache_df[cache_df['species'] != species_str]
        new_row = pd.DataFrame({'species': [species_str], 'taxonomic_group': [taxonomic_group]})
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=['species'], keep='last')
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
    if re.search(r'\b(miscellaneous|litter|unknown)\b', name.lower()):
        return 'Other'
    genus = name.split()[0] if name.split() else name
    if genus in SYNONYM_MAPPING:
        return SYNONYM_MAPPING[genus] + ' ' + ' '.join(name.split()[1:])
    return name

def clean_taxonomic_groups(name):
    if pd.isnull(name) or not isinstance(name, str) or name.strip() == '':
        return 'Other'
    cleaned = name.strip()
    if re.search(r'\b(miscellaneous|litter|unknown)\b', cleaned.lower()):
        return 'Other'
    cleaned = name.replace('_', ' ').strip()
    cleaned = re.sub(r'\s*sp(\.|\d+).*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'[\d_]+$', '', cleaned).strip()
    if cleaned == '':
        return 'Other'
    return cleaned

_QUALIFIER_PATTERNS = [r'\baff\.', r'\bcf\.', r'\bsp\.', r'\bspp\.', r'\bnear\b', r'\bs\.l\.', r'\bs\.str\.', r'\bsensu lato\b', r'\bsensu stricto\b', r'\bcomplex\b']
_RANK_MARKERS = ['subg.', 'subgen.', 'sect.', 'ser.', 'var.', 'varietas', 'subsp.', 'ssp.', 'nothosubsp.', 'f.', 'fo.', 'forma']
_RANK_TOKENS = {'subg', 'subgen', 'sect', 'ser', 'var', 'varietas', 'subsp', 'ssp', 'nothosubsp', 'f', 'fo', 'forma', 'aff', 'cf', 'sp', 'spp', 'nr', 'near'}
_GENUS_SYNONYMS = {'Lophogorgia': 'Leptogorgia'}

def _strip_junk(s):
    return s.strip(' .,;:!?')

def _normalize_case(name):
    if not name:
        return name
    words = name.split()
    if len(words) == 1:
        return words[0].capitalize()
    out = [words[0].capitalize()]
    for w in words[1:]:
        if w.lower().rstrip('.') in _RANK_TOKENS:
            out.append(w.lower())
        elif len(w) <= 2 and re.match(r'^[A-Za-z]+$', w):
            out.append(w.lower())
        elif '-' in w:
            parts = w.split('-')
            out.append('-'.join([parts[0].capitalize()] + [p.lower() for p in parts[1:]]))
        else:
            out.append(w.lower())
    return ' '.join(out)

def _valid(c):
    if not c or not re.match(r'^[A-Z]', c):
        return False
    toks = c.split()
    if not toks:
        return False
    if toks[-1].lower().rstrip('.') in _RANK_TOKENS:
        return False
    return True

def _expand_one(name):
    out = []
    if not name:
        return out
    if '(' in name and ')' in name:
        inside_m = re.search(r'\(([^)]+)\)', name)
        if inside_m:
            inside = inside_m.group(1).strip()
            without = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
            without = re.sub(r'\s+', ' ', without).strip()
            if without and without != name:
                out.append(without)
            parts = re.split(r'\s*\([^)]*\)\s*', name)
            if len(parts) == 2:
                before, after = parts[0].strip(), parts[1].strip()
                if before and after:
                    promoted = f"{inside} {after}".strip()
                    if promoted != name:
                        out.append(promoted)
    for rw in _RANK_MARKERS:
        m = re.search(r'\s+' + re.escape(rw) + r'\s+\S+\s*$', name, flags=re.I)
        if m:
            t = name[:m.start()].strip()
            if t and t != name:
                out.append(t)
    for rw in _RANK_MARKERS:
        m = re.search(r'^(.*?)\s+' + re.escape(rw) + r'\s+(\S+)\s+(\S+)\s*$', name, flags=re.I)
        if m:
            head, tail = m.group(1).strip(), m.group(3).strip()
            merged = f"{head} {tail}".strip()
            if merged and merged != name:
                out.append(merged)
        m2 = re.search(r'^(.*?)\s+' + re.escape(rw) + r'\s+(\S+)\s*$', name, flags=re.I)
        if m2:
            head, tail = m2.group(1).strip(), m2.group(2).strip()
            merged = f"{head} {tail}".strip()
            if merged and merged != name:
                out.append(merged)
    words = name.split()
    if len(words) >= 3:
        second = words[1].lower().rstrip('.')
        if second not in _RANK_TOKENS and not second.startswith('('):
            binom = ' '.join(words[:2])
            if binom != name:
                out.append(binom)
        third = words[2].lower().rstrip('.')
        if third not in _RANK_TOKENS:
            first_third = f"{words[0]} {words[2]}"
            if first_third != name:
                out.append(first_third)
    if re.search(r'\b\d+-', name):
        out.append(re.sub(r'(\b\d+)-', r'\1', name))
        out.append(re.sub(r'\b\d+-', '', name))
    return out

def generate_name_variants(name):
    if pd.isnull(name) or not isinstance(name, str):
        return []
    s = re.sub(r'\s+', ' ', name.strip())
    s = re.sub(r'\s+and\s+(allied\s+genera|allies|other\s+genera)\b.*', '', s, flags=re.I)
    s = _strip_junk(s)
    if not s:
        return []
    chunks = re.split(r'\s+and\s+|\s*;\s*|\s*&\s*', s, flags=re.I)
    pieces = []
    for ch in chunks:
        ch = _strip_junk(ch)
        if ch:
            pieces.append(ch)
    for p in list(pieces):
        words = p.split()
        if words and words[0] in _GENUS_SYNONYMS:
            pieces.append(_GENUS_SYNONYMS[words[0]] + p[len(words[0]):])
    raw = []
    for piece in pieces:
        raw.append(piece)
        normalized = _normalize_case(piece)
        if normalized and normalized != piece:
            raw.append(normalized)
            raw.extend(_expand_one(normalized))
        cleaned = piece
        for q in _QUALIFIER_PATTERNS:
            cleaned = re.sub(q, '', cleaned, flags=re.I)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = _strip_junk(cleaned)
        cleaned = _normalize_case(cleaned)
        if re.search(r'\b(aff|cf)\.', piece, flags=re.I) and cleaned:
            g = cleaned.split()[0]
            if g:
                raw.append(g)
        if cleaned:
            raw.append(cleaned)
            raw.extend(_expand_one(cleaned))
        if cleaned != piece:
            raw.extend(_expand_one(piece))
        cleaned_extra = clean_taxonomic_groups(piece)
        if cleaned_extra and cleaned_extra != piece:
            cleaned_extra = _normalize_case(cleaned_extra)
            raw.append(cleaned_extra)
            raw.extend(_expand_one(cleaned_extra))
        mapped = apply_synonym_mapping(cleaned_extra or piece)
        if mapped and mapped != piece and mapped != cleaned_extra:
            mapped = _normalize_case(mapped)
            raw.append(mapped)
            raw.extend(_expand_one(mapped))
        no_trailing = re.sub(r'[\d_]+$', '', piece).strip()
        if no_trailing and no_trailing != piece:
            no_trailing = _normalize_case(no_trailing)
            raw.append(no_trailing)
            raw.extend(_expand_one(no_trailing))
    seen, out = set(), []
    for c in raw:
        c = _strip_junk(re.sub(r'\s+', ' ', c).strip())
        if not _valid(c) or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out

@memory.cache
def get_taxonomic_info(scientific_name, depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return None
    base_name = apply_synonym_mapping(scientific_name)
    clean_name = clean_taxonomic_groups(base_name)
    if not clean_name or not clean_name.strip():
        return None
    variants = generate_name_variants(clean_name)
    if not variants:
        return None
    for variant in variants:
        clean_words = variant.split()
        search_strategies = []
        if len(clean_words) > 1:
            search_strategies.append({'name': variant, 'rank': 'SPECIES'})
        search_strategies.append({'name': variant})
        if len(clean_words) > 0:
            genus = clean_words[0]
            search_strategies.append({'name': genus, 'rank': 'GENUS'})
            search_strategies.append({'name': genus, 'rank': 'FAMILY'})
            search_strategies.append({'name': genus, 'rank': 'ORDER'})
            search_strategies.append({'name': genus, 'rank': 'CLASS'})
            search_strategies.append({'name': genus, 'rank': 'PHYLUM'})
        search_strategies.append({'name': variant, 'rank': None})
        for params in search_strategies:
            for attempt in range(3):
                try:
                    resp = requests.get("https://api.gbif.org/v1/species/match", params={'name': params['name'], 'rank': params.get('rank')}, timeout=30)
                    data = resp.json()
                    if data.get('matchType') in ['EXACT', 'FUZZY', 'HIGHERRANK'] and data.get('confidence', 0) >= 80:
                        result = {'order': data.get('order'), 'class': data.get('class'), 'subclass': data.get('subclass'), 'subphylum': data.get('subphylum'), 'phylum': data.get('phylum'), 'kingdom': data.get('kingdom')}
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
    if len(clean_name.split()) > 1:
        return get_taxonomic_info(clean_name.split()[0], depth+1)
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

def load_banding_codes(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {}
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    required = ['code4', 'code6', 'scientific_name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")
    lookup = {}
    for _, row in df.iterrows():
        sci = str(row['scientific_name']).strip()
        if not sci:
            continue
        c4 = str(row['code4']).strip().upper() if pd.notna(row['code4']) else ''
        c6 = str(row['code6']).strip().upper() if pd.notna(row['code6']) else ''
        if c4 and len(c4) == 4:
            lookup[c4] = sci
        if c6 and len(c6) == 6:
            lookup[c6] = sci
    return lookup

BANDING_CODE_TO_NAME = load_banding_codes(banding_codes_path)

def classify_taxonomic_group(species_name):
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return 'Other'
    species_str = str(species_name).strip()
    if species_str in EXCLUDED_SPECIES_LABELS:
        return 'Other'
    if species_str.upper() in BANDING_CODE_TO_NAME:
        return 'Aves'
    if re.search(r'\b(miscellaneous|litter|unknown)\b', species_str.lower()):
        return 'Other'
    clean_name = clean_taxonomic_groups(species_name)
    if not clean_name or clean_name == 'Other':
        return 'Other'
    mapped_name = apply_synonym_mapping(clean_name)
    if mapped_name in TAXONOMIC_ORDER:
        return mapped_name
    tax_info = get_taxonomic_info(clean_name)
    if not tax_info and mapped_name != clean_name:
        tax_info = get_taxonomic_info(mapped_name)
    taxonomic_group = assign_taxonomic_group(tax_info, mapped_name)
    if taxonomic_group != 'Other':
        return taxonomic_group
    if species_str in COLLAPSE_MAPPING:
        return COLLAPSE_MAPPING[species_str]
    return 'Other'

def set_plot_style():
    plt.rcParams.update({'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 18, 'axes.titlesize': 20, 'font.size': 20, 'lines.linewidth': 1.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 1.5})

def abbreviate_bifurcation_type(name):
    if name == 'fold bifurcation':
        return 'fold bif.'
    elif name == 'transcritical bifurcation':
        return 'transc. bif.'
    elif name == 'Hopf bifurcation':
        return 'Hopf bif.'
    elif name == 'no-bifurcation':
        return 'no-bif.'
    elif name == 'undetermined':
        return 'undet.'
    else:
        return name

def _series_qualifies(points, species):
    if len(points) < 20:
        print(f"Short series detected for {species}, skipping")
        return False
    if np.std(points) < 1e-10:
        print(f"Constant series detected for {species}, skipping")
        return False
    median_val = np.median(points)
    q25, q75 = np.percentile(points, [25, 75])
    iqr = q75 - q25
    if median_val > 0 and iqr < 0.05 * median_val:
        print(f"Near-constant series detected for {species}, skipping")
        return False
    return True

def _select_cp_strict(bkps, n_points, years, last_year, mid_year, window, restrict_to_late_half):
    cps = [int(b) for b in bkps if int(b) != n_points]
    if not cps:
        return None
    for b in cps:
        if b < 5 or b > n_points - 5:
            return None
        if years[b] >= last_year - window:
            return None
        if restrict_to_late_half and years[b] < mid_year:
            return None
    return cps[-1]

def _try_bottomup(series_df, species, late_transition_window, variable="ABUNDANCE", restrict_to_late_half=True):
    points = series_df[variable].values
    years = series_df['YEAR'].values
    last_year = years[-1]
    mid_year = years[0] + (years[-1] - years[0]) * MIN_CP_FRAC
    n_points = len(points)
    if not _series_qualifies(points, species):
        return None
    bottomup_models = ["l2", "rbf", "linear", "normal", "ar", "l1", "cosine"]
    for n_bkps in [1, 2, 3, 4, 5]:
        for model in bottomup_models:
            try:
                algo = rpt.BottomUp(model=model, min_size=5).fit(points)
                bkps = algo.predict(n_bkps=n_bkps)
            except Exception:
                continue
            cp = _select_cp_strict(bkps, n_points, years, last_year, mid_year, late_transition_window, restrict_to_late_half)
            if cp is None:
                continue
            transition_year = years[cp]
            print(f"BottomUp transition for {species}: {int(transition_year)} (Method: BottomUp_{model}_nbps{n_bkps})")
            return cp
    print(f"No valid transition found with BottomUp for {species}, skipping")
    return None

def _try_strict_method(series_df, species, late_transition_window, method_name, variable="ABUNDANCE", restrict_to_late_half=True):
    points = series_df[variable].values
    years = series_df['YEAR'].values
    last_year = years[-1]
    mid_year = years[0] + (years[-1] - years[0]) * MIN_CP_FRAC
    n_points = len(points)
    if not _series_qualifies(points, species):
        return None
    models = ["l2", "rbf", "linear", "normal", "ar", "l1", "cosine"]
    if method_name == 'dynp':
        for model in models:
            try:
                algo = rpt.Dynp(model=model, min_size=5, jump=1).fit(points)
                bkps = algo.predict(n_bkps=1)
            except Exception:
                continue
            cp = _select_cp_strict(bkps, n_points, years, last_year, mid_year, late_transition_window, restrict_to_late_half)
            if cp is None:
                continue
            transition_year = years[cp]
            print(f"Dynp transition for {species}: {int(transition_year)} (Method: Dynp_{model})")
            return cp
        return None
    if method_name == 'pelt':
        penalties = [10, 5, 3, 2, 1]
        for pen in penalties:
            for model in models:
                try:
                    algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(points)
                    penalty_value = pen * np.log(n_points) * np.var(points)
                    bkps = algo.predict(pen=penalty_value)
                except Exception:
                    continue
                cp = _select_cp_strict(bkps, n_points, years, last_year, mid_year, late_transition_window, restrict_to_late_half)
                if cp is None:
                    continue
                transition_year = years[cp]
                print(f"Pelt transition for {species}: {int(transition_year)} (Method: Pelt_{model}_pen{pen})")
                return cp
        return None
    if method_name == 'binseg':
        for n_bkps in [1, 2, 3, 4, 5]:
            for model in models:
                try:
                    algo = rpt.Binseg(model=model, min_size=5).fit(points)
                    bkps = algo.predict(n_bkps=n_bkps)
                except Exception:
                    continue
                cp = _select_cp_strict(bkps, n_points, years, last_year, mid_year, late_transition_window, restrict_to_late_half)
                if cp is None:
                    continue
                transition_year = years[cp]
                print(f"BinSeg transition for {species}: {int(transition_year)} (Method: BinSeg_{model}_nbps{n_bkps})")
                return cp
        return None
    if method_name == 'window':
        window_sizes = [20, 30, 40, 50, 60]
        for n_bkps in [1, 2, 3, 4, 5]:
            for width in window_sizes:
                try:
                    algo = rpt.Window(width=width, model="l2").fit(points)
                    bkps = algo.predict(n_bkps=n_bkps)
                except Exception:
                    continue
                cp = _select_cp_strict(bkps, n_points, years, last_year, mid_year, late_transition_window, restrict_to_late_half)
                if cp is None:
                    continue
                transition_year = years[cp]
                print(f"Window transition for {species}: {int(transition_year)} (Method: Window_width{width}_nbps{n_bkps})")
                return cp
        return None
    raise ValueError(f"Unknown method: {method_name}")

def find_transition(series_df, species, late_transition_window, variable="ABUNDANCE", plot_fig=False, restrict_to_late_half=True, method_filter=None):
    effective_restrict = True
    if method_filter == 'bottomup':
        result = _try_bottomup(series_df, species, late_transition_window, variable=variable, restrict_to_late_half=effective_restrict)
        if result is None:
            print(f"No valid transition found for {species}, skipping")
        return result
    if method_filter in ('dynp', 'pelt', 'binseg', 'window'):
        return _try_strict_method(series_df, species, late_transition_window, method_filter, variable=variable, restrict_to_late_half=effective_restrict)
    if method_filter is None:
        for m in ['dynp', 'pelt', 'binseg', 'window']:
            result = _try_strict_method(series_df, species, late_transition_window, m, variable=variable, restrict_to_late_half=effective_restrict)
            if result is not None:
                return result
        return _try_bottomup(series_df, species, late_transition_window, variable=variable, restrict_to_late_half=effective_restrict)
    raise ValueError(f"Unknown method_filter: {method_filter}")

def plot_dl_predictions(species_name, species_data, data_taxonomy, genus_taxonomy):
    result = process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, late_transition_window, species_to_plot=None, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy, output_file=None, cp_method=None, do_plots=False)
    if result is None or result.get('error') is not None:
        print(f"No valid result for {species_name}, skipping")
        return
    ts = result.get('ts')
    transition_year = result.get('transition_year')
    if ts is None or transition_year is None:
        print(f"Missing ts or transition_year for {species_name}, skipping")
        return
    years = np.asarray(ts.state['year'].values)
    transition_idx = int(np.where(years == transition_year)[0][0])
    eval_index = int(0.9 * transition_idx)
    if eval_index < 0:
        eval_index = 0
    save_dl_predictions_pdf(ts, species_name, eval_index)

def plot_dl_predictions_pdf(ts, species_name, eval_index, output_dir='pdf'):
    try:
        fig = ts.make_plotly(ens_avg=True, eval_index=eval_index)
        fig.add_vline(x=eval_index, line_dash="dash", line_color="grey")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_image(os.path.join(output_dir, f"{species_name}_dl_predictions.pdf"))
        print(f"PDF plot saved for {species_name}")
    except Exception as e:
        print(f"Error plotting DL predictions PDF for {species_name}: {str(e)}")

def _lowess_effective_df(x, span):
    n = len(x)
    k = max(2, int(np.ceil(span * n)))
    df = 0.0
    for i in range(n):
        d = np.abs(x - x[i])
        maxd = np.partition(d, k - 1)[k - 1]
        if maxd <= 0.0:
            df += 1.0
            continue
        u = d / maxd
        w = np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0)
        S0 = w.sum()
        S1 = (w * x).sum()
        S2 = (w * x * x).sum()
        det = S0 * S2 - S1 * S1
        if det <= 0.0:
            df += 1.0
            continue
        df += (S2 - 2.0 * S1 * x[i] + S0 * x[i] * x[i]) / det
    return float(df)

def robust_detrending(ts, series_length, method_preference='Lowess'):
    methods_to_try = []
    if method_preference == 'Lowess':
        methods_to_try = [('Lowess', {'span': 0.3}), ('Lowess', {'span': 0.4}), ('Lowess', {'span': 0.5}), ('Lowess', {'span': 0.6})]
    else:
        methods_to_try = [('Gaussian', {'bandwidth': 0.3}), ('Gaussian', {'bandwidth': 0.4}), ('Gaussian', {'bandwidth': 0.5}), ('Gaussian', {'bandwidth': 0.6})]
    if method_preference == 'Lowess':
        methods_to_try.extend([('Gaussian', {'bandwidth': 0.3}), ('Gaussian', {'bandwidth': 0.4})])
    else:
        methods_to_try.extend([('Lowess', {'span': 0.3}), ('Lowess', {'span': 0.4})])
    best_aicc = np.inf
    best_method = None
    best_params = None
    first_ok_method = None
    first_ok_params = None
    last_error = None
    for method, params in methods_to_try:
        try:
            if method == 'Lowess':
                ts.detrend(method='Lowess', span=params['span'])
            elif method == 'Gaussian':
                ts.detrend(method='Gaussian', bandwidth=params['bandwidth'])
        except Exception as e:
            last_error = e
            if 'smoothing' in ts.state.columns:
                ts.state['smoothing'] = np.nan
            if 'residuals' in ts.state.columns:
                ts.state['residuals'] = ts.state['state'].copy()
            continue
        if 'smoothing' not in ts.state.columns or 'residuals' not in ts.state.columns:
            continue
        smoothing_vals = ts.state['smoothing'].values
        original_vals = ts.state['state'].values
        valid_mask = ~np.isnan(smoothing_vals) & ~np.isnan(original_vals)
        n = int(np.sum(valid_mask))
        if n <= 10:
            return ('success', None)
        y = original_vals[valid_mask]
        fit = smoothing_vals[valid_mask]
        rss = float(np.sum((y - fit) ** 2))
        if rss <= 0.0:
            continue
        if method == 'Lowess':
            df = _lowess_effective_df(np.arange(n, dtype=float), params['span'])
            if df >= n - 2.0:
                continue
            aicc = n * np.log(rss / n) + 2.0 * df + (2.0 * df * (df + 1.0)) / (n - df - 1.0)
            if aicc < best_aicc:
                best_aicc = aicc
                best_method = method
                best_params = params
        else:
            if first_ok_method is None:
                first_ok_method = method
                first_ok_params = params
    chosen_method = best_method if best_method is not None else first_ok_method
    chosen_params = best_params if best_params is not None else first_ok_params
    if chosen_method is None:
        if last_error is not None:
            print(f"All detrending methods failed. Last error: {last_error}")
            return ('failed', last_error)
        print("All detrending methods overfitted.")
        return ('overfitting', 'all detrending methods overfitted')
    if chosen_method == 'Lowess':
        ts.detrend(method='Lowess', span=chosen_params['span'])
        print(f"Successfully detrended using {chosen_method} with params {chosen_params} (AICc={best_aicc:.3g})")
    else:
        ts.detrend(method='Gaussian', bandwidth=chosen_params['bandwidth'])
        print(f"Successfully detrended using {chosen_method} with params {chosen_params}")
    return ('success', None)

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
    mask = ~np.isnan(residuals)
    residuals_clean = residuals[mask]
    frac_clean = frac[mask]
    if len(residuals_clean) < 2 * period:
        return
    unique_frac_clean = np.unique(np.round(frac_clean, 3))
    if len(unique_frac_clean) != 12:
        return
    result = sm.tsa.seasonal_decompose(residuals_clean, model='additive', period=period, extrapolate_trend='freq')
    seasonal = result.seasonal
    out = residuals.copy()
    out[mask] = residuals_clean - seasonal
    ts.state['residuals'] = out

def brook_zero_handling(species_data):
    species_data = species_data.sort_values('TIME').copy()
    abundance = species_data['ABUNDANCE'].values.astype(float).copy()
    non_zero = abundance[abundance != 0]
    if len(non_zero) == 0:
        return species_data
    min_non_zero = non_zero.min()
    n = len(abundance)
    i = 0
    while i < n:
        if abundance[i] == 0:
            j = i
            while j < n and abundance[j] == 0:
                j += 1
            abundance[i] = min_non_zero
            for k in range(i + 1, j):
                abundance[k] = np.nan
            i = j
        else:
            i += 1
    species_data['ABUNDANCE'] = abundance
    species_data = species_data.dropna(subset=['ABUNDANCE']).reset_index(drop=True)
    return species_data

def process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, late_transition_window, species_to_plot=None, data_taxonomy=None, genus_taxonomy=None, output_file='csv/bifurcation_results.csv', cp_method=None, do_plots=True):
    try:
        setup_warning_suppression(worker_process=True)
        taxonomic_group = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
        dataset = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
        species_data = brook_zero_handling(species_data)
        series = species_data[["TIME", "ABUNDANCE"]].copy()
        series.columns = ["YEAR", "ABUNDANCE"]
        original_length = len(series)
        raw_abundance = species_data["ABUNDANCE"]
        if original_length <= 500:
            series_adjusted = series.reset_index(drop=True)
            classifier_paths = classifier_paths_500
            max_size = 500
        elif original_length <= 1500:
            series_adjusted = series.reset_index(drop=True)
            classifier_paths = classifier_paths_1500
            max_size = 1500
        else:
            series_adjusted = series.iloc[-1500:].reset_index(drop=True)
            classifier_paths = classifier_paths_1500
            max_size = 1500
        years_arr = series_adjusted['YEAR'].values
        mid_year = years_arr[0] + (years_arr[-1] - years_arr[0]) * MIN_CP_FRAC
        transition_idx = find_transition(series_adjusted, species_name, late_transition_window, method_filter=cp_method)
        if transition_idx is not None:
            if transition_idx >= len(series_adjusted):
                transition_idx = len(series_adjusted) - 1
            if years_arr[transition_idx] < mid_year:
                transition_idx = find_transition(series_adjusted, species_name, late_transition_window, restrict_to_late_half=True, method_filter=cp_method)
        if transition_idx is None:
            return {'species': species_name, 'dataset': dataset, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': 0.0}
        if transition_idx >= len(series_adjusted):
            transition_idx = len(series_adjusted) - 1
        change_points = [transition_idx] if transition_idx is not None else []
        transition_year = series_adjusted['YEAR'].iloc[min(transition_idx, len(series_adjusted) - 1)]
        max_year = float(species_data['TIME'].max())
        if transition_year >= max_year - late_transition_window:
            return {'species': species_name, 'dataset': dataset, 'error': 'late_transition'}
        ts = ewstools.TimeSeries(series_adjusted['ABUNDANCE'].values, transition=transition_idx)
        ts.time_values = series_adjusted['YEAR'].values
        ts.state['year'] = series_adjusted['YEAR'].values
        detrend_status, detrend_message = robust_detrending(ts, len(series_adjusted), method_preference='Lowess')
        if detrend_status == 'overfitting':
            return {'species': species_name, 'dataset': dataset, 'error': 'overfitting'}
        if detrend_status == 'failed':
            raise ValueError(f"Failed to detrend time series: {detrend_message}")
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
        classifier_ran = False
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
                classifier_ran = True
        if not classifier_ran:
            mean_probs = pd.Series([0, 0, 0, 1], index=[0, 1, 2, 3])
            raw_bifurcation_type = 'no-bifurcation'
            final_bifurcation_type = 'no-bifurcation'
            best_probability = 0.0
        if len(ts.dl_preds) > 0:
            last_pred_pos = int(ts.dl_preds['time'].max())
            if last_pred_pos >= len(series_adjusted):
                last_pred_pos = len(series_adjusted) - 1
            last_pred_year = series_adjusted['YEAR'].iloc[last_pred_pos]
            if last_pred_year < transition_year - late_transition_window:
                return {'species': species_name, 'dataset': dataset, 'error': 'preds_ended_before_cp'}
        if do_plots and species_to_plot is not None and (len(species_to_plot) == 0 or species_name in species_to_plot):
            plot_dl_predictions(ts, species_name, eval_index)
        pdf_species = ['Anaxyrus sp7', 'Alcyonium digitatum', 'Apostichopus parvimensis', 'Arphia']
        if do_plots and species_name in pdf_species:
            plot_dl_predictions_pdf(ts, species_name, eval_index)
        if classifier_ran:
            ews_metrics = {int(k): float(v) for k, v in mean_probs.to_dict().items()}
        else:
            ews_metrics = {}
        return {'species': species_name, 'dataset': dataset, 'tipping_point_type': final_bifurcation_type, 'raw_tipping_point_type': raw_bifurcation_type, 'ews_metrics': ews_metrics, 'change_points': change_points, 'taxonomic_group': taxonomic_group, 'transition_year': transition_year, 'smoothing_avg': smoothing_avg, 'current_abundance': raw_abundance.iloc[-1] if len(raw_abundance) > 0 else 0, 'max_abundance': raw_abundance.max(), 'best_probability': best_probability, 'ts': ts}
    except Exception as e:
        print(f"Error processing {species_name}: {str(e)}")
        traceback.print_exc()
        tf.keras.backend.clear_session()
        gc.collect()
        taxonomic_group_error = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
        dataset_error = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
        return {'species': species_name, 'dataset': dataset_error, 'tipping_point_type': 'no-bifurcation', 'raw_tipping_point_type': 'no-bifurcation', 'ews_metrics': {}, 'change_points': [], 'taxonomic_group': taxonomic_group_error, 'transition_year': None, 'smoothing_avg': None, 'current_abundance': 0, 'max_abundance': 0, 'best_probability': 0.0}
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

def update_taxonomic_groups_from_mapping(results_file='csv/bifurcation_results.csv'):
    try:
        results_df = read_results_with_fixed_columns(results_file)
        taxon_mapping_df = pd.read_csv('csv/taxonomic_cache.csv', encoding='utf-8')
        taxon_mapping_dict = dict(zip(taxon_mapping_df['species'].astype(str).str.lower(), taxon_mapping_df['taxonomic_group']))
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
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"Taxonomic groups updated for {updated_count} species in {results_file}")
    except Exception as e:
        print(f"Error updating taxonomic groups in {results_file}: {str(e)}")

def _classify_species_no_cache(species_name, data_taxonomy, genus_taxonomy):
    if not species_name or pd.isna(species_name) or str(species_name).strip() == '':
        return (str(species_name), 'Other')
    species_str = str(species_name).strip()
    if species_str in EXCLUDED_SPECIES_LABELS:
        return (species_str, 'Other')
    if species_str.upper() in BANDING_CODE_TO_NAME:
        return (species_str, 'Aves')
    if re.search(r'\b(miscellaneous|litter|unknown)\b', species_str.lower()):
        return (species_str, 'Other')
    if data_taxonomy is not None and species_str in data_taxonomy:
        group = data_taxonomy[species_str]
        if group != 'Other':
            return (species_str, group)
    if 'sp' in species_str.lower() or 'cf.' in species_str.lower() or 'aff.' in species_str.lower():
        parts = species_str.split()
        if len(parts) > 0:
            genus = parts[0]
            if genus_taxonomy is not None and genus in genus_taxonomy:
                group = genus_taxonomy[genus]
                if group != 'Other':
                    return (species_str, group)
            tax_info = get_taxonomic_info(genus)
            if tax_info:
                group = assign_taxonomic_group(tax_info, genus)
                if group != 'Other':
                    return (species_str, group)
    group = classify_taxonomic_group(species_str)
    return (species_str, group)

def ensure_all_species_classified(dl_file, fail_file, data_taxonomy, genus_taxonomy, n_jobs=-1):
    from main_orchestrator import _atomic_write_csv
    species_set = set()
    if os.path.exists(dl_file) and os.path.getsize(dl_file) > 0:
        df = pd.read_csv(dl_file)
        if 'species' in df.columns:
            species_set.update(df['species'].astype(str).str.strip().unique())
    if os.path.exists(fail_file) and os.path.getsize(fail_file) > 0:
        df = pd.read_csv(fail_file)
        if 'species' in df.columns:
            species_set.update(df['species'].astype(str).str.strip().unique())
    species_set = {sp for sp in species_set if sp and sp != 'Other'}
    cache_file = 'csv/taxonomic_cache.csv'
    existing = pd.DataFrame(columns=['species', 'taxonomic_group'])
    existing_map = {}
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            raw = pd.read_csv(cache_file)
            if 'species' in raw.columns and 'taxonomic_group' in raw.columns:
                raw['species'] = raw['species'].astype(str).str.strip()
                raw = raw.drop_duplicates(subset=['species'], keep='last')
                existing = raw
                existing_map = dict(zip(raw['species'], raw['taxonomic_group']))
        except Exception:
            existing_map = {}
    to_classify = sorted(sp for sp in species_set if existing_map.get(sp, 'Other') == 'Other')
    print(f"Reclassifying {len(to_classify)} species in parallel (n_jobs={n_jobs}); {len(existing_map)} already cached.")
    if not to_classify:
        return existing
    flushed = 0
    for start in range(0, len(to_classify), 100):
        chunk = to_classify[start:start + 100]
        results = Parallel(n_jobs=n_jobs)(delayed(_classify_species_no_cache)(sp, data_taxonomy, genus_taxonomy) for sp in chunk)
        new_df = pd.DataFrame(results, columns=['species', 'taxonomic_group'])
        new_df['species'] = new_df['species'].astype(str).str.strip()
        combined = pd.concat([existing[~existing['species'].isin(new_df['species'])], new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['species'], keep='last')
        _atomic_write_csv(cache_file, combined)
        existing = combined
        flushed += len(chunk)
        print(f"Flushed {flushed}/{len(to_classify)} species to {cache_file}.")
    print(f"Updated {cache_file} with {len(existing)} species.")
    return existing

EXCLUDED_SPECIES_LABELS = {'32 Species Weeds'}

def _augment_with_undetermined(results_df, probability_threshold):
    failed_df = pd.read_csv('csv/failed_species.csv')
    failed_df['species'] = failed_df['species'].astype(str).str.strip()
    if 'reason' in failed_df.columns:
        failed_df = failed_df[failed_df['reason'].astype(str).str.strip() != 'taxonomic_excluded']
    results_df = results_df.copy()
    results_df['species'] = results_df['species'].astype(str).str.strip()
    short_df = pd.read_csv('csv/bifurcation_results_short.csv')
    short_df['species'] = short_df['species'].astype(str).str.strip()
    short_df['_k'] = short_df['species'].str.split().str[:2].str.join(' ')
    results_df['_k'] = results_df['species'].str.split().str[:2].str.join(' ')
    short_map = dict(zip(short_df['_k'], short_df['tipping_point_type']))
    results_df['tipping_point_type'] = results_df['_k'].map(short_map).fillna('undetermined')
    results_df = results_df.drop(columns=['_k'])
    undetermined_species = sorted(set(failed_df['species']) - set(results_df['species']))
    undetermined_df = pd.DataFrame({'species': undetermined_species, 'dataset': 'failed_species', 'timeseries_id': 'POOLED', 'tipping_point_type': 'undetermined'})
    undetermined_df['taxonomic_group'] = undetermined_df['species'].apply(get_cached_taxonomic_group)
    undetermined_df = undetermined_df[~undetermined_df['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    undetermined_df = undetermined_df[~undetermined_df['species'].isin(EXCLUDED_SPECIES_LABELS)]
    return pd.concat([results_df, undetermined_df], ignore_index=True)

def create_summary_figure(results_df, probability_threshold):
    if 'tipping_point_type' not in results_df.columns or 'taxonomic_group' not in results_df.columns:
        print("Warning: Could not create summary figure - missing required columns")
        return
    results_df = _augment_with_undetermined(results_df, probability_threshold)
    plot_df = results_df.copy()
    plot_df = plot_df[plot_df['taxonomic_group'] != 'Other']
    plot_df = plot_df[~plot_df['taxonomic_group'].isin(EXCLUDED_GROUPS)]
    tipping_point_order = ['undetermined', 'no-bifurcation', 'fold bifurcation', 'transcritical bifurcation', 'Hopf bifurcation']
    tipping_point_colors = {'undetermined': '#d62728', 'no-bifurcation': '#999999', 'fold bifurcation': '#1f77b4', 'transcritical bifurcation': '#ff7f0e', 'Hopf bifurcation': '#9467bd'}
    summary = plot_df.groupby(['taxonomic_group', 'tipping_point_type']).size().unstack(fill_value=0)
    existing_columns = [col for col in tipping_point_order if col in summary.columns]
    summary = summary[existing_columns]
    summary = summary.reindex(TAXONOMIC_ORDER, fill_value=0)
    summary = summary.drop(index='Other', errors='ignore')
    summary = summary.loc[summary.sum(axis=1) > 0]
    total_row = summary.sum(axis=0)
    total_row.name = 'Total'
    summary_with_total = pd.concat([summary, total_row.to_frame().T])
    print("\nTaxonomic group summary (counts of bifurcation types, incl. undetermined):")
    print("-" * 80)
    header = "Taxonomic Group".ljust(25) + "".join([t.ljust(22) for t in ['undet', 'no-bif', 'fold', 'transc', 'Hopf']]) + "Total"
    print(header)
    print("-" * 80)
    for group in summary_with_total.index:
        row = summary_with_total.loc[group]
        total = row.sum()
        counts_str = (str(row.get('undetermined', 0)).rjust(6) + " " + str(row.get('no-bifurcation', 0)).rjust(6) + " " + str(row.get('fold bifurcation', 0)).rjust(6) + " " + str(row.get('transcritical bifurcation', 0)).rjust(6) + " " + str(row.get('Hopf bifurcation', 0)).rjust(6))
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
    undet_col = 'undetermined'
    undet_idx = summary.columns.get_loc(undet_col) if undet_col in summary.columns else -1
    undet_pcts = (summary.iloc[:, undet_idx].values / row_totals.replace(0, np.nan)) * 100 if undet_idx != -1 else np.zeros(len(summary))
    cap_mask = (undet_pcts >= 85) & (undet_pcts < 100)
    for i, col in enumerate(summary.columns):
        values = summary[col].values
        percentages = (values / row_totals.replace(0, np.nan)) * 100
        percentages = percentages.fillna(0).values
        if col == undet_col:
            percentages[cap_mask] = 85
        elif undet_idx != -1:
            other_sum = 100 - undet_pcts
            scale = np.ones_like(percentages)
            valid_scale_mask = cap_mask & (other_sum > 0)
            scale[valid_scale_mask] = 15 / other_sum[valid_scale_mask]
            percentages = percentages * scale
        color = tipping_point_colors.get(col, '#1f77b4')
        bar = ax.bar(summary.index, percentages, bottom=bottom, color=color, alpha=0.65, label=abbreviate_bifurcation_type(col))
        bars.append(bar)
        bottom += percentages
    for i, bar_collection in enumerate(bars):
        col = summary.columns[i]
        for j, rect in enumerate(bar_collection):
            height = rect.get_height()
            if height > 0:
                raw_value = summary.iloc[j, i]
                int_value = int(raw_value)
                raw_pct = (raw_value / row_totals.iloc[j]) * 100 if row_totals.iloc[j] > 0 else 0
                is_capped = (col == undet_col) and (raw_pct >= 85) and (raw_pct < 100)
                x_pos = rect.get_x() + rect.get_width() / 2
                y_pos = rect.get_y() + height / 2
                label = f'{int_value}*' if is_capped else f'{int_value}'
                ax.text(x_pos, y_pos, label, ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    plt.xlabel(None)
    plt.ylabel('Percentage within Group (%)')
    plt.xlim(-0.5, len(summary) - 0.5)
    plt.xticks(rotation=45, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Bifurcation type', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(False)
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/tipping_point_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()
