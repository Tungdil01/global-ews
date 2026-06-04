import pandas as pd
import numpy as np
from scipy import stats

def load_data(bif_path, raw_path):
    df_bif = pd.read_csv(bif_path)
    df_raw = pd.read_csv(raw_path, usecols=['valid_name','YEAR','MONTH','taxon','ABUNDANCE'])
    df_raw = df_raw.dropna(subset=['YEAR','valid_name','MONTH','ABUNDANCE'])
    df_raw['YEAR'] = df_raw['YEAR'].astype(int)
    df_raw['ABUNDANCE'] = pd.to_numeric(df_raw['ABUNDANCE'], errors='coerce')
    df_raw = df_raw[df_raw['ABUNDANCE'] > 0]
    return df_bif, df_raw

def filter_species_sibly(df_bif, df_raw):
    bif_species = set(df_bif['species'].str.strip())
    raw_groups = df_raw.groupby('valid_name')
    kept = set()
    for sp in bif_species:
        if sp not in raw_groups.groups:
            continue
        group = raw_groups.get_group(sp).copy()
        group['TIME'] = group['YEAR'] + (group['MONTH'] - 1) / 12.0
        group = group.sort_values('TIME')
        times = group['TIME'].values
        abund = group['ABUNDANCE'].values
        # find contiguous blocks where gap <= 1.1 years (step 4: contiguous subseries)
        blocks = []
        start = 0
        for i in range(1, len(times)):
            if times[i] - times[i-1] > 1.1:
                blocks.append((start, i))
                start = i
        blocks.append((start, len(times)))
        # select longest block with >=5 years (step 4: five or more years)
        best_block = None
        best_len = 0
        for s,e in blocks:
            if e - s >= 5:
                if e - s > best_len:
                    best_len = e - s
                    best_block = (s, e)
        if best_block is None:
            continue
        s, e = best_block
        block_times = times[s:e]
        block_abund = abund[s:e]
        # step 4: non-zero, not all same, pgr calculable
        if np.any(block_abund <= 0) or np.var(block_abund) == 0:
            continue
        kept.add(sp)
    return kept

def save_cropped(df_bif, kept_species, output_path):
    df_cropped = df_bif[df_bif['species'].str.strip().isin(kept_species)]
    df_cropped.to_csv(output_path, index=False)

def create_verification(kept_species, species_list, output_path):
    df_verif = pd.DataFrame({'species': species_list})
    df_verif['present'] = df_verif['species'].str.strip().isin(kept_species).astype(int)
    df_verif.to_csv(output_path, index=False)

def main():
    bif_path = 'csv/bifurcation_results_short.csv'
    raw_path = 'data/biotime_v2_full_2025/biotime_v2_rawdata_2025.csv'
    cropped_path = bif_path
    verif_path = 'csv/verification_sibly.csv'
    df_bif, df_raw = load_data(bif_path, raw_path)
    kept = filter_species_sibly(df_bif, df_raw)
    save_cropped(df_bif, kept, cropped_path)
    verification_list = [
        '32 Species Weeds', 'Abagrotis placida', 'Abarenicola pacifica', 'Abbottina rivularis',
        'Abies alba', 'Abies concolor', 'Abisara fylla', 'Ablabesmyia annulata', 'Abralia veranyi',
        'Abraliopsis morisii', 'Abraxas grossulariata', 'Abraxas niphonibia', 'Abudefduf bengalensis',
        'Abudefduf notatus', 'Abyssocucumis abyssorum', 'Abyssogena phaseoliformis', 'Abyssoninoe hibernica',
        'Abyssoninoe spB', 'Acacia alleniana', 'Acacia brockii', 'Acacia cataractae', 'Acacia celsa',
        'Acacia difficilis', 'Acacia dimidiata', 'Acacia', 'Acacia gonocarpa', 'Acacia humifusa',
        'Acacia multisiliqua', 'Acacia platycarpa', 'Acacia producta', 'Acacia scopulorum',
        'Acacia tolmerensis', 'Acacia torulosa', 'Acanthaluteres brownii', 'Acanthaluteres vittiger',
        'Acanthancora cyanocrypta', 'Acantharchus pomotis', 'Acanthephyra purpurea', 'Acanthephyra quadrispinosa',
        'Acantherus piperatus', 'Acanthis', 'Acanthoceras zachariasii', 'Acanthochromis polyacanthus',
        'Acanthocyclops', 'Acanthocyclops robustus', 'Acanthogobius lactipes', 'Acanthogorgia sp1',
        'Acanthohaustorius shoemakeri', 'Acanthohaustorius similis', 'Acanthohaustorius sp',
        'Acanthohaustorius spinosus', 'Acanthoica', 'Acanthoica quattrospina', 'Acanthophora',
        'Acanthosphaera zachariasii', 'Acanthostepheia malmgreni', 'Acanthostomella', 'Acanthurus bariene',
        'Acanthurus blochii', 'Acanthurus dussumieri', 'Acanthurus leucocheilus', 'Acanthurus mata',
        'Acanthurus nigricauda', 'Acanthurus sp', 'Acanthurus xanthopterus', 'Acari sp1', 'Acarnus erithacus',
        'Acartia (Acanthacartia) fossae', 'Acartia (Acartiura) hudsonica', 'Acartia (Acartiura) tranteri',
        'Acartia sp', 'Acaulis primarius', 'Accalathura crenulata', 'Accipiter gularis', 'Accipiter unidentified',
        'Acentria ephemerella', 'Acentrogobius audax', 'Aceratium concinnum', 'Acer campestre', 'Aceroides latipes',
        'Acer platanoides', 'Acer pseudoplatanus', 'Acer saccharum', 'Acestrorhynchus lacustris', 'Acetabularia sp',
        'Acheilognathus macropterus', 'Achelia', 'Achelia spinosa', 'Achnanthes longipes', 'Achnanthes minutissima',
        'Achnatherum sibiricum', 'Achondrostoma arcasii', 'Acila castrensis', 'Acilius', 'Acipenser fulvescens',
        'Acmispon plebeius', 'Acosmeryx naga', 'Acosmetia caliginosa', 'Acrocalanus monachus',
        'Acrocephalus arundinaceus', 'Acrocirridae', 'Acroloxus lacustris', 'Acromegalomma lobiferum',
        'Acromegalomma pigmentum', 'Acromegalomma quadrioculatum', 'Acromegalomma splendidum',
        'Acromegalomma vesiculosum', 'Acroneuria abnormis', 'Acronicta aceris', 'Acronicta auricoma',
        'Acronicta dactylina', 'Acronicta impleta', 'Acronicta leporina', 'Acronicta psi', 'Acronicta strigosa',
        'Acronychia acidula', 'Acronychia acronychioides', 'Acronychia pedunculata', 'Acropoma japonicum',
        'Acropora cytherea', 'Acropora humilis', 'Acropora hyacinthus', 'Acropora nana', 'Acropora robusta',
        'Acropora secale', 'Acropora sp', 'Acropora valida', 'Acrotritia ardua', 'Actebia praecox', 'Actinastrum',
        'Actinauge', 'Actiniidae', 'Actinocyclus senarius', 'Actinocythereis sp1', 'Actinodaphne henryi',
        'Actinopoda', 'Actinopterygii', 'Actinoptychus', 'Actinoseta sp1', 'Actinostemon concepcionis',
        'Actinostemon klotzschii', 'Actinotrichia', 'Actophilornis africanus', 'Acupalpus exiguus',
        'Adalaria proxima', 'Adelphagrotis indeterminata', 'Adelphagrotis stellaris', 'Adenandra villosa',
        'Adenomera marmorata', 'Adeonellopsis sp1', 'Adicella reducta', 'Adna anglica', 'Adoristes ovatus',
        'Aechmophorus occidentalis', 'Aedicira', 'Aegaeon cataphractus', 'Aeginopsis', 'Aeolidiella glauca',
        'Aeoliscus strigatus', 'Aepyceros melampus'
    ]
    create_verification(kept, verification_list, verif_path)

if __name__ == '__main__':
    main()
