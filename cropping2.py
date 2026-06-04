import pandas as pd
import numpy as np

MICRO_GROUPS = {'Bacteria','Choanoflagellata','Ciliophora','Cryptophyta','Haptophyta','Ochrophyta','Pancryptista','Rhizaria','Myzozoa','Euglenozoa'}
GAP_THRESHOLD = 50
MIN_SPAN_MACRO = 20      # years for macroorganisms
MIN_SPAN_MICRO = 20      # months for microorganisms

def load_data(bif_path, raw_path):
    df_bif = pd.read_csv(bif_path)
    df_raw = pd.read_csv(raw_path, usecols=['valid_name','YEAR','MONTH','taxon'])
    df_raw = df_raw.dropna(subset=['YEAR','valid_name','MONTH'])
    df_raw['YEAR'] = df_raw['YEAR'].astype(int)
    return df_bif, df_raw

#def filter_by_probability(df, threshold):
#    return df[df['best_probability'] >= threshold]

def filter_species(df_bif, df_raw, gap_threshold, micro_groups, min_span_macro, min_span_micro):
    bif_species = set(df_bif['species'].str.strip())
    raw_groups = df_raw.groupby('valid_name')
    kept = set()
    for sp in bif_species:
        if sp not in raw_groups.groups:
            continue
        group = raw_groups.get_group(sp)
        taxon = group['taxon'].dropna().iloc[0] if not group['taxon'].dropna().empty else None
        is_micro = taxon in micro_groups if taxon else False
        if is_micro:
            months = group['YEAR'] * 12 + group['MONTH']
            months_unique = np.sort(months.unique())
            if len(months_unique) == 0:
                continue
            span_months = months_unique[-1] - months_unique[0]
            max_gap = np.max(np.diff(months_unique)) if len(months_unique) > 1 else 0
            if span_months < min_span_micro:
                continue
        else:
            years = np.sort(group['YEAR'].unique())
            if len(years) == 0:
                continue
            span_years = years[-1] - years[0]
            max_gap = np.max(np.diff(years)) if len(years) > 1 else 0
            if span_years < min_span_macro:
                continue
        if max_gap > gap_threshold:
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
    verif_path = 'csv/verification.csv'
#    probability_threshold = 0.5
    df_bif, df_raw = load_data(bif_path, raw_path)
#    df_bif = filter_by_probability(df_bif, probability_threshold)
    kept = filter_species(df_bif, df_raw, GAP_THRESHOLD, MICRO_GROUPS, MIN_SPAN_MACRO, MIN_SPAN_MICRO)
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
