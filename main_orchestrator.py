#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import gc, sys, time, pickle, shutil, multiprocessing as mp, queue, tempfile, json, traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
from main_dl import load_combined_data, process_species, process_one_species_filter, compute_quality_score, save_results, get_cached_taxonomic_group, EXCLUDED_GROUPS, create_summary_figure, build_species_whitelist
from main_dev import process_dev_for_species

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
DL_FILE = 'csv/bifurcation_results_dl_ews.csv'
DEV_FILE = 'csv/bifurcation_results_dev.csv'
FAIL_FILE = 'csv/failed_species.csv'
FILTER_ERROR_LOG = 'csv/filter_errors.log'

def map_dl_type(dl_label):
    if dl_label in ['fold', 'fold bifurcation', 'transcritical', 'transcritical bifurcation']:
        return 'fold'
    elif dl_label in ['Hopf', 'Hopf bifurcation']:
        return 'hopf'
    elif dl_label in ['no-bifurcation', 'none']:
        return 'none'
    else:
        return 'other'

def map_dev_type(dev_label):
    if dev_label == 'fold_or_transcritical':
        return 'fold'
    elif dev_label == 'Hopf':
        return 'hopf'
    elif dev_label == 'period-doubling':
        return 'period_doubling'
    elif dev_label == 'no-bifurcation':
        return 'none'
    else:
        return 'other'

def compute_agreement(dl_results, dev_results):
    df_dl = pd.DataFrame(dl_results)
    df_dev = pd.DataFrame(dev_results)
    merged = pd.merge(df_dl, df_dev, on='species', suffixes=('_dl', '_dev'))
    if merged.empty:
        print("No matching species for comparison.")
        return
    merged['dl_map'] = merged['tipping_point_type'].apply(map_dl_type)
    merged['dev_map'] = merged['bifurcation_type'].apply(map_dev_type)
    merged['dev_bin'] = merged['early_warning'].astype(int)
    merged['dl_bin'] = ((merged['best_probability'] >= 0.7) & (merged['dl_map'] != 'none')).astype(int)
    df_type = merged[merged['best_probability'] >= 0.7].copy()
    df_type = df_type[df_type['dev_map'] != 'period_doubling']
    if len(df_type) == 0:
        print('No valid rows for type comparison after filtering.')
    else:
        ct_type = pd.crosstab(df_type['dev_map'], df_type['dl_map'], margins=True)
        print('\nContingency table: DEV (at transition year) vs DL (at transition) bifurcation type')
        print(ct_type)
        kappa_type = cohen_kappa_score(df_type['dev_map'], df_type['dl_map'])
        acc_type = accuracy_score(df_type['dev_map'], df_type['dl_map'])
        print(f"Cohen's kappa for type agreement (DL confidence >=0.7, DEV != period-doubling): {kappa_type:.3g}")
        print(f"Observed agreement (accuracy): {acc_type:.3g}")
    ct_bin = pd.crosstab(merged['dev_bin'], merged['dl_bin'], margins=True)
    print('\nContingency table: DEV early_warning vs DL (best>=0.7 & type!=no-bif)')
    print(ct_bin)
    kappa_bin = cohen_kappa_score(merged['dev_bin'], merged['dl_bin'])
    print(f"Cohen's kappa for binary agreement: {kappa_bin:.3g}")

def _atomic_write_csv(file_path, df):
    fd, tmp_path = tempfile.mkstemp(suffix='.csv', dir=os.path.dirname(file_path) or '.')
    try:
        with os.fdopen(fd, 'w') as f:
            df.to_csv(f, index=False)
        os.replace(tmp_path, file_path)
    except Exception:
        os.remove(tmp_path)
        raise

def _acquire_file_lock(lock_path, timeout=60):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise RuntimeError("Could not acquire file lock: " + lock_path)
            time.sleep(0.05)

def _release_file_lock(lock_path):
    os.remove(lock_path)

def append_to_csv(file_path, data_dict):
    if not data_dict or 'error' in data_dict:
        return
    lock_path = file_path + '.lock'
    _acquire_file_lock(lock_path)
    try:
        df = pd.DataFrame([data_dict])
        if 'dev_series' in df.columns:
            df = df.drop(columns=['dev_series'])
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            _atomic_write_csv(file_path, df)
        else:
            existing = pd.read_csv(file_path)
            key_cols = ['species', 'dataset'] if 'dataset' in existing.columns and 'dataset' in df.columns else ['species']
            if key_cols == ['species', 'dataset']:
                mask = (existing['species'] == data_dict['species']) & (existing['dataset'] == data_dict['dataset'])
                if mask.any():
                    existing = existing[~mask]
            else:
                if data_dict['species'] in existing['species'].values:
                    existing = existing[existing['species'] != data_dict['species']]
            combined = pd.concat([existing, df], ignore_index=True)
            _atomic_write_csv(file_path, combined)
    finally:
        _release_file_lock(lock_path)

def process_both_analyses(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl=True, run_dev=True, output_dl=DL_FILE, output_dev=DEV_FILE):
    result_dl = None
    result_dev = None
    if run_dl:
        try:
            result_dl = process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, data_taxonomy=data_taxonomy, output_file=output_dl)
            if result_dl is not None and 'error' not in result_dl:
                append_to_csv(output_dl, result_dl)
        except Exception as e:
            print(f"DL-EWS ERROR for {species_name}: {e}")
            result_dl = {'species': species_name, 'error': str(e)}
    if run_dev:
        try:
            result_dev = process_dev_for_species(species_name, species_data, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
        except Exception as e:
            print(f"DEV ERROR for {species_name}: {e}")
            result_dev = {'species': species_name, 'error': str(e)}
        if result_dev is not None and 'error' not in result_dev:
            dataset = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
            taxonomic_group = get_cached_taxonomic_group(species_name, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
            result_dev['dataset'] = dataset
            result_dev['taxonomic_group'] = taxonomic_group
            append_to_csv(output_dev, result_dev)
    return result_dl, result_dev

def _analysis_target(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl, run_dev, conn):
    dl_res, dev_res = process_both_analyses(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl=run_dl, run_dev=run_dev)
    conn.send((dl_res, dev_res))
    conn.close()

def run_analysis_with_timeout(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl=True, run_dev=True, timeout=1800):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_analysis_target, args=(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl, run_dev, child_conn))
    p.daemon = True
    p.start()
    p.join(timeout)
    try:
        if p.is_alive():
            p.terminate()
            p.join(2)
            if p.is_alive():
                p.kill()
                p.join()
            print(f"Timeout ({timeout}s) exceeded for {species_name}, skipping.")
            if run_dl and run_dev:
                return None, {'species': species_name, 'error': 'timeout'}
            elif run_dl:
                return {'species': species_name, 'error': 'timeout'}, None
            elif run_dev:
                return None, {'species': species_name, 'error': 'timeout'}
            else:
                return None, None
        if parent_conn.poll(1):
            result = parent_conn.recv()
            if isinstance(result, tuple) and len(result) == 2:
                return result
    except (EOFError, OSError):
        pass
    finally:
        parent_conn.close()
        child_conn.close()
    return None, {'species': species_name, 'error': 'child_error'}

def consumer_worker(q, stop_event, data_taxonomy, genus_taxonomy):
    processed_count = 0
    timeout_seconds = 1800
    while not stop_event.is_set():
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        species_name, species_data, run_dl, run_dev = item
        dl_res, dev_res = run_analysis_with_timeout(species_name, species_data, data_taxonomy, genus_taxonomy, run_dl=run_dl, run_dev=run_dev, timeout=timeout_seconds)
        dataset = species_data['dataset'].iloc[0] if 'dataset' in species_data.columns else 'Unknown'
        dl_error = run_dl and (dl_res is None or 'error' in dl_res)
        dev_error = run_dev and (dev_res is None or 'error' in dev_res)
        if dl_error or dev_error:
            timed_out = False
            if run_dl and (dl_res is None or (isinstance(dl_res, dict) and dl_res.get('error') == 'timeout')):
                timed_out = True
            if run_dev and (dev_res is None or (isinstance(dev_res, dict) and dev_res.get('error') == 'timeout')):
                timed_out = True
            reason = 'timeout' if timed_out else 'error'
            log_failed_task(species_name, dataset, reason)
        processed_count += 1
    print(f"Consumer processed {processed_count} species.")

def get_species_datasets(full_data):
    species_datasets = {}
    for sp, grp in full_data.groupby('SPECIES_ID')['dataset']:
        species_datasets[sp] = grp.unique().tolist()
    return species_datasets

def get_tasks_from_file(file_path):
    tasks = set()
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df = pd.read_csv(file_path)
        if 'species' in df.columns and 'dataset' in df.columns:
            for _, row in df.iterrows():
                tasks.add((str(row['species']).strip(), str(row['dataset']).strip()))
        elif 'species' in df.columns:
            for sp in df['species'].unique():
                tasks.add((str(sp).strip(), 'Unknown'))
    return tasks

def select_best_non_gbif_for_species(species, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, failure_log_func=None, dataset_candidates=None):
    sp_data = full_data[full_data['SPECIES_ID'] == species]
    non_gbif = dataset_candidates if dataset_candidates is not None else [d for d in sp_data['dataset'].unique() if d != 'GBIF']
    candidates = []
    for ds in non_gbif:
        try:
            result = process_one_species_filter(species, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, dataset_filter=ds, failure_log_func=failure_log_func, compute_quality=False, filter_func=check_annual_series)
        except Exception:
            continue
        if result is None:
            continue
        sp_name, data = result
        annual = data.groupby(np.floor(data['TIME']).astype(int))['ABUNDANCE'].mean().reset_index()
        annual.columns = ['YEAR', 'ABUNDANCE']
        annual = annual.dropna(subset=['ABUNDANCE'])
        candidates.append((ds, data, annual))
    if not candidates:
        return None, None
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]
    best_quality = -np.inf
    best_data = None
    best_dataset = None
    for ds, data, annual in candidates:
        quality = compute_quality_score(annual, reliability=3)
        if quality > best_quality:
            best_quality = quality
            best_data = data
            best_dataset = ds
    return best_data, best_dataset

def producer(species_datasets, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, queue, stop_event, dl_tasks, dev_tasks, failed_tasks, num_consumers, only_gbif=False):
    passed_count = 0
    try:
        if not only_gbif:
            for sp, ds_list in tqdm(species_datasets.items(), desc="Producer: queueing non-GBIF species"):
                if stop_event.is_set():
                    break
                non_gbif = [d for d in ds_list if d != 'GBIF']
                if not non_gbif:
                    continue
                existing_ds = None
                for ds in non_gbif:
                    if (sp.strip(), ds.strip()) in dl_tasks or (sp.strip(), ds.strip()) in dev_tasks:
                        existing_ds = ds
                        break
                if existing_ds is not None:
                    task_key = (sp.strip(), existing_ds.strip())
                    dl_done = task_key in dl_tasks
                    dev_done = task_key in dev_tasks
                    if dl_done and dev_done:
                        continue
                    if task_key in failed_tasks:
                        continue
                    sp_data = full_data[(full_data['SPECIES_ID'] == sp) & (full_data['dataset'] == existing_ds)]
                    if len(sp_data) == 0:
                        log_failed_task(sp, existing_ds, 'data_missing')
                        continue
                    if not dl_done or not dev_done:
                        while not stop_event.is_set():
                            try:
                                queue.put((sp, sp_data, not dl_done, not dev_done), timeout=0.1)
                                break
                            except queue.Full:
                                time.sleep(0.5)
                        passed_count += 1
                    continue
                candidate_ds = [d for d in non_gbif if (sp.strip(), d.strip()) not in failed_tasks]
                if not candidate_ds:
                    continue
                best_data, best_ds = select_best_non_gbif_for_species(sp, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, failure_log_func=log_failed_task, dataset_candidates=candidate_ds)
                if best_data is not None and best_ds is not None:
                    task_key = (sp.strip(), best_ds.strip())
                    if task_key not in dl_tasks and task_key not in dev_tasks and task_key not in failed_tasks:
                        while not stop_event.is_set():
                            try:
                                queue.put((sp, best_data, True, True), timeout=0.1)
                                break
                            except queue.Full:
                                time.sleep(0.5)
                        passed_count += 1
        else:
            for sp, ds_list in tqdm(species_datasets.items(), desc="Producer: queueing GBIF species"):
                if stop_event.is_set():
                    break
                if 'GBIF' not in ds_list:
                    continue
                task_key = (sp.strip(), 'GBIF')
                if task_key in failed_tasks:
                    continue
                dl_done = task_key in dl_tasks
                dev_done = task_key in dev_tasks
                if dl_done and dev_done:
                    continue
                if dl_done or dev_done:
                    sp_data = full_data[(full_data['SPECIES_ID'] == sp) & (full_data['dataset'] == 'GBIF')]
                    if len(sp_data) == 0:
                        log_failed_task(sp, 'GBIF', 'data_missing')
                        continue
                    if not dl_done or not dev_done:
                        while not stop_event.is_set():
                            try:
                                queue.put((sp, sp_data, not dl_done, not dev_done), timeout=0.1)
                                break
                            except queue.Full:
                                time.sleep(0.5)
                        passed_count += 1
                    continue
                gbif_result = process_one_species_filter(sp, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, dataset_filter='GBIF', failure_log_func=log_failed_task, filter_func=check_annual_series)
                if gbif_result is not None:
                    gbif_species, gbif_data = gbif_result
                    if task_key not in dl_tasks and task_key not in dev_tasks:
                        while not stop_event.is_set():
                            try:
                                queue.put((sp, gbif_data, True, True), timeout=0.1)
                                break
                            except queue.Full:
                                time.sleep(0.5)
                        passed_count += 1
    finally:
        print(f"Producer: {passed_count} species queued.")
        for _ in range(num_consumers):
            queue.put(None)

def setup_data(min_records):
    whitelist = None
    post1900_file = 'csv/species_post1900.csv'
    if os.path.exists(post1900_file) and os.path.getsize(post1900_file) > 0:
        try:
            whitelist_df = pd.read_csv(post1900_file)
            if 'species' in whitelist_df.columns:
                whitelist = set(whitelist_df['species'].tolist())
                print(f"Loaded whitelist of {len(whitelist)} species from {post1900_file}")
        except Exception as e:
            print(f"Error loading whitelist: {e}")
    if whitelist is None:
        print("Building whitelist from BioTIME...")
        whitelist = build_species_whitelist(biotime_path, min_year=1900, min_records=min_records)
        os.makedirs('csv', exist_ok=True)
        pd.DataFrame({'species': list(whitelist)}).to_csv(post1900_file, index=False)
        print(f"Built and saved whitelist of {len(whitelist)} species.")
    full_data, species_list_raw, data_taxonomy, genus_taxonomy = load_combined_data(biotime_path, rivfishtime_path, cedar_creek_paths, gbif_path, gpdd_path, min_records=min_records, use_existing_biotime=True, prefilter=True, filter_func=check_annual_series, min_time=1900, whitelist=whitelist)
    if 'DATASET' in full_data.columns and 'dataset' not in full_data.columns:
        full_data = full_data.rename(columns={'DATASET': 'dataset'})
    if 'STUDY_ID' in full_data.columns and 'study_id' not in full_data.columns:
        full_data = full_data.rename(columns={'STUDY_ID': 'study_id'})
    taxon_mapping = {}
    if os.path.exists('csv/taxon_mapping.csv') and os.path.getsize('csv/taxon_mapping.csv') > 0:
        try:
            taxon_df = pd.read_csv('csv/taxon_mapping.csv')
            taxon_mapping = dict(zip(taxon_df['species'].str.lower(), taxon_df['taxon']))
        except Exception as e:
            print(f"Error reading taxon_mapping.csv: {e}")
    return full_data, species_list_raw, data_taxonomy, genus_taxonomy, taxon_mapping

def _keep_best_group(group):
    if len(group) == 1:
        return group
    non_null_counts = group.drop(columns=['species']).notna().sum(axis=1)
    max_count = non_null_counts.max()
    best_rows = group[non_null_counts == max_count]
    return best_rows.iloc[[-1]]

def deduplicate_file(file_path, keep='last'):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return
    df = pd.read_csv(file_path)
    if 'species' not in df.columns:
        return
    if 'dataset' in df.columns:
        df_dedup = df.groupby(['species', 'dataset'], group_keys=False).apply(_keep_best_group).reset_index(drop=True)
    else:
        df_dedup = df.groupby('species', group_keys=False).apply(_keep_best_group).reset_index(drop=True)
    before = len(df)
    if len(df_dedup) < before:
        _atomic_write_csv(file_path, df_dedup)
        print(f"Deduplicated {file_path}: removed {before - len(df_dedup)} duplicates (kept best row).")

def get_failed_tasks_from_file(fail_file):
    failed = set()
    if os.path.exists(fail_file) and os.path.getsize(fail_file) > 0:
        try:
            df = pd.read_csv(fail_file)
            if 'species' in df.columns and 'dataset' in df.columns:
                for _, row in df.iterrows():
                    failed.add((str(row['species']).strip(), str(row['dataset']).strip()))
            elif 'species' in df.columns:
                for sp in df['species'].unique():
                    failed.add((str(sp).strip(), 'Unknown'))
        except Exception as e:
            print(f"Error reading failed_species.csv: {e}")
    return failed

def log_failed_task(species_name, dataset, reason):
    os.makedirs('csv', exist_ok=True)
    lock_path = FAIL_FILE + '.lock'
    _acquire_file_lock(lock_path)
    try:
        df = pd.DataFrame([{'species': species_name, 'dataset': dataset, 'reason': reason}])
        if not os.path.exists(FAIL_FILE) or os.path.getsize(FAIL_FILE) == 0:
            _atomic_write_csv(FAIL_FILE, df)
        else:
            existing = pd.read_csv(FAIL_FILE)
            if 'dataset' not in existing.columns:
                existing['dataset'] = 'Unknown'
            mask = (existing['species'] == species_name) & (existing['dataset'] == dataset)
            if mask.any():
                existing = existing[~mask]
            combined = pd.concat([existing, df], ignore_index=True)
            _atomic_write_csv(FAIL_FILE, combined)
    finally:
        _release_file_lock(lock_path)

def clean_excluded_from_results(dl_file, dev_file, fail_file, data_taxonomy, genus_taxonomy, taxon_mapping):
    for f in [dl_file, dev_file]:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        df = pd.read_csv(f)
        if 'species' not in df.columns:
            continue
        has_dataset = 'dataset' in df.columns
        rows_to_keep = []
        for idx, row in df.iterrows():
            sp = row['species']
            ds = row['dataset'] if has_dataset else 'Unknown'
            if 'taxonomic_group' in df.columns and pd.notna(row['taxonomic_group']):
                tax_group = row['taxonomic_group']
            else:
                tax_group = get_cached_taxonomic_group(sp, taxon_mapping=taxon_mapping, data_taxonomy=data_taxonomy, force_recompute=False, genus_taxonomy=genus_taxonomy)
            if tax_group in EXCLUDED_GROUPS:
                log_failed_task(sp, ds, 'taxonomic_excluded')
                continue
            rows_to_keep.append(row)
        if len(rows_to_keep) < len(df):
            new_df = pd.DataFrame(rows_to_keep)
            _atomic_write_csv(f, new_df)
            print(f"Removed {len(df)-len(rows_to_keep)} excluded species from {f}")

def check_annual_series(annual_df):
    abund = annual_df['ABUNDANCE'].values
    years = annual_df['YEAR'].values
    cleaned = abund.copy()
    if len(cleaned) < 4:
        return False, 'length<4'
    if np.var(cleaned) == 0:
        return False, 'variance=0'
    if len(np.unique(cleaned)) < 4:
        return False, 'unique<4'
    if np.sum(np.diff(cleaned) != 0) < 3:
        return False, 'changes<3'
    if len(years) != (years[-1] - years[0] + 1):
        return False, 'gaps_in_years'
    return True, ''

def filter_late_transitions(merged_df, full_data):
    full_data['TIME'] = pd.to_numeric(full_data['TIME'], errors='coerce')
    max_year_df = full_data.groupby(['SPECIES_ID', 'dataset'])['TIME'].max().reset_index()
    max_year_df.columns = ['species', 'dataset', 'max_year']
    merged_df = merged_df.merge(max_year_df, on=['species', 'dataset'], how='left')
    mask = (merged_df['transition_year'].notna()) & (merged_df['transition_year'] >= merged_df['max_year'] - 4)
    if mask.any():
        late_rows = merged_df[mask]
        for _, row in late_rows.iterrows():
            log_failed_task(row['species'], row['dataset'], 'late_transition')
        merged_df = merged_df[~mask].copy()
    merged_df = merged_df.drop(columns=['max_year'])
    return merged_df

def process_dl_sync(full_data, sp, ds, data_taxonomy):
    sp_data = full_data[full_data['SPECIES_ID'] == sp]
    if ds != 'Unknown':
        sp_data = sp_data[sp_data['dataset'] == ds]
    if len(sp_data) == 0:
        return
    try:
        process_species(sp, sp_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, species_to_plot=None, data_taxonomy=data_taxonomy, output_file=DL_FILE)
    except Exception as e:
        print(f"Error filling DL for {sp} ({ds}): {e}")

def process_dev_sync(full_data, sp, ds, data_taxonomy, genus_taxonomy, taxon_mapping):
    sp_data = full_data[full_data['SPECIES_ID'] == sp]
    if ds != 'Unknown':
        sp_data = sp_data[sp_data['dataset'] == ds]
    if len(sp_data) == 0:
        return
    try:
        result = process_dev_for_species(sp, sp_data, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy)
        if result is not None and 'error' not in result:
            result['dataset'] = ds
            result['taxonomic_group'] = get_cached_taxonomic_group(sp, taxon_mapping=taxon_mapping, data_taxonomy=data_taxonomy, force_recompute=False, genus_taxonomy=genus_taxonomy)
            append_to_csv(DEV_FILE, result)
    except Exception as e:
        print(f"Error filling DEV for {sp} ({ds}): {e}")

def ensure_species_sync(full_data, taxon_mapping, data_taxonomy, genus_taxonomy):
    dl_file = DL_FILE
    dev_file = DEV_FILE
    if not os.path.exists(dl_file) or not os.path.exists(dev_file):
        print("One or both result files missing. Cannot sync.")
        return
    deduplicate_file(dl_file, keep='last')
    deduplicate_file(dev_file, keep='last')
    df_dl = pd.read_csv(dl_file)
    df_dev = pd.read_csv(dev_file)
    dl_tasks = set(zip(df_dl['species'], df_dl['dataset'])) if 'dataset' in df_dl.columns else set(zip(df_dl['species'], ['Unknown']*len(df_dl)))
    dev_tasks = set(zip(df_dev['species'], df_dev['dataset'])) if 'dataset' in df_dev.columns else set(zip(df_dev['species'], ['Unknown']*len(df_dev)))
    missing_in_dl = dev_tasks - dl_tasks
    missing_in_dev = dl_tasks - dev_tasks
    n_processes = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_processes) as pool:
        if missing_in_dl:
            print(f"Missing in DL: {len(missing_in_dl)} tasks. Running DL-EWS in parallel...")
            pool.starmap(process_dl_sync, [(full_data, sp, ds, data_taxonomy) for (sp, ds) in missing_in_dl])
        if missing_in_dev:
            print(f"Missing in DEV: {len(missing_in_dev)} tasks. Running DEV in parallel...")
            pool.starmap(process_dev_sync, [(full_data, sp, ds, data_taxonomy, genus_taxonomy, taxon_mapping) for (sp, ds) in missing_in_dev])
    deduplicate_file(dl_file, keep='last')
    deduplicate_file(dev_file, keep='last')
    print("Species sync completed.")

def ensure_dev_json_for_species(full_data, data_taxonomy, genus_taxonomy):
    consolidate_dev_series()
    if not os.path.exists(DEV_FILE) or os.path.getsize(DEV_FILE) == 0:
        print("DEV file not found, skipping JSON check.")
        return
    df_dev = pd.read_csv(DEV_FILE)
    json_species = set()
    if os.path.exists('csv/dev_series.json') and os.path.getsize('csv/dev_series.json') > 0:
        with open('csv/dev_series.json', 'r') as f:
            json_data = json.load(f)
            json_species = set(json_data.keys())
    backup_path = DEV_FILE + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(DEV_FILE, backup_path)
        print(f"Backup of DEV file saved to {backup_path}")
    if 'species' in df_dev.columns:
        initial_len = len(df_dev)
        df_dev_filtered = df_dev[df_dev['species'].isin(json_species)]
        removed = initial_len - len(df_dev_filtered)
        if removed > 0:
            _atomic_write_csv(DEV_FILE, df_dev_filtered)
            print(f"Removed {removed} species without JSON from DEV file. Kept {len(df_dev_filtered)} species.")
        else:
            print("All DEV species have corresponding JSON.")
    else:
        print("DEV file missing 'species' column, cannot filter.")

def consolidate_dev_series():
    json_files = [f for f in os.listdir('csv/partial') if f.startswith('dev_series_') and f.endswith('.jsonl')]
    if not json_files:
        print("No DEV series JSONL files found.")
        return
    all_dev = {}
    for fname in json_files:
        path = os.path.join('csv/partial', fname)
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                for sp, dev_series in data.items():
                    if len(dev_series) == 0:
                        continue
                    last_idx = dev_series[-1][0]
                    if sp not in all_dev:
                        all_dev[sp] = (dev_series, last_idx)
                    else:
                        existing_dev, existing_last = all_dev[sp]
                        if last_idx > existing_last:
                            all_dev[sp] = (dev_series, last_idx)
    final_dev = {sp: dev for sp, (dev, _) in all_dev.items()}
    if final_dev:
        with open('csv/dev_series.json', 'w') as f:
            json.dump(final_dev, f, indent=None, separators=(',',':'))
        print(f"Consolidated DEV series saved to csv/dev_series.json ({len(final_dev)} species).")
    else:
        print("No valid DEV series to consolidate.")

def merge_results(dl_file=DL_FILE, dev_file=DEV_FILE, out_file='csv/bifurcation_results.csv', short_file='csv/bifurcation_results_short.csv'):
    if not os.path.exists(dl_file) or not os.path.exists(dev_file):
        print("Both DL and DEV files are required. Cannot merge.")
        return
    df_dl = pd.read_csv(dl_file)
    df_dev = pd.read_csv(dev_file)
    df_dl = df_dl[df_dl['transition_year'].notna() & (df_dl['transition_year'] != '')]
    merged = pd.merge(df_dl, df_dev, on=['species', 'dataset', 'taxonomic_group'], how='outer')
    shared_cols = ['species', 'dataset', 'taxonomic_group']
    dl_cols = ['tipping_point_type', 'raw_tipping_point_type', 'ews_metrics', 'transition_year', 'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
    dev_cols = ['early_warning', 'final_re', 'final_im', 'final_abs', 'bifurcation_type', 'time_to_transition']
    all_cols = shared_cols + dl_cols + dev_cols
    existing_cols = [col for col in all_cols if col in merged.columns]
    merged = merged[existing_cols]
    return merged

def run_phase(only_gbif, n_jobs, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, dl_tasks, dev_tasks, failed_tasks):
    queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()
    consumers = []
    for _ in range(n_jobs):
        p = mp.Process(target=consumer_worker, args=(queue, stop_event, data_taxonomy, genus_taxonomy))
        p.start()
        consumers.append(p)
    try:
        producer(get_species_datasets(full_data), full_data, taxon_mapping, data_taxonomy, genus_taxonomy, queue, stop_event, dl_tasks, dev_tasks, failed_tasks, n_jobs, only_gbif=only_gbif)
    except KeyboardInterrupt:
        stop_event.set()
    for p in consumers:
        p.join()

def main(min_records=20, n_jobs=-1, species_batch_size=1000, per_species_timeout=1800):
    try:
        deduplicate_file(DL_FILE, keep='last')
        deduplicate_file(DEV_FILE, keep='last')
        if n_jobs == -1:
            n_jobs = max(1, mp.cpu_count() - 1)
        full_data, species_list_raw, data_taxonomy, genus_taxonomy, taxon_mapping = setup_data(min_records)
        clean_excluded_from_results(DL_FILE, DEV_FILE, FAIL_FILE, data_taxonomy, genus_taxonomy, taxon_mapping)
        dl_tasks = get_tasks_from_file(DL_FILE)
        dev_tasks = get_tasks_from_file(DEV_FILE)
        failed_tasks = get_failed_tasks_from_file(FAIL_FILE)
        print(f"Loaded {len(species_list_raw)} raw species. Already processed DL tasks: {len(dl_tasks)}. Already processed DEV tasks: {len(dev_tasks)}. Failed tasks: {len(failed_tasks)}.")
        print(f"Interleaved producer-consumer with {n_jobs} consumers (timeout={per_species_timeout}s).")
        print("Phase 1: Processing non-GBIF datasets...")
        run_phase(False, n_jobs, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, dl_tasks, dev_tasks, failed_tasks)
        print("Phase 1 completed.")
        merged_df_non_gbif = merge_results()
        if merged_df_non_gbif is not None and not merged_df_non_gbif.empty:
            if 'dataset' in merged_df_non_gbif.columns:
                merged_df_non_gbif = merged_df_non_gbif[merged_df_non_gbif['dataset'] != 'GBIF']
            if not merged_df_non_gbif.empty:
                merged_df_non_gbif = filter_late_transitions(merged_df_non_gbif, full_data)
                print("Generating summary figure from non-GBIF results...")
                create_summary_figure(merged_df_non_gbif)
                print("Summary figure generated.")
            else:
                print("No non-GBIF results to generate figure.")
        else:
            print("No merged results to generate figure.")
        print("Phase 2: Processing GBIF datasets...")
        run_phase(True, n_jobs, full_data, taxon_mapping, data_taxonomy, genus_taxonomy, dl_tasks, dev_tasks, failed_tasks)
        print("Phase 2 completed.")
#        ensure_species_sync(full_data, taxon_mapping, data_taxonomy, genus_taxonomy)
#        print("Species sync completed.")
        ensure_dev_json_for_species(full_data, data_taxonomy, genus_taxonomy)
        print("DEV JSON sync completed.")
        merged_df_full = merge_results()
        if merged_df_full is not None and not merged_df_full.empty:
            merged_df_gbif = merged_df_full[merged_df_full['dataset'] == 'GBIF'].copy()
            if not merged_df_gbif.empty:
                merged_df_gbif = filter_late_transitions(merged_df_gbif, full_data)
            if merged_df_non_gbif is not None and not merged_df_non_gbif.empty:
                merged_df_final = pd.concat([merged_df_non_gbif, merged_df_gbif], ignore_index=True)
            else:
                merged_df_final = merged_df_gbif
            out_file = 'csv/bifurcation_results.csv'
            short_file = 'csv/bifurcation_results_short.csv'
            merged_df_final.to_csv(out_file, index=False)
            shutil.copyfile(out_file, short_file)
            print(f"Merged results saved to {out_file} and {short_file}")
        else:
            print("No merged results to save.")
        print("Merge completed.")
        results_dl = []
        results_dev = []
        if os.path.exists(DL_FILE):
            results_dl = pd.read_csv(DL_FILE).to_dict('records')
            print(f"DL-EWS results saved: {len(results_dl)} species.")
        if os.path.exists(DEV_FILE):
            results_dev = pd.read_csv(DEV_FILE).to_dict('records')
            print(f"DEV results saved: {len(results_dev)} species.")
        if results_dl and results_dev:
            print("\nComputing agreement between DL-EWS and DEV...")
            compute_agreement(results_dl, results_dev)
            print("Agreement computed.")
        return results_dl, results_dev
    except Exception as e:
        print("ERROR in main orchestrator:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
