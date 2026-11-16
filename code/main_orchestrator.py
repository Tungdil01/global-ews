#!/usr/bin/env python
# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import re, time, shutil, signal, tempfile, json, hashlib
import multiprocessing as mp
import queue as queue_module
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import cohen_kappa_score, accuracy_score
min_records=20
probability_threshold = 0.5
late_transition_window = 0.5
from main_dl import load_combined_data, process_species, get_cached_taxonomic_group, EXCLUDED_GROUPS, EXCLUDED_SPECIES_LABELS, BANDING_CODE_TO_NAME, create_summary_figure, update_taxonomic_groups_from_mapping, ensure_all_species_classified, _classify_species_no_cache, script_dir
classifier_base = os.path.join(script_dir, '..', 'saved_classifiers', 'bury_pnas_21')
root_path_500 = os.path.join(classifier_base, 'len500')
root_path_1500 = os.path.join(classifier_base, 'len1500')
classifier_names_500 = [name for name in os.listdir(root_path_500) if name.endswith('.pkl')]
classifier_paths_500 = [os.path.join(root_path_500, name) for name in classifier_names_500[:20]]
classifier_names_1500 = [name for name in os.listdir(root_path_1500) if name.endswith('.pkl')]
classifier_paths_1500 = [os.path.join(root_path_1500, name) for name in classifier_names_1500[:20]]
bifurcation_map = {0: 'fold bifurcation', 1: 'Hopf bifurcation', 2: 'transcritical bifurcation', 3: 'no-bifurcation'}
DATASET_PRIORITY = {'BioTIME': 0, 'RivFishTIME': 1, 'GPDD': 2, 'NPS': 3, 'LPD': 4, 'GLBA': 5, 'VRData': 6, 'CedarCreek': 7, 'TSX': 8}
CP_METHODS = ['dynp', 'pelt', 'binseg', 'window']
DL_FILE = 'csv/bifurcation_results_dl_ews.csv'
FAIL_FILE = 'csv/failed_species.csv'
FILTER_ERROR_LOG = 'csv/filter_errors.log'
IUCN_FILE = 'csv/iucn_species.csv'
IUCN_EXCLUDED_STATUSES = {'NOT ASSESSED', 'DATA_DEFICIENT'}

def _atomic_write_csv(file_path, df):
    fd, tmp_path = tempfile.mkstemp(suffix='.csv', dir=os.path.dirname(file_path) or '.')
    try:
        with os.fdopen(fd, 'w') as f:
            df.to_csv(f, index=False)
        os.replace(tmp_path, file_path)
    except Exception:
        os.remove(tmp_path)
        raise

def _acquire_file_lock(lock_path, timeout=900):
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

def remove_failed_task(species_name, dataset, timeseries_id, fail_file=FAIL_FILE):
    if not os.path.exists(fail_file) or os.path.getsize(fail_file) == 0:
        return
    lock_path = fail_file + '.lock'
    _acquire_file_lock(lock_path)
    try:
        df = pd.read_csv(fail_file)
        if 'species' not in df.columns:
            return
        if 'dataset' not in df.columns:
            df['dataset'] = 'Unknown'
        if 'timeseries_id' not in df.columns:
            df['timeseries_id'] = 'Unknown'
        mask = (df['species'].astype(str).str.strip() == str(species_name).strip()) & (df['dataset'].astype(str).str.strip() == str(dataset).strip()) & (df['timeseries_id'].astype(str).str.strip() == str(timeseries_id).strip())
        if not mask.any():
            return
        df_kept = df[~mask].reset_index(drop=True)
        _atomic_write_csv(fail_file, df_kept)
    finally:
        _release_file_lock(lock_path)

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
            if 'raw_tipping_point_type' not in existing.columns and 'raw_tipping_point_type' in df.columns:
                df = df.drop(columns=['raw_tipping_point_type'])
            has_ts = ('timeseries_id' in existing.columns) and ('timeseries_id' in df.columns)
            if has_ts:
                mask = (existing['species'] == data_dict['species']) & (existing['dataset'] == data_dict['dataset']) & (existing['timeseries_id'].astype(str) == str(data_dict['timeseries_id']))
            elif ('dataset' in existing.columns) and ('dataset' in df.columns):
                mask = (existing['species'] == data_dict['species']) & (existing['dataset'] == data_dict['dataset'])
            else:
                mask = existing['species'] == data_dict['species']
            if mask.any():
                existing = existing[~mask]
            combined = pd.concat([existing, df], ignore_index=True)
            _atomic_write_csv(file_path, combined)
    finally:
        _release_file_lock(lock_path)
    if file_path == DL_FILE and 'species' in data_dict and 'dataset' in data_dict and 'timeseries_id' in data_dict:
        remove_failed_task(data_dict['species'], data_dict['dataset'], data_dict['timeseries_id'])

def process_species_with_retries(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, output_file=DL_FILE):
    if species_data is None or len(species_data) == 0:
        return None, dataset
    candidates = []
    last_error = None
    for method in [None] + CP_METHODS:
        try:
            result = process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, late_transition_window, species_to_plot=None, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy, output_file=output_file, cp_method=method)
        except Exception as e:
            print(f"DL ERROR for {species_name} on {dataset}/{timeseries_id} (method={method}): {e}")
            result = {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': str(e)}
        if result is None:
            continue
        if result.get('error') is not None:
            last_error = result
            continue
        result['timeseries_id'] = timeseries_id
        prob = float(result.get('best_probability', 0.0))
        candidates.append((prob, result))
    if not candidates:
        if last_error is not None:
            return last_error, dataset
        return None, dataset
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], dataset

def process_both_analyses(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl=True, run_dev=False, output_dl=DL_FILE, output_dev=None):
    result_dl = None
    result_dev = None
    if run_dl:
        methods = ['dynp', 'pelt', 'binseg', 'window', 'bottomup']
        last_result = None
        for method in methods:
            try:
                result = process_species(species_name, species_data, classifier_paths_500, classifier_paths_1500, bifurcation_map, late_transition_window, species_to_plot=None, data_taxonomy=data_taxonomy, genus_taxonomy=genus_taxonomy, output_file=output_dl, cp_method=method)
            except Exception as e:
                print(f"DL-EWS ERROR for {species_name} on {dataset}/{timeseries_id} (method={method}): {e}")
                result = {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': str(e)}
            if result is None:
                continue
            last_result = result
            if 'error' not in result:
                prob = float(result.get('best_probability', 0.0))
                ty = result.get('transition_year')
                if prob > 0 and ty is not None:
                    result_dl = result
                    break
        if result_dl is None:
            result_dl = last_result
        if result_dl is not None:
            result_dl['timeseries_id'] = timeseries_id
            if 'error' not in result_dl:
                prob = float(result_dl.get('best_probability', 0.0))
                ty = result_dl.get('transition_year')
                if prob > 0 and ty is not None:
                    data_to_save = {k: v for k, v in result_dl.items() if k != 'ts'}
                    append_to_csv(output_dl, data_to_save)
                else:
                    log_failed_task(species_name, dataset, timeseries_id, 'zero_probability')
    return result_dl, result_dev

def _analysis_target(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl, run_dev, conn):
    dl_res, dev_res = process_both_analyses(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl=run_dl, run_dev=run_dev)
    conn.send((dl_res, dev_res))
    conn.close()

def run_analysis_with_timeout(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl=True, run_dev=False, timeout=900):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_analysis_target, args=(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl, run_dev, child_conn))
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
            print(f"Timeout ({timeout}s) exceeded for {species_name} on {dataset}/{timeseries_id}, skipping.")
            if run_dl and run_dev:
                return None, {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': 'timeout'}
            elif run_dl:
                return {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': 'timeout'}, None
            elif run_dev:
                return None, {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': 'timeout'}
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
    return None, {'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'error': 'child_error'}

def consumer_worker(q, stop_event, data_taxonomy, genus_taxonomy, per_species_timeout):
    processed_count = 0
    timeout_seconds = per_species_timeout
    while not stop_event.is_set():
        try:
            item = q.get(timeout=0.1)
        except queue_module.Empty:
            continue
        if item is None:
            break
        species_name, dataset, timeseries_id, species_data, run_dl, run_dev = item
        dl_res, dev_res = run_analysis_with_timeout(species_name, dataset, timeseries_id, species_data, data_taxonomy, genus_taxonomy, run_dl=run_dl, run_dev=run_dev, timeout=timeout_seconds)
        if run_dl:
            if isinstance(dl_res, dict) and 'error' in dl_res:
                error_val = dl_res.get('error')
                reason = 'timeout_dl' if error_val == 'timeout' else str(error_val)
            elif dl_res is None:
                reason = 'error'
            else:
                reason = None
        else:
            reason = None
        if reason is not None:
            log_failed_task(species_name, dataset, timeseries_id, reason)
        processed_count += 1
    print(f"Consumer processed {processed_count} tasks.")

def _norm_key(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def get_species_datasets(datasets_dict, dl_tasks, failed_tasks):
    tasks = []
    dl_tasks_norm = {(_norm_key(s), _norm_key(d), _norm_key(t)) for s, d, t in dl_tasks}
    failed_tasks_norm = {(_norm_key(s), _norm_key(d), _norm_key(t)) for s, d, t in failed_tasks}
    for ds_name, df in datasets_dict.items():
        if str(ds_name).lower() == 'gbif' or len(df) == 0:
            continue
        df = df.copy()
        df['TIMESERIES_ID'] = df['TIMESERIES_ID'].astype(str).str.strip()
        for (sp, ts), ts_df in df.groupby(['SPECIES_ID', 'TIMESERIES_ID']):
            sp_str = str(sp).strip()
            if sp_str == 'Other':
                continue
            ts_str = str(ts).strip()
            key_norm = (_norm_key(sp_str), _norm_key(ds_name), _norm_key(ts_str))
            if key_norm in dl_tasks_norm or key_norm in failed_tasks_norm:
                continue
            tasks.append((sp_str, ds_name, ts_str, ts_df))
    return tasks

def get_tasks_from_file(file_path):
    tasks = set()
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df = pd.read_csv(file_path)
        if 'species' in df.columns and 'dataset' in df.columns and 'timeseries_id' in df.columns:
            for _, row in df.iterrows():
                tasks.add((str(row['species']).strip(), str(row['dataset']).strip(), str(row['timeseries_id']).strip()))
        elif 'species' in df.columns and 'dataset' in df.columns:
            for _, row in df.iterrows():
                tasks.add((str(row['species']).strip(), str(row['dataset']).strip(), 'Unknown'))
        elif 'species' in df.columns:
            for sp in df['species'].unique():
                tasks.add((str(sp).strip(), 'Unknown', 'Unknown'))
    return tasks

def producer(tasks, queue, stop_event, num_consumers):
    passed_count = 0
    try:
        for sp, ds, ts, sp_data in tqdm(tasks, desc="Producer: queueing tasks"):
            if stop_event.is_set():
                break
            while not stop_event.is_set():
                try:
                    queue.put((sp, ds, ts, sp_data, True, False), timeout=0.1)
                    break
                except queue_module.Full:
                    time.sleep(0.5)
            passed_count += 1
    finally:
        print(f"Producer: {passed_count} tasks queued.")
        for _ in range(num_consumers):
            queue.put(None)

def setup_data(min_records):
    datasets_dict, species_list_raw, data_taxonomy, genus_taxonomy = load_combined_data(min_records=min_records, prefilter=True, filter_func=check_annual_series, min_time=1900, whitelist=None)
    for ds_name in list(datasets_dict.keys()):
        df = datasets_dict[ds_name]
        if 'DATASET' in df.columns and 'dataset' not in df.columns:
            df = df.rename(columns={'DATASET': 'dataset'})
        if 'STUDY_ID' in df.columns and 'study_id' not in df.columns:
            df = df.rename(columns={'STUDY_ID': 'study_id'})
        datasets_dict[ds_name] = df
    return datasets_dict, species_list_raw, data_taxonomy, genus_taxonomy

def get_failed_tasks_from_file(fail_file):
    failed = set()
    if os.path.exists(fail_file) and os.path.getsize(fail_file) > 0:
        try:
            df = pd.read_csv(fail_file)
            if 'species' in df.columns and 'dataset' in df.columns and 'timeseries_id' in df.columns:
                for _, row in df.iterrows():
                    failed.add((str(row['species']).strip(), str(row['dataset']).strip(), str(row['timeseries_id']).strip()))
            elif 'species' in df.columns and 'dataset' in df.columns:
                for _, row in df.iterrows():
                    failed.add((str(row['species']).strip(), str(row['dataset']).strip(), 'Unknown'))
            elif 'species' in df.columns:
                for sp in df['species'].unique():
                    failed.add((str(sp).strip(), 'Unknown', 'Unknown'))
        except Exception as e:
            print(f"Error reading failed_species.csv: {e}")
    return failed

def log_failed_task(species_name, dataset, timeseries_id, reason):
    os.makedirs('csv', exist_ok=True)
    lock_path = FAIL_FILE + '.lock'
    _acquire_file_lock(lock_path)
    try:
        df = pd.DataFrame([{'species': species_name, 'dataset': dataset, 'timeseries_id': timeseries_id, 'reason': reason}])
        if not os.path.exists(FAIL_FILE) or os.path.getsize(FAIL_FILE) == 0:
            _atomic_write_csv(FAIL_FILE, df)
        else:
            existing = pd.read_csv(FAIL_FILE)
            if 'dataset' not in existing.columns:
                existing['dataset'] = 'Unknown'
            if 'timeseries_id' not in existing.columns:
                existing['timeseries_id'] = 'Unknown'
            if 'reason' not in existing.columns:
                existing['reason'] = ''
            mask = (existing['species'] == species_name) & (existing['dataset'] == dataset) & (existing['timeseries_id'].astype(str) == str(timeseries_id))
            if reason == 'timeout_dl' and mask.any():
                prev_reason = str(existing.loc[mask, 'reason'].iloc[0]).strip()
                if prev_reason == 'timeout_dl':
                    df['reason'] = 'timeout'
            if mask.any():
                existing = existing[~mask]
            combined = pd.concat([existing, df], ignore_index=True)
            _atomic_write_csv(FAIL_FILE, combined)
    finally:
        _release_file_lock(lock_path)

def clean_excluded_from_results(dl_file, data_taxonomy, genus_taxonomy):
    for f in [dl_file]:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        df = pd.read_csv(f)
        if 'species' not in df.columns:
            continue
        has_dataset = 'dataset' in df.columns
        has_ts = 'timeseries_id' in df.columns
        rows_to_keep = []
        for idx, row in df.iterrows():
            sp = row['species']
            ds = row['dataset'] if has_dataset else 'Unknown'
            ts = row['timeseries_id'] if has_ts else 'Unknown'
            if 'taxonomic_group' in df.columns and pd.notna(row['taxonomic_group']):
                tax_group = row['taxonomic_group']
            else:
                tax_group = get_cached_taxonomic_group(sp, data_taxonomy=data_taxonomy, force_recompute=False, genus_taxonomy=genus_taxonomy)
            if tax_group in EXCLUDED_GROUPS or str(sp).strip() in EXCLUDED_SPECIES_LABELS:
                log_failed_task(sp, ds, ts, 'taxonomic_excluded')
                continue
            rows_to_keep.append(row)
        if len(rows_to_keep) < len(df):
            new_df = pd.DataFrame(rows_to_keep) if rows_to_keep else df.iloc[0:0]
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

def move_zero_prob_to_fail(dl_file, fail_file, reason='zero_probability'):
    if not os.path.exists(dl_file) or os.path.getsize(dl_file) == 0:
        print(f"{dl_file} missing or empty. Skipping zero-probability move.")
        return
    df = pd.read_csv(dl_file)
    if 'best_probability' not in df.columns or 'species' not in df.columns or 'dataset' not in df.columns:
        return
    df['best_probability'] = pd.to_numeric(df['best_probability'], errors='coerce').fillna(0.0)
    has_ts = 'timeseries_id' in df.columns
    if has_ts:
        df['timeseries_id'] = df['timeseries_id'].astype(str).str.strip()
    df['species'] = df['species'].astype(str).str.strip()
    df['dataset'] = df['dataset'].astype(str).str.strip()
    success_mask = df['best_probability'] > 0
    if 'transition_year' in df.columns:
        success_mask = success_mask & df['transition_year'].notna()
    df_kept = df[success_mask].reset_index(drop=True)
    df_fail_new = df[~success_mask].reset_index(drop=True)
    _atomic_write_csv(dl_file, df_kept)
    if len(df_fail_new) > 0:
        if has_ts:
            df_fail_new = df_fail_new[['species', 'dataset', 'timeseries_id']].copy()
        else:
            df_fail_new = df_fail_new[['species', 'dataset']].copy()
        df_fail_new['reason'] = reason
        if os.path.exists(fail_file) and os.path.getsize(fail_file) > 0:
            df_fail = pd.read_csv(fail_file)
            combined = pd.concat([df_fail, df_fail_new], ignore_index=True)
        else:
            combined = df_fail_new
        _atomic_write_csv(fail_file, combined)
    print(f"Moved {len(df_fail_new)} entries from {dl_file} to {fail_file}, kept {len(df_kept)} successful results.")

def merge_results(dl_file=DL_FILE, out_file='csv/bifurcation_results.csv', short_file='csv/bifurcation_results_short.csv'):
    if not os.path.exists(dl_file):
        print("DL file missing. Cannot merge.")
        return
    df_dl = pd.read_csv(dl_file)
    df_dl = df_dl[df_dl['transition_year'].notna()]
    dl_cols = ['species', 'dataset', 'timeseries_id', 'taxonomic_group', 'tipping_point_type', 'raw_tipping_point_type', 'ews_metrics', 'transition_year', 'change_points', 'smoothing_avg', 'current_abundance', 'max_abundance', 'best_probability']
    existing_cols = [col for col in dl_cols if col in df_dl.columns]
    merged = df_dl[existing_cols].copy()
    return merged

def run_phase(tasks, n_jobs, data_taxonomy, genus_taxonomy, per_species_timeout):
    queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()
    consumers = []
    for _ in range(n_jobs):
        p = mp.Process(target=consumer_worker, args=(queue, stop_event, data_taxonomy, genus_taxonomy, per_species_timeout))
        p.start()
        consumers.append(p)
    try:
        producer(tasks, queue, stop_event, n_jobs)
    except KeyboardInterrupt:
        stop_event.set()
    for p in consumers:
        p.join()

def deduplicate_backups(backup_dir='csv/backups', base_prefix='bifurcation_results_dl_ews'):
    if not os.path.isdir(backup_dir):
        return
    files = sorted([f for f in os.listdir(backup_dir) if f.startswith(base_prefix + '_') and f.endswith('.csv')], reverse=True)
    if len(files) <= 1:
        return
    hash_groups = {}
    for fname in files:
        fpath = os.path.join(backup_dir, fname)
        if os.path.getsize(fpath) == 0:
            continue
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        hash_groups.setdefault(h, []).append(fname)
    removed = 0
    for h, group in hash_groups.items():
        for fname in group[1:]:
            os.remove(os.path.join(backup_dir, fname))
            removed += 1
    if removed > 0:
        print('Removed ' + str(removed) + ' duplicate backups from ' + backup_dir + '.')

def backup_dl_file(dl_file, backup_dir='csv/backups'):
    if not os.path.exists(dl_file) or os.path.getsize(dl_file) == 0:
        return None
    os.makedirs(backup_dir, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S_') + '{:06d}'.format(now.microsecond)
    base = os.path.splitext(os.path.basename(dl_file))[0]
    ext = os.path.splitext(dl_file)[1]
    backup_path = os.path.join(backup_dir, base + '_' + timestamp + ext)
    shutil.copy2(dl_file, backup_path)
    print('Backup saved to ' + backup_path)
    return backup_path

def restore_missing_rows_from_backups(dl_file, backup_dir='csv/backups', fail_file=FAIL_FILE):
    if not os.path.exists(dl_file) or os.path.getsize(dl_file) == 0:
        return
    if not os.path.isdir(backup_dir):
        return
    base = os.path.splitext(os.path.basename(dl_file))[0]
    ext = os.path.splitext(dl_file)[1]
    backup_files = sorted([f for f in os.listdir(backup_dir) if f.startswith(base + '_') and f.endswith(ext)], reverse=True)
    if not backup_files:
        return
    df_current = pd.read_csv(dl_file)
    required = ['species', 'dataset', 'timeseries_id', 'best_probability']
    missing_cols = [c for c in required if c not in df_current.columns]
    if missing_cols:
        print('Skipping restore, missing columns in ' + dl_file + ': ' + str(missing_cols))
        return
    permanent_reasons = {'data_missing', 'taxonomic_excluded', 'late_transition', 'length<4', 'variance=0', 'unique<4', 'changes<3', 'gaps_in_years', 'timeout'}
    failed_keys = set()
    if os.path.exists(fail_file) and os.path.getsize(fail_file) > 0:
        df_fail = pd.read_csv(fail_file)
        if 'species' in df_fail.columns and 'dataset' in df_fail.columns and 'timeseries_id' in df_fail.columns and 'reason' in df_fail.columns:
            for _, frow in df_fail.iterrows():
                reason = str(frow['reason']).strip()
                if reason in permanent_reasons:
                    failed_keys.add((_norm_key(frow['species']), _norm_key(frow['dataset']), _norm_key(frow['timeseries_id'])))
    current_keys = {(_norm_key(s), _norm_key(d), _norm_key(t)) for s, d, t in zip(df_current['species'].astype(str).str.strip(), df_current['dataset'].astype(str).str.strip(), df_current['timeseries_id'].astype(str).str.strip())}
    new_rows = []
    skipped_zero = 0
    skipped_excluded = 0
    skipped_failed = 0
    skipped_labels = 0
    for fname in backup_files:
        fpath = os.path.join(backup_dir, fname)
        if os.path.getsize(fpath) == 0:
            continue
        df_b = pd.read_csv(fpath)
        missing_b = [c for c in required if c not in df_b.columns]
        if missing_b:
            continue
        for _, row in df_b.iterrows():
            k = (_norm_key(row['species']), _norm_key(row['dataset']), _norm_key(row['timeseries_id']))
            if k in current_keys:
                continue
            if k in failed_keys:
                skipped_failed += 1
                continue
            if float(row['best_probability']) == 0.0:
                skipped_zero += 1
                continue
            if str(row['species']).strip() in EXCLUDED_SPECIES_LABELS:
                skipped_labels += 1
                continue
            if 'taxonomic_group' in df_b.columns and pd.notna(row.get('taxonomic_group')) and str(row['taxonomic_group']).strip() in EXCLUDED_GROUPS:
                skipped_excluded += 1
                continue
            new_rows.append(row)
            current_keys.add(k)
    if skipped_zero > 0 or skipped_excluded > 0 or skipped_failed > 0 or skipped_labels > 0:
        print('Skipped ' + str(skipped_zero) + ' zero-probability rows, ' + str(skipped_excluded) + ' excluded-taxonomy rows, ' + str(skipped_labels) + ' excluded-label rows, and ' + str(skipped_failed) + ' known-failed rows during restore.')
    if not new_rows:
        print('No missing rows to restore from backups.')
        return
    df_new = pd.DataFrame(new_rows)
    for col in df_current.columns:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[df_current.columns]
    combined = pd.concat([df_current, df_new], ignore_index=True)
    _atomic_write_csv(dl_file, combined)
    print('Restored ' + str(len(new_rows)) + ' missing rows to ' + dl_file)

def clean_failed_series_from_results(dl_file, datasets_dict, late_transition_window):
    if not os.path.exists(dl_file) or os.path.getsize(dl_file) == 0:
        return
    df = pd.read_csv(dl_file)
    if 'species' not in df.columns or 'dataset' not in df.columns or 'timeseries_id' not in df.columns:
        return
    df['species'] = df['species'].astype(str).str.strip()
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['timeseries_id'] = df['timeseries_id'].astype(str).str.strip()
    segments = {}
    for ds_name, ds_df in datasets_dict.items():
        if len(ds_df) == 0:
            continue
        ds_df = ds_df.copy()
        ds_df['TIME'] = pd.to_numeric(ds_df['TIME'], errors='coerce')
        ds_df['TIMESERIES_ID'] = ds_df['TIMESERIES_ID'].astype(str).str.strip()
        if 'dataset' not in ds_df.columns:
            ds_df['dataset'] = ds_name
        ds_df['dataset'] = ds_df['dataset'].astype(str).str.strip()
        for (sp, ds_val, ts), grp in ds_df.groupby(['SPECIES_ID', 'dataset', 'TIMESERIES_ID']):
            segments[(str(sp).strip(), str(ds_val).strip(), str(ts).strip())] = grp
    rows_to_keep = []
    removed_stale = 0
    removed_filter = 0
    removed_late = 0
    for _, row in df.iterrows():
        key = (row['species'], row['dataset'], row['timeseries_id'])
        grp = segments.get(key)
        if grp is None or len(grp) == 0:
            removed_stale += 1
            continue
        annual = grp.groupby(np.floor(grp['TIME']).astype(int))['ABUNDANCE'].mean().reset_index()
        annual.columns = ['YEAR', 'ABUNDANCE']
        annual['ABUNDANCE'] = annual['ABUNDANCE'].astype(float)
        annual = annual.dropna(subset=['ABUNDANCE'])
        passed, reason = check_annual_series(annual)
        if not passed:
            log_failed_task(row['species'], row['dataset'], row['timeseries_id'], reason)
            removed_filter += 1
            continue
        max_year = float(grp['TIME'].max())
        transition_year = row.get('transition_year')
        if pd.notna(transition_year) and float(transition_year) >= max_year - late_transition_window:
            log_failed_task(row['species'], row['dataset'], row['timeseries_id'], 'late_transition')
            removed_late += 1
            continue
        rows_to_keep.append(row)
    total_removed = removed_stale + removed_filter + removed_late
    if total_removed > 0:
        new_df = pd.DataFrame(rows_to_keep) if rows_to_keep else df.iloc[0:0]
        _atomic_write_csv(dl_file, new_df)
        print(f"Removed {total_removed} stale rows from {dl_file}: stale_timeseries={removed_stale}, filter_failed={removed_filter}, late_transition={removed_late}.")

def reconcile_fail_file_with_dl(dl_file, fail_file):
    if not os.path.exists(dl_file) or os.path.getsize(dl_file) == 0:
        return
    if not os.path.exists(fail_file) or os.path.getsize(fail_file) == 0:
        return
    df_dl = pd.read_csv(dl_file)
    if 'species' not in df_dl.columns or 'dataset' not in df_dl.columns or 'timeseries_id' not in df_dl.columns:
        return
    dl_keys = {(_norm_key(s), _norm_key(d), _norm_key(t)) for s, d, t in zip(df_dl['species'].astype(str).str.strip(), df_dl['dataset'].astype(str).str.strip(), df_dl['timeseries_id'].astype(str).str.strip())}
    lock_path = fail_file + '.lock'
    _acquire_file_lock(lock_path)
    try:
        df_fail = pd.read_csv(fail_file)
        if 'species' not in df_fail.columns or 'dataset' not in df_fail.columns or 'timeseries_id' not in df_fail.columns or 'reason' not in df_fail.columns:
            return
        transient_reasons = {'error', 'timeout_dl', 'timeout', 'child_error'}
        species_arr = df_fail['species'].astype(str).str.strip()
        dataset_arr = df_fail['dataset'].astype(str).str.strip()
        ts_arr = df_fail['timeseries_id'].astype(str).str.strip()
        reason_arr = df_fail['reason'].astype(str).str.strip()
        keep_mask = np.array([not ((_norm_key(s), _norm_key(d), _norm_key(t)) in dl_keys and r in transient_reasons) for s, d, t, r in zip(species_arr, dataset_arr, ts_arr, reason_arr)], dtype=bool)
        removed = int((~keep_mask).sum())
        if removed > 0:
            df_fail_kept = df_fail[keep_mask].reset_index(drop=True)
            _atomic_write_csv(fail_file, df_fail_kept)
            print('Reconciled ' + fail_file + ': removed ' + str(removed) + ' stale transient-failure entries already present in ' + dl_file + '.')
    finally:
        _release_file_lock(lock_path)

DEV_FILE = 'csv/bifurcation_results_dev.csv'

def build_short_outputs(dl_file, dev_file, large_file, short_file, min_records, datasets_dict):
    import build_short as bs
    df_dl, df_dev = bs.read_data(dl_file, dev_file)
    if df_dl is None:
        print(f"{dl_file} missing; skipping short outputs.")
        return
    column_order = df_dl.columns.tolist()
    dataset_filter = 'non_gbif'
    df_dl_f, df_dev_f = bs.filter_gbif(df_dl, df_dev, dataset_filter)
    merged = bs.merge_dfs(df_dl_f, df_dev_f)
    merged = bs.combine_taxonomic(merged)
    merged = bs.convert_insecta_to_hexapoda(merged)
    bs.build_output(merged, datasets_dict, large_file, dataset_filter, include_dev=False, remove_late=False, column_order=column_order)
    bs.build_output(merged, datasets_dict, short_file, dataset_filter, include_dev=True, remove_late=True, window_years=late_transition_window, min_best_probability=probability_threshold, column_order=column_order)

def clean_excluded_from_fail(fail_file, data_taxonomy, genus_taxonomy):
    if not os.path.exists(fail_file) or os.path.getsize(fail_file) == 0:
        return
    df = pd.read_csv(fail_file)
    if 'species' not in df.columns or 'reason' not in df.columns:
        return
    if 'dataset' not in df.columns:
        df['dataset'] = 'Unknown'
    if 'timeseries_id' not in df.columns:
        df['timeseries_id'] = 'Unknown'
    df['species'] = df['species'].astype(str).str.strip()
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['timeseries_id'] = df['timeseries_id'].astype(str).str.strip()
    cache_file = 'csv/taxonomic_cache.csv'
    cache_map = {}
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache_df = pd.read_csv(cache_file)
        cache_df['species'] = cache_df['species'].astype(str).str.strip()
        cache_df = cache_df.drop_duplicates(subset=['species'], keep='last')
        cache_map = dict(zip(cache_df['species'], cache_df['taxonomic_group']))
    unique_species = sorted(df['species'].unique())
    to_classify = [sp for sp in unique_species if cache_map.get(sp, 'Other') == 'Other']
    if to_classify:
        batch_size = 100
        for start in range(0, len(to_classify), batch_size):
            chunk = to_classify[start:start + batch_size]
            results = Parallel(n_jobs=-1)(delayed(_classify_species_no_cache)(sp, data_taxonomy, genus_taxonomy) for sp in chunk)
            new_df = pd.DataFrame(results, columns=['species', 'taxonomic_group'])
            new_df['species'] = new_df['species'].astype(str).str.strip()
            existing_df = pd.DataFrame(list(cache_map.items()), columns=['species', 'taxonomic_group'])
            combined = pd.concat([existing_df[~existing_df['species'].isin(new_df['species'])], new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['species'], keep='last')
            _atomic_write_csv(cache_file, combined)
            cache_map = dict(zip(combined['species'], combined['taxonomic_group']))
            print(f"Flushed {start + len(chunk)}/{len(to_classify)} species to {cache_file}.")
    tax_groups = df['species'].map(cache_map).fillna('Other')
    mask = tax_groups.isin(EXCLUDED_GROUPS).values | df['species'].isin(EXCLUDED_SPECIES_LABELS).values
    relabeled = int(mask.sum())
    if relabeled > 0:
        df.loc[mask, 'reason'] = 'taxonomic_excluded'
        _atomic_write_csv(fail_file, df)
        print(f"Relabeled {relabeled} excluded species in {fail_file} with reason='taxonomic_excluded'.")
    else:
        print(f"No excluded species to relabel in {fail_file}.")

def load_iucn_species_set(iucn_file):
    if not os.path.exists(iucn_file) or os.path.getsize(iucn_file) == 0:
        print(f"{iucn_file} missing or empty; no IUCN-priority species available for prioritization.")
        return set()
    df = pd.read_csv(iucn_file)
    if 'species' not in df.columns or 'iucn_status' not in df.columns:
        raise ValueError(f"{iucn_file} missing required columns: species, iucn_status")
    df['species'] = df['species'].astype(str).str.strip()
    df['iucn_status'] = df['iucn_status'].fillna('').astype(str).str.strip().str.upper()
    valid_mask = (df['iucn_status'] != '') & (~df['iucn_status'].isin(IUCN_EXCLUDED_STATUSES))
    return set(df.loc[valid_mask, 'species'])

def partition_tasks_by_priority(tasks, priority_species):
    priority_tasks = []
    other_tasks = []
    for task in tasks:
        sp = str(task[0]).strip()
        if sp in priority_species:
            priority_tasks.append(task)
        else:
            other_tasks.append(task)
    return priority_tasks, other_tasks

def get_requeue_tasks(fail_file, priority_species):
    if not os.path.exists(fail_file) or os.path.getsize(fail_file) == 0:
        return set()
    df = pd.read_csv(fail_file)
    if 'species' not in df.columns or 'dataset' not in df.columns or 'timeseries_id' not in df.columns or 'reason' not in df.columns:
        return set()
    df['species'] = df['species'].astype(str).str.strip()
    df['dataset'] = df['dataset'].astype(str).str.strip()
    df['timeseries_id'] = df['timeseries_id'].astype(str).str.strip()
    df['reason'] = df['reason'].astype(str).str.strip()
    mask = (df['reason'] == 'timeout_dl') & (df['species'].isin(priority_species))
    return {(row['species'], row['dataset'], row['timeseries_id']) for _, row in df[mask].iterrows()}

def run_pipeline(n_jobs, per_species_timeout):
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    if n_jobs < 1:
        n_jobs = max(1, mp.cpu_count() + n_jobs + 1)
    backup_dl_file(DL_FILE)
    deduplicate_backups()
    restore_missing_rows_from_backups(DL_FILE)
    datasets_dict, species_list_raw, data_taxonomy, genus_taxonomy = setup_data(min_records)
    clean_failed_series_from_results(DL_FILE, datasets_dict, late_transition_window)
    clean_excluded_from_results(DL_FILE, data_taxonomy, genus_taxonomy)
    update_taxonomic_groups_from_mapping(DL_FILE)
    reconcile_fail_file_with_dl(DL_FILE, FAIL_FILE)
    dl_tasks = get_tasks_from_file(DL_FILE)
    failed_tasks = get_failed_tasks_from_file(FAIL_FILE)
    print(f"Loaded {len(species_list_raw)} raw species. Already processed DL tasks: {len(dl_tasks)}. Failed tasks: {len(failed_tasks)}.")
    priority_species = load_iucn_species_set(IUCN_FILE)
    print(f"Loaded {len(priority_species)} IUCN-priority species from {IUCN_FILE}.")
    requeue_tasks = get_requeue_tasks(FAIL_FILE, priority_species)
    print(f"Requeueing {len(requeue_tasks)} IUCN-priority tasks with reason='timeout_dl'.")
    requeue_keys_norm = {(_norm_key(s), _norm_key(d), _norm_key(t)) for s, d, t in requeue_tasks}
    failed_tasks_for_exclusion = failed_tasks - requeue_tasks
    tasks = get_species_datasets(datasets_dict, dl_tasks, failed_tasks_for_exclusion)
    if not tasks:
        print("No tasks to process.")
    else:
        priority_tasks_all, other_tasks = partition_tasks_by_priority(tasks, priority_species)
        priority_tasks = []
        requeue_phase_tasks = []
        for task in priority_tasks_all:
            sp, ds, ts, _ = task
            k = (_norm_key(sp), _norm_key(ds), _norm_key(ts))
            if k in requeue_keys_norm:
                requeue_phase_tasks.append(task)
            else:
                priority_tasks.append(task)
        print(f"Task partition: {len(priority_tasks)} IUCN-priority, {len(other_tasks)} remaining, {len(requeue_phase_tasks)} requeue (timeout_dl).")
        if priority_tasks:
            print(f"Phase 1 (IUCN priority): processing {len(priority_tasks)} (species, dataset, segment) tasks...")
            run_phase(priority_tasks, n_jobs, data_taxonomy, genus_taxonomy, per_species_timeout)
            print("Phase 1 completed.")
        if other_tasks:
            print(f"Phase 2 (remaining): processing {len(other_tasks)} (species, dataset, segment) tasks...")
            run_phase(other_tasks, n_jobs, data_taxonomy, genus_taxonomy, per_species_timeout)
            print("Phase 2 completed.")
        if requeue_phase_tasks:
            print(f"Phase 3 (requeue timeout_dl): processing {len(requeue_phase_tasks)} (species, dataset, segment) tasks...")
            run_phase(requeue_phase_tasks, n_jobs, data_taxonomy, genus_taxonomy, per_species_timeout)
            print("Phase 3 completed.")
        print("All phases completed.")
    print("Moving failed entries from DL to FAIL_FILE...")
    move_zero_prob_to_fail(DL_FILE, FAIL_FILE, reason='zero_probability')
    print("Move completed.")
    print("Refreshing taxonomic_cache.csv for all species in DL and FAIL...")
    ensure_all_species_classified(DL_FILE, FAIL_FILE, data_taxonomy, genus_taxonomy, n_jobs=n_jobs)
    update_taxonomic_groups_from_mapping(DL_FILE)
    print("DL_FILE updated with fresh taxonomic groups.")
    print("Relabeling excluded species in FAIL_FILE...")
    clean_excluded_from_fail(FAIL_FILE, data_taxonomy, genus_taxonomy)
    print("Re-pruning excluded species from DL_FILE...")
    clean_excluded_from_results(DL_FILE, data_taxonomy, genus_taxonomy)
    print("Reconciling FAIL_FILE with DL_FILE...")
    reconcile_fail_file_with_dl(DL_FILE, FAIL_FILE)
    merged_df_non_gbif = merge_results()
    if merged_df_non_gbif is not None and not merged_df_non_gbif.empty:
        if 'dataset' in merged_df_non_gbif.columns:
            merged_df_non_gbif = merged_df_non_gbif[merged_df_non_gbif['dataset'] != 'GBIF']
    merged_df_full = merge_results()
    if merged_df_full is not None and not merged_df_full.empty:
        if merged_df_non_gbif is not None and not merged_df_non_gbif.empty:
            merged_df_final = merged_df_non_gbif.copy()
        else:
            merged_df_final = merged_df_full.copy()
        out_file = 'csv/bifurcation_results.csv'
        large_file = 'csv/bifurcation_results_large.csv'
        short_file = 'csv/bifurcation_results_short.csv'
        merged_df_final.to_csv(out_file, index=False)
        print(f"Merged results saved to {out_file}")
        update_taxonomic_groups_from_mapping(out_file)
        build_short_outputs(DL_FILE, DEV_FILE, large_file, short_file, min_records, datasets_dict)
        if os.path.exists(large_file) and os.path.getsize(large_file) > 0:
            plot_df = pd.read_csv(large_file)
            if 'taxonomic_group' in plot_df.columns:
                plot_df = plot_df[~plot_df['taxonomic_group'].isin(EXCLUDED_GROUPS)]
                plot_df = plot_df[plot_df['taxonomic_group'] != 'Other']
            if 'transition_year' in plot_df.columns:
                plot_df = plot_df.dropna(subset=['transition_year'])
            if not plot_df.empty:
                print("Generating summary figure from non-GBIF results...")
                create_summary_figure(plot_df, probability_threshold)
                print("Summary figure generated.")
            else:
                print("No non-GBIF results to generate figure.")
        else:
            print("No merged results to generate figure.")
    else:
        print("No merged results to save.")
    print("Merge completed.")
    results_dl = []
    if os.path.exists(DL_FILE):
        results_dl = pd.read_csv(DL_FILE).to_dict('records')
        print(f"DL-EWS results saved: {len(results_dl)} segments.")
    return results_dl

def main(n_jobs=-1, species_batch_size=1000, per_species_timeout=900):
    run_pipeline(n_jobs, per_species_timeout)

if __name__ == "__main__":
    main()
