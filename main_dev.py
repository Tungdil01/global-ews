#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys, gc, warnings, json, tempfile, shutil, time
import numpy as np
import pandas as pd
from scipy.linalg import eig, lstsq
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from tqdm import tqdm
warnings.filterwarnings('ignore')
from main_dl import load_combined_data

def set_seed(seed=1234):
    np.random.seed(seed)

def s_map_with_theta(ts, E, tau, theta, predict_step=1):
    n = len(ts)
    if n < E*tau + predict_step + 1:
        return [], 0.0
    X = []
    for t in range(E*tau, n - predict_step):
        vec = np.array([ts[t - i*tau] for i in range(E)])
        X.append(vec)
    X = np.array(X)
    if len(X) < 2:
        return [], 0.0
    Y = np.array([ts[t+predict_step] for t in range(E*tau, n - predict_step)])
    n_pts = len(X)
    jacobians = []
    preds = []
    obs = []
    for i in range(n_pts):
        x0 = X[i]
        dists = np.linalg.norm(X - x0, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist == 0:
            mean_dist = 1.0
        weights = np.exp(-theta * dists / mean_dist)
        X_w = np.hstack([np.ones((n_pts,1)), X])
        W = np.diag(weights)
        lhs = X_w.T @ W @ X_w
        rhs = X_w.T @ W @ Y
        coeff = lstsq(lhs, rhs)[0]
        J = np.zeros((E, E))
        J[0, :] = coeff[1:1+E]
        for k in range(1, E):
            J[k, k-1] = 1.0
        jacobians.append(J)
        pred = coeff[0] + np.dot(coeff[1:], X[i])
        preds.append(pred)
        obs.append(Y[i])
    if len(preds) > 1 and np.std(obs) > 1e-10 and np.std(preds) > 1e-10:
        rho = pearsonr(preds, obs)[0]
    else:
        rho = 0.0
    return jacobians, rho

def compute_dev_for_window(segment, E, tau, theta_candidates):
    if np.std(segment) < 1e-10:
        return None, None, None, None
    seg_norm = (segment - np.mean(segment)) / np.std(segment)
    if np.isnan(seg_norm).any():
        return None, None, None, None
    best_rho = -np.inf
    best_jac = []
    for th in theta_candidates:
        jac, rho = s_map_with_theta(seg_norm, E, tau, th, predict_step=1)
        if rho > best_rho:
            best_rho = rho
            best_jac = jac
    if len(best_jac) == 0:
        return None, None, None, None
    abs_vals = []
    re_vals = []
    im_vals = []
    for J in best_jac:
        eigvals = eig(J, left=False, right=False)
        idx = np.argmax(np.abs(eigvals))
        lam = eigvals[idx]
        abs_vals.append(abs(lam))
        re_vals.append(lam.real)
        im_vals.append(lam.imag)
    mean_abs = np.mean(abs_vals)
    mean_re = np.mean(re_vals)
    mean_im = np.mean(im_vals)
    return mean_abs, mean_re, mean_im, best_rho

def compute_dev_series(abundance, E, tau, window_size, theta_candidates):
    ts = abundance.values.flatten()
    n = len(ts)
    if window_size < 10:
        return []
    dev_vals = []
    for start in range(0, n - window_size + 1):
        seg = ts[start:start+window_size]
        if np.std(seg) == 0:
            continue
        mean_abs, mean_re, mean_im, rho = compute_dev_for_window(seg, E, tau, theta_candidates)
        if mean_abs is None:
            continue
        dev_vals.append((start + window_size, mean_abs, mean_re, mean_im, rho))
    return dev_vals

def early_warning_from_dev(dev_series, threshold=0.85):
    if len(dev_series) < 3:
        return False
    abs_vals = [d[1] for d in dev_series]
    if max(abs_vals) < threshold:
        return False
    last_3_abs = abs_vals[-3:]
    indices = np.array([0,1,2])
    slope = np.polyfit(indices, last_3_abs, 1)[0]
    return slope > 0

def _save_partial_result(species_name, warning, final_re, final_im, bifurcation_type, pid):
    os.makedirs('csv/partial', exist_ok=True)
    out_file = f'csv/partial/results_{pid}.csv'
    tmp_file = out_file + '.tmp'
    df_part = pd.DataFrame([{
        'species': species_name,
        'early_warning': warning,
        'final_re': float(f"{final_re:.3g}"),
        'final_im': float(f"{final_im:.3g}"),
        'bifurcation_type': bifurcation_type
    }])
    if not os.path.exists(out_file):
        df_part.to_csv(tmp_file, index=False, encoding='utf-8')
    else:
        df_part.to_csv(tmp_file, mode='a', header=False, index=False, encoding='utf-8')
    shutil.move(tmp_file, out_file)

def _save_partial_dev_series(species_name, dev_series, pid):
    os.makedirs('csv/partial', exist_ok=True)
    out_file = f'csv/partial/dev_series_{pid}.jsonl'
    tmp_file = out_file + '.tmp'
    record = {species_name: dev_series}
    with open(tmp_file, 'a') as f:
        f.write(json.dumps(record, separators=(',',':')) + '\n')
    shutil.move(tmp_file, out_file)

def _acquire_dev_file_lock(lock_path, timeout=60):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise RuntimeError("Could not acquire dev file lock: " + lock_path)
            time.sleep(0.05)

def _release_dev_file_lock(lock_path):
    os.remove(lock_path)

def _save_dev_series(species_name, dev_series, pid):
    os.makedirs('csv/partial', exist_ok=True)
    out_file = f'csv/partial/dev_series_{pid}.jsonl'
    lock_path = out_file + '.lock'
    _acquire_dev_file_lock(lock_path)
    try:
        record = {species_name: dev_series}
        with open(out_file, 'a') as f:
            f.write(json.dumps(record, separators=(',',':')) + '\n')
    finally:
        _release_dev_file_lock(lock_path)

def _acquire_species_lock(species_name):
    lock_dir = 'csv/locks'
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{species_name}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def _release_species_lock(species_name):
    lock_path = os.path.join('csv/locks', f"{species_name}.lock")
    try:
        os.remove(lock_path)
    except OSError:
        pass

def select_parameters(ts, E_range, tau_range, window_candidates, theta=0, improvement_threshold=0.01):
    n = len(ts)
    valid_windows = [w for w in window_candidates if 10 <= w < n]
    if not valid_windows:
        raise ValueError("No valid window candidates")
    results = []
    for w in valid_windows:
        best_rho = -np.inf
        best_E = None
        best_tau = None
        for E in E_range:
            if E * max(tau_range) >= w:
                continue
            for tau in tau_range:
                if E * tau >= w:
                    continue
                rhos = []
                for start in range(0, n - w + 1):
                    seg = ts[start:start+w]
                    if np.std(seg) == 0:
                        continue
                    seg_norm = (seg - np.mean(seg)) / np.std(seg)
                    _, rho = s_map_with_theta(seg_norm, E, tau, theta, predict_step=1)
                    if rho is not None and not np.isnan(rho):
                        rhos.append(rho)
                if len(rhos) == 0:
                    continue
                mean_rho = np.mean(rhos)
                if mean_rho > best_rho or (best_E is not None and abs(mean_rho - best_rho) < 1e-12 and (E < best_E or (E == best_E and tau < best_tau))):
                    best_rho = mean_rho
                    best_E = E
                    best_tau = tau
        if best_E is not None:
            results.append((w, best_rho, best_E, best_tau))
    if not results:
        raise ValueError("No valid parameter combinations found")
    results.sort(key=lambda x: x[0])
    windows = [r[0] for r in results]
    rhos = [r[1] for r in results]
    if len(windows) == 1:
        w_opt, rho_opt, E_opt, tau_opt = results[0]
        return E_opt, tau_opt, w_opt
    diffs = np.diff(rhos)
    rel_diffs = diffs / np.maximum(np.abs(rhos[:-1]), 1e-6)
    plateau_idx = None
    for i in range(len(rel_diffs)):
        if abs(rel_diffs[i]) < improvement_threshold:
            plateau_idx = i
            break
    if plateau_idx is None:
        plateau_idx = len(rel_diffs) - 1
    w_opt = windows[plateau_idx]
    E_opt = results[plateau_idx][2]
    tau_opt = results[plateau_idx][3]
    return E_opt, tau_opt, w_opt

def process_one_species(species_name, full_data, min_records=10):
    if not _acquire_species_lock(species_name):
        return None
    try:
        sp_data = full_data[full_data['SPECIES_ID'] == species_name].copy()
        if len(sp_data) == 0:
            return None
        sp_data = sp_data.sort_values('TIME')
        abundance = sp_data['ABUNDANCE']
        ts_vals = abundance.values
        n = len(ts_vals)
        if n < min_records:
            return None
        E_range = range(1, min(10, n//5)+1)
        tau_range = range(1, min(10, n//5)+1)
        window_candidates = [int(n*0.05), int(n*0.1), int(n*0.15), int(n*0.2), int(n*0.25), int(n*0.3), int(n*0.4), int(n*0.5)]
        window_candidates = [w for w in window_candidates if 10 <= w < n]
        if not window_candidates:
            return None
        try:
            E, tau, best_window = select_parameters(ts_vals, E_range, tau_range, window_candidates, theta=0, improvement_threshold=0.01)
        except ValueError:
            return None
        theta_candidates = np.arange(0, 2.6, 0.5).tolist()
        dev_series = compute_dev_series(abundance, E, tau, best_window, theta_candidates)
        if len(dev_series) == 0:
            return None
        abs_vals = [d[1] for d in dev_series]
        warning = early_warning_from_dev(dev_series, threshold=0.85)
        final_abs = dev_series[-1][1]
        final_re = dev_series[-1][2]
        final_im = dev_series[-1][3]
        if final_abs >= 0.85:
            if abs(final_im) < 0.05:
                if final_re > 0.5:
                    btype = 'fold'
                elif final_re < -0.5:
                    btype = 'period-doubling'
                else:
                    btype = 'fold_or_transcritical'
            else:
                btype = 'Hopf'
        else:
            btype = 'no-bifurcation'
        if len(dev_series) >= 3:
            last_5_idx = np.array([d[0] for d in dev_series[-5:]])
            last_5_abs = np.array([d[1] for d in dev_series[-5:]])
            if np.std(last_5_idx) > 0 and np.std(last_5_abs) > 0:
                coeff = np.polyfit(last_5_idx, last_5_abs, 1)
                if coeff[0] > 0:
                    time_to_transition = (1.0 - coeff[1]) / coeff[0]
                else:
                    time_to_transition = np.inf
            else:
                time_to_transition = np.inf
        else:
            time_to_transition = np.inf
        dev_series_clean = [(int(idx), float(abs_val), float(re), float(im), float(rho)) for idx, abs_val, re, im, rho in dev_series]
        pid = os.getpid()
        _save_partial_result(species_name, warning, final_re, final_im, btype, pid)
        _save_partial_dev_series(species_name, dev_series_clean, pid)
        return (species_name, warning, final_re, final_im, btype, dev_series_clean)
    except Exception:
        return None
    finally:
        _release_species_lock(species_name)

def get_processed_species(partial_dir):
    processed = set()
    if not os.path.exists(partial_dir):
        return processed
    for fname in os.listdir(partial_dir):
        if fname.startswith('results_') and fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(partial_dir, fname))
            if 'species' in df.columns:
                processed.update(df['species'].unique())
    return processed

def _clean_all_locks():
    lock_dir = 'csv/locks'
    if os.path.exists(lock_dir):
        for fname in os.listdir(lock_dir):
            if fname.endswith('.lock'):
                try:
                    os.remove(os.path.join(lock_dir, fname))
                except OSError:
                    pass

def run_analysis(full_data, species_list, n_jobs=-1, priority_half=0, min_records=10):
    species_sorted = sorted(species_list)
    mid = len(species_sorted) // 2
    if priority_half == 1:
        primary = species_sorted[:mid]
        secondary = species_sorted[mid:]
    elif priority_half == 2:
        primary = species_sorted[mid:]
        secondary = species_sorted[:mid]
    else:
        primary = species_sorted
        secondary = []
    all_to_process = primary + secondary
    partial_dir = 'csv/partial'
    os.makedirs(partial_dir, exist_ok=True)
    _clean_all_locks()
    processed_species = get_processed_species(partial_dir)
    species_to_process = [sp for sp in all_to_process if sp not in processed_species]
    print(f"Already processed: {len(processed_species)}, remaining to process: {len(species_to_process)}")
    if species_to_process:
        results = Parallel(n_jobs=n_jobs)(delayed(process_one_species)(sp, full_data, min_records) for sp in tqdm(species_to_process))
    csv_files = []
    jsonl_files = []
    for fname in os.listdir(partial_dir):
        path = os.path.join(partial_dir, fname)
        if fname.startswith('results_') and fname.endswith('.csv'):
            csv_files.append((path, os.path.getmtime(path)))
        elif fname.startswith('dev_series_') and fname.endswith('.jsonl'):
            jsonl_files.append((path, os.path.getmtime(path)))
    csv_files.sort(key=lambda x: x[1])
    jsonl_files.sort(key=lambda x: x[1])
    all_rows = []
    for path, _ in csv_files:
        all_rows.append(pd.read_csv(path))
    if all_rows:
        output_df = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset='species', keep='last')
        output_df.to_csv('csv/early_warning_dev.csv', index=False, encoding='utf-8')
        print("Early warning results saved to csv/early_warning_dev.csv")
    else:
        output_df = pd.DataFrame()
        print("No results to save.")
    all_dev_series = {}
    for path, _ in jsonl_files:
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                for sp, dev_series in data.items():
                    if len(dev_series) == 0:
                        continue
                    last_idx = dev_series[-1][0]
                    if sp not in all_dev_series:
                        all_dev_series[sp] = (dev_series, last_idx)
                    else:
                        existing_dev, existing_last = all_dev_series[sp]
                        if last_idx > existing_last:
                            all_dev_series[sp] = (dev_series, last_idx)
    final_dev_series = {sp: dev for sp, (dev, _) in all_dev_series.items()}
    if final_dev_series:
        with open('csv/dev_series.json', 'w') as f:
            json.dump(final_dev_series, f, indent=None, separators=(',',':'))
        print("DEV time series saved to csv/dev_series.json")
    print("\nEarly warning summary:")
    if not output_df.empty:
        print(output_df['early_warning'].value_counts())
        print("\nBifurcation type summary:")
        print(output_df['bifurcation_type'].value_counts())
    return output_df

def process_dev_for_species(species_name, species_data, data_taxonomy=None, genus_taxonomy=None, min_records=10):
    try:
        sp_data = species_data.copy()
        sp_data = sp_data.sort_values('TIME')
        abundance = sp_data['ABUNDANCE']
        ts_vals = abundance.values
        n = len(ts_vals)
        if n < min_records:
            return {'species': species_name, 'error': 'insufficient length'}
        E_range = range(1, min(10, n//5)+1)
        tau_range = range(1, min(10, n//5)+1)
        window_candidates = [int(n*0.05), int(n*0.1), int(n*0.15), int(n*0.2), int(n*0.25), int(n*0.3), int(n*0.4), int(n*0.5)]
        window_candidates = [w for w in window_candidates if 10 <= w < n]
        if not window_candidates:
            return {'species': species_name, 'error': 'no valid window'}
        try:
            E, tau, best_window = select_parameters(ts_vals, E_range, tau_range, window_candidates, theta=0, improvement_threshold=0.01)
        except ValueError:
            return {'species': species_name, 'error': 'parameter selection failed'}
        theta_candidates = np.arange(0, 2.6, 0.5).tolist()
        dev_series = compute_dev_series(abundance, E, tau, best_window, theta_candidates)
        if len(dev_series) == 0:
            return {'species': species_name, 'error': 'no dev series'}
        warning = early_warning_from_dev(dev_series, threshold=0.85)
        final_re = dev_series[-1][2]
        final_im = dev_series[-1][3]
        final_abs = dev_series[-1][1]
        if final_abs >= 0.85:
            if abs(final_im) < 0.05:
                if final_re > 0.5:
                    btype = 'fold'
                elif final_re < -0.5:
                    btype = 'period-doubling'
                else:
                    btype = 'fold_or_transcritical'
            else:
                btype = 'Hopf'
        else:
            btype = 'no-bifurcation'
        if len(dev_series) >= 3:
            last_5_idx = np.array([d[0] for d in dev_series[-5:]])
            last_5_abs = np.array([d[1] for d in dev_series[-5:]])
            if np.std(last_5_idx) > 0 and np.std(last_5_abs) > 0:
                coeff = np.polyfit(last_5_idx, last_5_abs, 1)
                if coeff[0] > 0:
                    time_to_transition = (1.0 - coeff[1]) / coeff[0]
                else:
                    time_to_transition = np.inf
            else:
                time_to_transition = np.inf
        else:
            time_to_transition = np.inf
        dev_series_clean = [(int(idx), float(abs_val), float(re), float(im), float(rho)) for idx, abs_val, re, im, rho in dev_series]
        pid = os.getpid()
        _save_dev_series(species_name, dev_series_clean, pid)
        return {
            'species': species_name,
            'early_warning': warning,
            'final_re': final_re,
            'final_im': final_im,
            'final_abs': final_abs,
            'bifurcation_type': btype,
            'time_to_transition': time_to_transition
        }
    except Exception as e:
        return {'species': species_name, 'error': str(e)}

def main():
    set_seed(1234)
    min_records = 10
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    biotime_path = os.path.join(data_dir, 'biotime_v2_full_2025', 'biotime_v2_rawdata_2025.csv')
    rivfishtime_path = os.path.join(data_dir, 'RivFishTIME_1873_12_Dataset', '1873_2_RivFishTIME_SurveyTable.csv')
    cedar_creek_paths = {
        'e120_plant': os.path.join(data_dir, 'knb-lter-cdr.273.11', 'e120_Plant_aboveground_biomass.csv'),
        'e120_arthropod': os.path.join(data_dir, 'knb-lter-cdr.418.8', 'e120_Main_Plots_All_Arthropod_Insect_Sweepnet_Sampling_1996_2006.txt'),
        'e244': os.path.join(data_dir, 'knb-lter-cdr.383.10', 'E244_Enemy_Removal_Aboveground_Biomass.csv')
    }
    gbif_path = os.path.join(data_dir, 'gbif', 'output.csv')
    gpdd_path = os.path.join(data_dir, 'gpdd', 'gpdd_final.csv')
    priority_half = 0
    if len(sys.argv) > 1:
        try:
            priority_half = int(sys.argv[1])
        except ValueError:
            pass
    print("Loading combined data with full filtering (microbial, Sibly, Brook, quality selection)...")
    full_data, species_list, _ = load_combined_data(
        biotime_path, rivfishtime_path, cedar_creek_paths, gbif_path, gpdd_path,
        min_records=min_records, use_existing_biotime=True
    )
    run_analysis(full_data, species_list, n_jobs=-1, priority_half=priority_half, min_records=min_records)

if __name__ == "__main__":
    main()
