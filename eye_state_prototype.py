import numpy as np
import pandas as pd
import stumpy
from scipy import signal

def fpm(Q: np.ndarray, T: np.ndarray, th=3.0):
    """Fast Pattern Matching"""
    def threshold(D):
        return np.nanmax([np.nanmean(D) - th * np.std(D), np.nanmin(D)])
    matches = stumpy.match(Q, T, max_distance=threshold)
    # match_indices = matches[:, 1]
    return matches

def index_matching(indices1: np.ndarray, indices2: np.ndarray, max_distance=50):
    "Match indices saved in two arrays."
    matched_pairs = []
    no_match = []
    for idx1 in indices1:
        dists = np.abs(indices2 - idx1)
        min_dist = np.min(dists)
        if min_dist < max_distance:
            matched_pairs.append([idx1, indices2[np.argmin(dists)]]) # when there are two equal-dist matches, always keep the first onr
        else:
            no_match.append(idx1)
    return np.array(matched_pairs), np.array(no_match)

def get_apex(T: np.ndarray, m: int, match_indices: np.ndarray):
    """Estimated apex in each extracted matches."""
    apex_indices = []
    apex_proms = []
    for idx in match_indices:
        apex_prom = np.max(T[idx:idx+m]) - np.min(T[idx:idx+m])
        apex_idx = idx + np.argmin(T[idx:idx+m])
        apex_indices.append(apex_idx)
        apex_proms.append(apex_prom)
    return np.array(apex_indices), np.array(apex_proms)

def get_stats(diff: np.ndarray) -> dict:
    """Get statistics (avg, std, median) and save them in a dict."""
    diff_stats = dict()
    diff_stats["avg"] = np.mean(diff)
    diff_stats["std"] = np.std(diff)
    diff_stats["median"] = np.median(diff)
    return diff_stats

def combined_gaussian(sig1, sig2, avg, prom):
    """Manual Protype composed of two Gaussians."""
    y1 = - prom * signal.gaussian(200, std=sig1) + avg
    y2 = - prom * signal.gaussian(200, std=sig2) + avg
    y = np.append(y1[:100], y2[100:])
    return y[60:160]


def smooth(data: np.ndarray, window_len=5, window="flat"):
    "Function for smoothing the data. For now, window type: the moving average (flat)."
    if data.ndim != 1:
        raise ValueError("Only accept 1D array as input.")
    
    if data.size < window_len:
        raise ValueError("The input data should be larger than the window size.")
    
    if window == "flat":
        kernel = np.ones(window_len) / window_len
        s = np.convolve(data, kernel, mode="same")
    return s

def cal_results(ear_r, ear_l, prototype, save_path=None):
    """Caculate and save find peaks and fast pattern matching results."""
    # find peaks
    h_th = 0.15 # height threshold
    p_th = None # prominence threshold
    t_th = None # threshold
    d_th = 50 # distance threshold

    peaks_r, properties_r = signal.find_peaks(-ear_r, height=-h_th, threshold=t_th, prominence=p_th, distance=d_th)
    heights_r = - properties_r["peak_heights"]
    peaks_l, properties_l = signal.find_peaks(-ear_l, height=-h_th, threshold=-0.1, prominence=p_th, distance=d_th)
    heights_l = - properties_l["peak_heights"]

    # fast pattern matching
    m = len(prototype) # m = 100
    fpm_th = 3.0
    matches_r = fpm(Q=prototype, T=ear_r, th=fpm_th)
    matches_l = fpm(Q=prototype, T=ear_l, th=fpm_th)
    match_indices_r = matches_r[:, 1]
    match_indices_l = matches_l[:, 1]

    # index matching
    sorted_indices_r = np.sort(match_indices_r)
    sorted_indices_l = np.sort(match_indices_l)
    matched_pairs, no_match = index_matching(sorted_indices_r, sorted_indices_l, max_distance=50)

    # save results
    results = {}
    results["match_indices_r"] = matches_r[:, 1]
    results["match_values_r"] = matches_r[:, 0] 
    results["match_indices_l"] = matches_l[:, 1]
    results["match_values_l"] = matches_l[:, 0]
    results["sorted_indices_r"] = sorted_indices_r
    results["sorted_indices_l"] = sorted_indices_l
    results["matched_pairs_r"] = matched_pairs[:, 0]
    results["matched_pairs_l"] = matched_pairs[:, 1]
    results["peaks_r"] = peaks_r
    results["heights_r"] = heights_r
    results["peaks_l"] = peaks_l
    results["heights_l"] = heights_l

    if save_path is not None:
        results_df = pd.DataFrame({key:pd.Series(value) for key, value in results.items()})
        results_df.to_csv(save_path)
    
    return results