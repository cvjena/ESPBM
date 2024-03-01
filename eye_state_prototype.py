import numpy as np
import pandas as pd
import stumpy
from scipy import signal
import matplotlib.pyplot as plt

##########Prototype##########
# prototype extraction
def motif_extraction(ear_ts: np.ndarray, m=100, max_matches=10):
    """Extract top motifs in EAR time series."""
    mp = stumpy.stump(ear_ts, m) # matrix profile
    motif_distances, motif_indices = stumpy.motifs(ear_ts, mp[:, 0], max_matches=max_matches)
    return motif_distances, motif_indices

# manual prototype definition
def combined_gaussian(sig1: float, sig2: float, avg: float, prom: float, m=100, mu=40, noise=None):
    """Manual protype composed of two Gaussians."""
    y1 = - prom * signal.gaussian(2*m, std=sig1) + avg
    y2 = - prom * signal.gaussian(2*m, std=sig2) + avg
    y = np.append(y1[:m], y2[m:])
    
    if noise is not None: 
        y = y + noise
    
    return y[m-mu:2*m-mu]

##########Matching##########
# pattern matching
def fpm(Q: np.ndarray, T: np.ndarray, th=3.0):
    """Fast Pattern Matching"""
    def threshold(D):
        return np.nanmax([np.nanmean(D) - th * np.std(D), np.nanmin(D)])
    matches = stumpy.match(Q, T, max_distance=threshold)
    # match_indices = matches[:, 1]
    return matches

# simple threholding 
def find_peaks_in_ear_ts(ear_ts: np.ndarray, h_th=0.15, p_th=None, t_th=None, d_th=50):
    """
    Find peaks in EAR time series.
    h_th = 0.15 # height threshold
    p_th = None # prominence threshold
    t_th = None # threshold
    d_th = 50 # distance threshold
    """
    peaks, properties = signal.find_peaks(-ear_ts, height=-h_th, threshold=t_th, prominence=p_th, distance=d_th)
    heights = - properties["peak_heights"]
    return peaks, heights

##########Analysis##########
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

def cal_results(ear_r: np.ndarray, ear_l: np.ndarray, prototype, fpm_th=3.0, h_th=0.15, p_th=None, t_th=None, d_th=50, save_path=None):
    """Caculate and save find peaks and fast pattern matching results."""
    # fast pattern matching
    m = len(prototype)
    matches_r = fpm(Q=prototype, T=ear_r, th=fpm_th)
    matches_l = fpm(Q=prototype, T=ear_l, th=fpm_th)
    match_indices_r = matches_r[:, 1]
    match_indices_l = matches_l[:, 1]

    # index matching
    sorted_indices_r = np.sort(match_indices_r)
    sorted_indices_l = np.sort(match_indices_l)
    matched_pairs, no_match = index_matching(sorted_indices_r, sorted_indices_l, max_distance=50)

    # find peaks
    peaks_r, heights_r = find_peaks_in_ear_ts(-ear_r, height=-h_th, threshold=t_th, prominence=p_th, distance=d_th)
    peaks_l, heights_l = find_peaks_in_ear_ts(-ear_l, height=-h_th, threshold=t_th, prominence=p_th, distance=d_th)

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

# visulization
def plot_ear(ear_r: np.ndarray, ear_l: np.ndarray, xmin=0, xmax=20000):
    """Plot right and left ear score."""
    fig, axs = plt.subplots(2, figsize=(20, 6), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    axs[0].plot(ear_r, c='r', label='right eye')
    axs[0].minorticks_on()
    if len(ear_r) < xmax:
        xmax = len(ear_r)
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_title("EAR Time Series", fontsize="30")
    axs[0].set_ylabel('right', fontsize="18")
    axs[1].plot(ear_l, c='b', label='left eye')
    axs[1].set_ylabel('left', fontsize="18")
    axs[1].set_xlabel('Frame', fontsize="18")
    plt.show()
    return True

def plot_mp(ts: np.ndarray, mp: np.ndarray):
    """Plot EAR and Matrix Profile."""
    fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('EAR Score and Matrix Profile', fontsize='30')

    axs[0].plot(ts)
    axs[0].set_ylabel('EAR', fontsize='20')
    axs[0].set(xlim=[0, len(ts)], ylim=[0, 0.4])
    axs[0].minorticks_on()
    axs[1].set_xlabel('Frame', fontsize ='20')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].plot(mp[:, 0])
    plt.show()
    return True

# other utilities
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