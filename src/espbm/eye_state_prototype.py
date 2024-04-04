import numpy as np
import pandas as pd
import stumpy
from scipy import signal
import matplotlib.pyplot as plt

##########Prototyping##########
# prototype extraction
def motif_extraction(ear_ts: np.ndarray, m=100, max_matches=10):
    """Extract top motifs in EAR time series."""
    mp = stumpy.stump(ear_ts, m) # matrix profile
    motif_distances, motif_indices = stumpy.motifs(ear_ts, mp[:, 0], max_matches=max_matches)
    return motif_distances, motif_indices

def learn_prototypes(ear_ts: np.ndarray, m=100, max_matches=10):
    """Return top motifs."""
    _, motif_indices = motif_extraction(ear_ts, m, max_matches)
    motifs = np.array([ear_ts[idx:idx+m] for idx in (motif_indices[0] + m // 2)])
    return motifs

# manual definition
def combined_gaussian(sig1: float, sig2: float, avg: float, prom: float, m=100, mu=40, noise=None):
    """Manual prototype composed of two Gaussians."""
    y1 = - prom * signal.gaussian(2*m, std=sig1) + avg
    y2 = - prom * signal.gaussian(2*m, std=sig2) + avg
    y = np.append(y1[:m], y2[m:])
    
    if noise is not None: 
        y = y + noise
    
    return y[m-mu:2*m-mu]

def nosie(noise_std: float, m=100):
    "Random noise based on learned data."
    np.random.seed(0)
    noise = (np.random.random(2*100) * 2 - 1) * noise_std
    return noise

##########Detection##########
# blink pattern matching
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

# find peaks
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

def cal_bpm_results(ear_r: np.ndarray, ear_l: np.ndarray, prototype, fpm_th=3.0, h_th=0.15, p_th=None, t_th=None, d_th=50, save_path=None):
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
    peaks_r, heights_r = find_peaks_in_ear_ts(ear_r, h_th, t_th, p_th, d_th)
    peaks_l, heights_l = find_peaks_in_ear_ts(ear_l, h_th, t_th, p_th, d_th)

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

##########Analysis##########
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

##########Analysis##########
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

def plot_zoom_in(ax, ear, sorted_indices, peaks, heights, subregion, zoom_in_box, m=100):
    """Zoom in subregion on the orginial plot."""
    x1, x2, y1, y2 = subregion
    x_in, y_in, w_in, h_in = zoom_in_box
    axin = ax.inset_axes([x_in, y_in, w_in, h_in], 
                        xlim=(x1, x2), ylim=(y1, y2), 
                        xticklabels=[], yticklabels=[])
    for i, match_idx in enumerate(sorted_indices):
        axin.axvspan(match_idx, match_idx + m, 0, 1, facecolor="lightgrey")
        axin.plot(ear, c="r", zorder=1)
        axin.scatter(peaks, heights, marker='x', zorder=2)
        axin.set_xticks([])
        axin.set_yticks([])

    ax.indicate_inset_zoom(axin, edgecolor="black")

def plot_results(ax, ears, results, side, m=100, h_th=0.15, xmin=0, xmax=10000, ymin=-0.1, ymax=0.5, zoom_in_params=None):
    """Plot fast pattern matching vs simple thresholding results for each EAR time series."""
    # set values
    if side == 'right':
        c = 'r'
        ear = ears[0]
        sorted_indices = results["sorted_indices_r"]
        peaks = results["peaks_r"]
        heights = results["heights_r"]
    else:
        c = 'b'
        ear = ears[1]
        sorted_indices = results["sorted_indices_l"]
        peaks = results["peaks_l"]
        heights = results["heights_l"]
    
    # EAR time series
    ax.plot(ear, c=c, zorder=1)
 
    # find peaks
    ax.hlines(h_th, xmin, xmax, linestyles='dashed', zorder=0) # showing threshold
    ax.scatter(peaks, heights, marker='x', zorder=2)

    # fpm detected regions
    for i, match_idx in enumerate(sorted_indices):
        ax.axvspan(match_idx, match_idx + m, 0, 1, facecolor="lightgrey", zorder=-1)
    
    # show numbering
    plot_indices = sorted_indices [(sorted_indices > xmin) & (sorted_indices < xmax)]
    for j in range(len(plot_indices)):
        if np.diff(plot_indices)[j-1] <= 200:
            ax.text(plot_indices[j]+50, 0.38, str(j+1), fontsize=10)
        else:
            ax.text(plot_indices[j]-100, 0.38, str(j+1), fontsize=10)
    
    if zoom_in_params != None:
        subregion, zoom_in_box = zoom_in_params
        plot_zoom_in(ax, ear, sorted_indices, peaks, heights, subregion, zoom_in_box)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])
    if side == 'right':
        ax.set_xticks([])
    else:
        ax.set_xlabel("Frame", fontsize=10)

# other plotting functions for visualization
def plot_fpm(T: np.ndarray, match_indices: np.ndarray, m: int, th: float, xmin=0, xmax=20000, save_path=None, show=False):
    """Plot fast pattern matching, one graph."""
    fig, ax = plt.subplots(figsize=(20, 6), sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Fast Pattern Matching', fontsize='20')

    ax.plot(T)
    ax.set_ylabel('EAR', fontsize='14')

    for i, match_idx in enumerate(match_indices):
        ax.axvspan(match_indices[i], match_indices[i] + m, 0, 1, facecolor="lightgrey")

    ax.set(xlim=[xmin, xmax])
    ax.minorticks_on()
    ax.set_ylabel('EAR')
    ax.set_xlabel('Frame', fontsize ='14')

    if save_path != None:
        plt.savefig(save_path)

    if show == False:
        plt.close()
    else:
        plt.show()

def plot_fpm2(ear_r: np.ndarray, ear_l: np.ndarray, match_indices_r: np.ndarray, match_indices_l: np.ndarray, m: int, th: float, xmin=0, xmax=20000, save_path=None, show=False):
    """Plot fast pattern matching, right and left."""
    fig, axs = plt.subplots(2, figsize=(20, 6), sharex=True, gridspec_kw={'hspace': 0.1})
    plt.suptitle('Fast Pattern Matching', fontsize='20')

    axs[0].plot(ear_r, c='r')
    axs[1].plot(ear_l, c='b')

    for i, match_idx in enumerate(match_indices_r):
        axs[0].axvspan(match_idx, match_idx + m, 0, 1, facecolor="lightgrey")

    for i, match_idx in enumerate(match_indices_l):
        axs[1].axvspan(match_idx, match_idx + m, 0, 1, facecolor="lightgrey")

    axs[0].set(xlim=[xmin, xmax])
    axs[0].minorticks_on()
    axs[0].set_ylabel('EAR right')
    axs[1].set_ylabel('EAR left')
    axs[1].set_xlabel('Frame')
    
    if save_path != None:
        plt.savefig(save_path)

    if show == False:
        plt.close()
    else:
        plt.show()

# histogram
def plot_prom_hist(proms, eye="right", rel_freq=True, xmin=0, xmax=0.5, ymin=0, ymax=0.3, save_path=None):
    """Plot histogram for EAR promineces (both eyes)."""
    if eye == "right":
        color = "r"
    else:
        color = "b"
        
    n_bins = int(np.sqrt(len(proms)))

    fig, ax = plt.subplots()
    ax.set_title(f"Histogram for EAR prominence ({eye} eye)")
    ax.set_xlabel("EAR Prominence")
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])
    if rel_freq == True:
        ax.hist(proms, bins=n_bins, edgecolor="white", weights=np.ones_like(proms) / len(proms), color=color)
        ax.set_ylabel("Relative Frequency")
    else:
        ax.hist(proms, bins=n_bins, edgecolor="white", color=color)
        ax.set_ylabel("Frequency")
    
    if save_path != None:
        plt.savefig(save_path)