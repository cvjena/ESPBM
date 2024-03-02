__all__ = ["extract_candidates"]

import numpy as np
import stumpy

def extract_candidates(
    ear_ts: np.ndarray, 
    window_length:int=100, 
    max_matches:int=10
) -> list[np.ndarray]:
    """
    Extracts candidate motifs from the given time series using the Matrix Profile algorithm.

    Parameters:
    - ear_ts (np.ndarray): The input time series.
    - window_length (int): The length of the sliding window used for candidates extraction. Default is 100. Should be based on the FPS of the video.
    - max_matches (int): The maximum number of candidates to extract. Default is 10.

    Returns:
    - candidates (list[np.ndarray]): A list of candidate candidates extracted from the time series.
    """
    # input validation
    if not isinstance(ear_ts, np.ndarray):
        raise TypeError("ear_ts must be a numpy array")
    if ear_ts.ndim != 1:
        raise ValueError("ear_ts must be a 1D array")
    if not isinstance(window_length, int):
        raise TypeError("window_length must be an integer")
    if not isinstance(max_matches, int):
        raise TypeError("max_matches must be an integer")
    
    mp = stumpy.stump(ear_ts, window_length) # matrix profile
    _, candidates_idx = stumpy.motifs(ear_ts, mp[:, 0], max_matches=max_matches)
    return [ear_ts[idx:idx+window_length] for idx in (candidates_idx[0] + window_length // 2)]
