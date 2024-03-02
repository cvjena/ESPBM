__all__ = ["find_prototype", "match_found_intervals"]

import numpy as np
import stumpy

def find_prototype(ear_ts: np.ndarray, prototype: np.ndarray, th=3.0):
    """
    Find occurrences of a prototype pattern within a time series.

    Parameters:
    ear_ts (np.ndarray): The time series to search for occurrences of the prototype pattern.
    prototype (np.ndarray): The prototype pattern to search for within the time series.
    th (float, optional): The threshold value used to determine matches. Defaults to 3.0.

    Returns:
    list: A list of intervals where the prototype pattern is found in the time series.
          Each interval is represented as [from, to, distance_to_prototype].
    """
    def threshold(D):
        return np.nanmax([np.nanmean(D) - th * np.std(D), np.nanmin(D)])
    
    matches = stumpy.match(prototype, ear_ts, max_distance=threshold)
    # sort the matches by index to get the original order
    matches = sorted(matches, key=lambda x: x[1])
    
    intervals = []
    # transform the matches to be [from, to, distance_to_prototype]
    for match in matches:
        intervals.append([match[1], match[1] + len(prototype), match[0]])
    
    return intervals

def describe(matches: np.ndarray):
    """
    Prints information about the given matches array.
    
    Parameters:
        matches (np.ndarray): An array of matches.
    
    Returns:
        None
    """
    print(f"Contains {len(matches)} matches")
    print(f"Matches: {matches}")

def index_matching(
    matches_l: np.ndarray,
    matches_r: np.ndarray,
    max_match_distance: int=50,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Perform index matching between two arrays of matches.

    Args:
        matches_l (np.ndarray): Array of matches for the left side.
        matches_r (np.ndarray): Array of matches for the right side.
        max_match_distance (int, optional): Maximum distance allowed between matches. Defaults to 50.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of matches, 
        where the first list corresponds to the matches from the left side and the second list 
        corresponds to the matches from the right side.

    """
    start_idx_l = np.array(matches_l)[:, 0]
    start_idx_r = np.array(matches_r)[:, 0]
    
    pairs = []
    
    for i, idx_l in enumerate(start_idx_l):
        dists = np.abs(start_idx_r - idx_l)
        
        min_dist = np.min(dists)
        min_argl = np.argmin(dists)
        
        if min_dist < max_match_distance:
            # when there are two equal-dist matches, always keep the first one
            pairs.append([i, min_argl])
            
    n_matches_l = [matches_l[i] for i, _ in pairs]
    n_matches_r = [matches_r[j] for _, j in pairs]
    
    # TODO return also the non-matches        
    return n_matches_l, n_matches_r