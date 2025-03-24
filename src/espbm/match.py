__all__ = ["find_prototype", "describe", "index_matching", "optim"]

from typing import Any
import warnings

import numpy as np
import stumpy
from scipy import optimize

from espbm.manual import create_prototype


def find_prototype(ear_ts: np.ndarray, prototype: np.ndarray, max_prototype_distance: float = 3.0):
    """
    Find occurrences of a prototype pattern within a time series.

    Parameters:
    ear_ts (np.ndarray): The time series to search for occurrences of the prototype pattern.
    prototype (np.ndarray): The prototype pattern to search for within the time series.
    max_prototype_distance (float, optional): The threshold value used to determine matches. Defaults to 3.0.

    Returns:
    list: A list of intervals where the prototype pattern is found in the time series.
          Each interval is represented as [from, to, distance_to_prototype].
    """

    def threshold(D):
        return np.nanmax([np.nanmean(D) - max_prototype_distance * np.std(D), np.nanmin(D)])

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
    max_match_distance: int = 50,
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


def optim(
    interval: np.ndarray,
    prototype_params: tuple[float, float, float, float, float],
    window_size: int = 60,
) -> tuple[np.ndarray, tuple[float, float, float, float, float]] | tuple[None, None]:
    """
    Optimize the blinking prototype parameters for a given interval.

    Parameters:
        interval (np.ndarray): The interval to optimize the prototype parameters for.
        prototype_params (tuple[float, float, float, float, float]): The prototype parameters.
        window_size (int, optional): The size of the window. Defaults to 60.

    Returns:
        tuple[np.ndarray, tuple[float, float, float, float, float]]: The optimized prototype and its parameters.
    """

    def optimize_prototype_fun(params: tuple, interval: np.ndarray):
        """
        Optimize the prototype function by minimizing the L1 norm between the interval and the prototype.
        Args:
            params (list or array-like): Parameters to create the prototype.
            interval (list or array-like): The interval to compare against the prototype.
        Returns:
            float: The L1 norm (Manhattan distance) between the interval and the prototype.
        """

        prototype = create_prototype(params, window_size=window_size)
        return np.linalg.norm(interval - prototype, ord=1)

    with warnings.catch_warnings():
        # we ignore the warnings from the optimization process
        # as in really really rare cases on of the guesses might be out of bounds
        # and the optimization process will still converge
        warnings.simplefilter("ignore")
        # Note: for now the ranges are hardcoded, but they are sufficient for the given data
        ret = optimize.minimize(
            optimize_prototype_fun,
            prototype_params,
            args=(interval,),
            method="Powell",
            bounds=[
                (0.05, 15.0),  # sig1
                (0.05, 15.0),  # sig2
                (0.15, 0.50),  # baseline
                (0.01, 0.75),  # prominance
                (0.05, 0.50),  # apex_location
            ],
            options={"maxiter": 100, "disp": False},
        )
    if not ret.success:
        return None, None

    optimized_params = ret.x
    optimized_prototype = create_prototype(optimized_params, window_size=window_size)
    return optimized_prototype, tuple(optimized_params)


def interval_stats(
    interval: np.ndarray,
    prototype_params: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    # lets compute some of the important statistics for the interval
    stats: dict[str, Any] = {}

    sig1, sig2, baseline, height, apex_location = prototype_params
    apex_location = int(apex_location * len(interval))

    stats["sig1"] = sig1
    stats["sig2"] = sig2
    stats["baseline"] = baseline
    apex_score = interval[apex_location]
    stats["apex_score"] = apex_score
    stats["heights"] = height
    stats["prominance"] = baseline - apex_score

    stats["apex_location"] = apex_location
    # get the score at apex
    x_on = int(apex_location - 2.5 * sig1)
    x_of = int(apex_location + 2.5 * sig2)
    stats["onset_x"] = x_on
    stats["offset_x"] = x_of
    stats["width"] = x_of - x_on

    # intersections points left and right
    # is the point between onset and apex where the prototype crosses the baseline
    # is the point between apex and offset where the prototype crosses the baseline
    ips_l = (apex_location + x_on) // 2
    ips_r = (apex_location + x_of) // 2

    stats["ips_left"] = ips_l
    stats["ips_right"] = ips_r
    stats["internal_width"] = ips_r - ips_l

    return stats
