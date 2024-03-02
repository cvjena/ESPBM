__all__ = ["ear_time_series", "candidates_overview", "matches"]

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure


def ear_time_series(
    ear_r: np.ndarray,
    ear_l: np.ndarray, 
    xmin: int | None = None, 
    xmax: int | None = None,
) -> tuple[figure.Figure, np.ndarray]:
    """
    Plot the EAR (Eye Aspect Ratio) time series for the right and left eyes.
    This is a helper function to visualize the EAR values over time.

    Parameters:
    - ear_r (np.ndarray): Array containing the EAR values for the right eye.
    - ear_l (np.ndarray): Array containing the EAR values for the left eye.
    - xmin (int | None): Minimum x-axis value. If None, the minimum value will be determined automatically.
    - xmax (int | None): Maximum x-axis value. If None, the maximum value will be determined automatically.

    Returns:
    - fig (figure.Figure): The matplotlib Figure object containing the plotted time series.
    """
    # input validation
    if not isinstance(ear_r, np.ndarray):
        raise TypeError("ear_r must be a numpy array")
    if not isinstance(ear_l, np.ndarray):
        raise TypeError("ear_l must be a numpy array")
    if ear_r.ndim != 1:
        raise ValueError("ear_r must be a 1D array")
    if ear_l.ndim != 1:
        raise ValueError("ear_l must be a 1D array")
    
    if xmin is not None and not isinstance(xmin, int):
        raise TypeError("xmin must be an integer")
    if xmax is not None and not isinstance(xmax, int):
        raise TypeError("xmax must be an integer")
    
    if xmin is None:
        xmin = 0

    if xmax is None:
        xmax = max(len(ear_r), len(ear_l))
    elif xmax > max(len(ear_r), len(ear_l)):
        xmax = max(len(ear_r), len(ear_l))

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 6), sharex=True, sharey=True, gridspec_kw={'hspace': 0.1})
    
    axs[0].plot(ear_r, c='red',  label='Right Eye [EAR Value]')
    axs[1].plot(ear_l, c='blue', label='Left Eye [EAR Value]')

    axs[0].minorticks_on()
    axs[0].set_xlim([xmin, xmax])
    axs[1].set_xlabel('Frame [#]', fontsize="18")
    
    axs[0].set_ylabel('Right Eye [EAR Value]', fontsize="12")
    axs[1].set_ylabel('Left Eye [EAR Value]',  fontsize="12")
    
    fig.suptitle("Eye Aspect Ratio [EAR] Time Series", fontsize="20")
    return fig, axs

    
def candidates_overview(candidates: list[np.ndarray] | np.ndarray) -> figure.Figure:
    """
    Generate a figure showing the overview of candidates' EAR time series.
    
    Parameters:
        candidates (list[np.ndarray] | np.ndarray): A list of numpy arrays or a single numpy array representing the EAR time series of candidates.
        
    Returns:
        figure.Figure: The generated matplotlib figure object.
        
    Raises:
        TypeError: If candidates is not a list of numpy arrays.
    """
    if isinstance(candidates, np.ndarray):
        candidates = [candidates]
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list of numpy arrays")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    
    for idx, candidate in enumerate(candidates):
        ax.plot(candidate, label=f"Candidate {idx+1}")
        
    ax.set_xlabel('Frame [#]', fontsize="18")
    ax.set_ylabel('EAR Value', fontsize="18")
    
    ax.set_ylim([0, 0.5])
    
    ax.legend()
    fig.suptitle(f"Top {len(candidates)} candidates from EAR time series", fontsize="20")
    return fig

    
def matches(
    ear_l: np.ndarray,
    ear_r: np.ndarray,
    matches_l: list[np.ndarray],
    matches_r: list[np.ndarray],
    xmin: int | None = None,
    xmax: int | None = None,
) -> figure.Figure:
    """
    Plot the ear time series with highlighted matches.

    Args:
        ear_l (np.ndarray): Left ear time series.
        ear_r (np.ndarray): Right ear time series.
        matches_l (list[np.ndarray]): List of matches for the left ear.
        matches_r (list[np.ndarray]): List of matches for the right ear.
        xmin (int | None, optional): Minimum x-axis value. Defaults to None.
        xmax (int | None, optional): Maximum x-axis value. Defaults to None.

    Returns:
        figure.Figure: The matplotlib figure object.
    """
    fig, axs = ear_time_series(ear_r, ear_l, xmin, xmax)
    
    for match in matches_l:
        axs[0].axvspan(match[0], match[1], color='green', alpha=0.3)
    for match in matches_r:
        axs[1].axvspan(match[0], match[1], color='green', alpha=0.3)
    return fig