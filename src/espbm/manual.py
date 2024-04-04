__all__ = ["define_prototype"]

import numpy as np
from scipy.signal import gaussian

def define_prototype(
    sig1: float = 6.3,
    sig2: float = 13.6,
    baseline: float = 0.3,
    prominance: float = 0.25,
    apex_location:float | None = None,
    window_size: int=100,
    noise: bool=False,
    return_params: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[float, float, float, float, int]]:
    """
    Define a manual prototype composed of two Gaussians.
    
    The default parameters are set to the values used in the paper.
    They are learned from the data and are used to define the prototype.
    
    We compute the onset and offset as 3 * sig1 and 3 * sig2, respectively.
    If these values are not inside the window, the definition will raise an error.
    
    Parameters:
        sig1 (float): Standard deviation of the first Gaussian.
        sig2 (float): Standard deviation of the second Gaussian.
        baseline (float): Baseline value of the prototype.
        prominance (float): Prominence of the Gaussians.
        apex_location (int | None): Location of the apex of the prototype. If None, it is set to 40% of the window size.
        window_size (int): Size of the window.
        noise (bool): Flag indicating whether to add noise to the prototype.
        return_params (bool): Flag indicating whether to return the prototype parameters.
    
    Returns:
        numpy.ndarray: The manual prototype.
    """
    
    if apex_location is None:
        # default 40% of the window size
        # this how the prototype is defined in the paper
        apex_location = 0.4
    apex_location = int(window_size * apex_location)
    onset_x  = apex_location - 3 * sig1
    offset_x = apex_location + 3 * sig2
    
    if onset_x < 0 or offset_x > window_size:
        raise ValueError("Onset and offset are not inside the window. Choose different parameters.")

    # TODO replace with scipy.stats.norm.pdf 
    y1 = -prominance * gaussian(window_size*2, std=sig1) + baseline
    y2 = -prominance * gaussian(window_size*2, std=sig2) + baseline
    y = np.append(y1[:window_size], y2[window_size:])
    if noise:
        y = y + _noise_fct(0.02, window_size)
    
    y = y[window_size - apex_location: 2*window_size - apex_location]
    if return_params:
        return y, (sig1, sig2, baseline, prominance, apex_location)
    
    return y

def _noise_fct(
    noise_std: float, 
    window_size=10
) -> np.ndarray:
    "Random noise based on learned data."
    # create custom random generator to not interfere with other random calls!
    rand_generator = np.random.RandomState(0)
    return (rand_generator.random(2*window_size) * 2 - 1) * noise_std