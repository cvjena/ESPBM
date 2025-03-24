__all__ = ["define_prototype", "create_prototype"]

import numpy as np
from scipy.signal.windows import gaussian


def define_prototype(
    sig1: float = 6.0,
    sig2: float = 12.0,
    baseline: float = 0.3,
    prominance: float = 0.25,
    apex_location: float = 1 / 3,
    window_size: int = 60,
    noise: bool = False,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[float, float, float, float, float]]:
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
        window_size (int): Size of the window. Defaults to 60 (learned from the data, see JeFaPaTo project).
        noise (bool): Flag indicating whether to add noise to the prototype.
        return_params (bool): Flag indicating whether to return the prototype parameters.

    Returns:
        numpy.ndarray: The manual prototype.
    """
    apex_idx = int(window_size * apex_location)
    x_on = apex_idx - 2.5 * sig1
    x_of = apex_idx + 2.5 * sig2

    if x_on < 0 or x_of > window_size:
        raise ValueError("Onset and offset are not inside the window. Choose different parameters.")

    y1 = -prominance * gaussian(M=window_size * 2, std=sig1) + baseline
    y2 = -prominance * gaussian(M=window_size * 2, std=sig2) + baseline

    y = np.append(y1[int(window_size * (1 - apex_location)) : window_size], y2[window_size : window_size + int(window_size * (1 - apex_location))])
    if noise:
        y = y + _noise_fct(0.02, len(y))

    if return_params:
        return y, (sig1, sig2, baseline, prominance, apex_location)

    return y


def create_prototype(x: tuple | np.ndarray, window_size: int = 60) -> np.ndarray:
    """
    Define a manual prototype composed of two Gaussians.

    The default parameters are set to the values used in the paper.
    They are learned from the data and are used to define the prototype.

    We compute the onset and offset as 2.5 * sig1 and 2.5 * sig2, respectively.
    This function is used during the optimization process to create the prototype for
    each extracted interval.

    Parameters:
        x (tuple | np.ndarray): The 5 parameters of the prototype.
            sig1 (float): Standard deviation of the first Gaussian.
            sig2 (float): Standard deviation of the second Gaussian.
            baseline (float): Baseline value of the prototype.
            prominance (float): Prominence of the Gaussians.
            apex_location (float): Location of the apex of the prototype. If None, it is set to 40% of the window size.
        window_size (int): Size of the window. Defaults to 60 (learned from the data, see JeFaPaTo project).

    Returns:
        numpy.ndarray: The manual prototype.
    """
    sig1, sig2, baseline, prominance, apex_location = x
    y1 = -prominance * gaussian(window_size * 2, std=sig1) + baseline
    y2 = -prominance * gaussian(window_size * 2, std=sig2) + baseline
    y = np.append(y1[int(window_size * (1 - apex_location)) : window_size], y2[window_size : window_size + int(window_size * (1 - apex_location))])
    return y


def _noise_fct(noise_std: float, window_size=10) -> np.ndarray:
    "Random noise based on learned data."
    # create custom random generator to not interfere with other random calls!
    rand_generator = np.random.RandomState(0)
    return (rand_generator.random(window_size) * 2 - 1) * noise_std
