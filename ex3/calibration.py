import numpy as np


def compute_kx_ky(calib_dict):
    """
    Given a calibration dictionary, compute kx and ky (in units of [px/mm]).
    
    kx -> Number of pixels per millimeter in x direction (ie width)
    ky -> Number of pixels per millimeter in y direction (ie height)
    """
    
    #
    # TO IMPLEMENT
    #
    
    return -1, -1


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal lenght f and baseline b from provided calibration points.

    Note:
    In real life multiple points are useful for calibration - in case there are erroneous points.
    Here, this is not the case. It's OK to use a single point to estimate f, b.
    
    Parameters
    ----------
        calib_dict: dict
            Incomplete calibaration dictionary
        calib_points: pd.DataFrame
            Calibration points provided with data. (Units are given in [mm])
        n_points: int
            Number of points used for estimation
        
    Returns
    -------
        f: float
            Focal lenght [mm]
        b: float
            Baseline [mm]
    """
    # Choose n_points from DataFrame
    if n_points is not None:
        calib_points = calib_points.head(n_points)
    else: 
        n_points = len(calib_points)

    #
    # TO IMPLEMENT
    #

    return -1, -1
