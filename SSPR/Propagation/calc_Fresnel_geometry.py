""" This function calculates the common parameters to calculate the Fresnel transformation for a
multi-distance propagation data set
"""

import numpy as np


def calc_Fresnel_geometry(lam, det_pixel_size, z01, z02, z12=None, scale_fac=1):
    """
    :param lam: Wavelength of the incident radiation
    :param det_pixel_size:  Detector pixel size
    :param z01: Distance from source (focus spot) to sample
    :param z02: Distance from source (focus spot) to detector, if given
    :param z12: Distance from source (focus spot) to detector, if given
    :param scale_fac:  Scales the magnification further. For example, if another lens is introduced that magnifies by a
                   factor of 2, then 2 would be the scale factor

    :return:
    F_calc: Fresnel numbers corresponding to the given parameters
    """
    if isinstance(z01, (int, float)):
        z01 = np.array(z01, dtype=np.float32).reshape(-1, 1)

    num_distances = len(z01)
    M = np.zeros((num_distances, 1))  # Magnification
    z_eff = np.zeros((num_distances, 1))  # Effective propagation distance
    dx_eff = np.zeros((num_distances, 1))  # Effective pixel size
    F_calc = np.zeros((num_distances, 1))  # Fresnel numbers corresponding to above parameters

    if z02 is None:
        z02 = np.zeros((num_distances, 1))  # Distance from source to detector
        # Calculate parameters for each distance
        for k in range(num_distances):
            z01[k, 0] = z01[k, 0]
            z02[k, 0] = z12 + z01[k, 0]
            M[k, 0] = z02 / z01[k, 0]
            z_eff[k, 0] = z12[k, 0] / M[k, 0]
            dx_eff[k, 0] = det_pixel_size / M[k, 0] / scale_fac
            F_calc[k, 0] = dx_eff[k, 0] ** 2 / (z_eff[k, 0] * lam)
    else:
        z12 = np.zeros((num_distances, 1))  # Distance from sample to detector
        # Calculate parameters for each distance
        for k in range(num_distances):
            z01[k, 0] = z01[k, 0]
            z12[k, 0] = z02 - z01[k, 0]
            M[k, 0] = z02 / z01[k, 0]
            z_eff[k, 0] = z12[k, 0] / M[k, 0]
            dx_eff[k, 0] = det_pixel_size / M[k, 0] / scale_fac
            F_calc[k, 0] = dx_eff[k, 0] ** 2 / (z_eff[k, 0] * lam)

    return F_calc
