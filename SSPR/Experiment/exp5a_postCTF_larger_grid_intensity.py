"""

References:
    1) "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
    by D. Voelz et al. (for sampling criteria)

The purpose of this code is to extend/pad the final image such that the sampling conditions are met for forward and
backward propagation, otherwise periodic patterns and false interference occurs. This is not needed for all data, but
should be checked, otherwise that phase retrieval result will be completely wrong


"""

import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage, cropToCenter


def smooth_from_row_to_constant_downward(image, start_row, constant_value=None, num_smooth_pixels=10):
    """
    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
    downwards, over the specified number of rows, replacing any existing values.

    Parameters:
        image (2D array): The input image.
        start_row (int): The row from which to start the smooth transition.
        constant_value (float): The value to which the transition should lead.
        num_smooth_pixels (int): The number of rows over which to smooth the transition.

    Returns:
        smoothed_image (2D array): The image with a smooth transition downward to the constant value.
    """
    # Copy the image to avoid modifying the original
    smoothed_image = np.copy(image)

    # Ensure the start_row is within the bounds of the image
    if start_row >= smoothed_image.shape[0] or start_row < 0:
        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")

    # Get the values from the specified start row
    values_at_start_row = smoothed_image[start_row, :]

    # Calculate the range for smoothing (ensure it does not exceed bounds)
    end_smooth_row = min(smoothed_image.shape[0] - 1, start_row + num_smooth_pixels)

    # Smooth transition from the values in the start row to the constant value
    for i, row in enumerate(range(start_row, end_smooth_row + 1)):
        weight = i / num_smooth_pixels  # Linear interpolation weight
        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value

    # Set all rows below the smoothing region to the constant value
    smoothed_image[end_smooth_row + 1:, :] = constant_value

    return smoothed_image



save = True
extend_image = True
plot_result = True

N_pad = 6000  # Pad size to satisfy sampling criteria

# For the image extension - if True
ellipse_size_y = 0.4
ellipse_size_x = 0.4
transition_length_y = 50
transition_length_x = 50
fade_to_val = 1.0
num_segments = None

run_holo = "586"

# Directories with data
dir_main = "/Users/danielhodge/Desktop/"
dir_exp = "run" + run_holo + "_exp_preprocessed/"

# File to import
tiff_holo_with_speckle_ffc_extended_decon = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended_decon.tiff"

# File to save
tiff_holo_with_speckle_ffc_extended_decon_larger_grid = ("run" + run_holo +
                                                         "_exp_holos_with_speckle_FFC_extended_decon_larger_grid.tiff")

# Import hologram intensity
I = np.array(imread(dir_main + dir_exp + tiff_holo_with_speckle_ffc_extended_decon), dtype=np.float32)

I = padToSize(img=I, outputSize=[N_pad, N_pad], padMethod='replicate', padType='both', padValue=None)

if extend_image:
    I, _ = fadeoutImage(img=I,
                             fadeMethod='rectangle',
                             ellipseSize=[ellipse_size_y, ellipse_size_x],
                             transitionLength=[transition_length_y, transition_length_x],
                             fadeToVal=fade_to_val,
                             numSegments=num_segments,
                             bottomApply=False)

# I = smooth_from_row_to_constant_downward(I, start_row=3800, constant_value=1.0, num_smooth_pixels=100)

if plot_result:
    showImg(I)

imwrite(dir_main + dir_exp + tiff_holo_with_speckle_ffc_extended_decon_larger_grid, I)
