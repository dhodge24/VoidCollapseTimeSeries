import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

from utils.utilities import create_circular_mask, fadeoutImage
from parameters.Padding import Padding
from parameters.DataDimensions import DataDimensions
from preprocessing.process_image import flip_and_pad


def extend_horizontally(image, mask, percentage_split=0.75):
    """
    The purpose of this function is to use a mask to find corresponding edges of a diffraction pattern
    confined to a circular aperture and extend the edges horizontally to fill the computational domain.
    The extension is applied only to the top percentage of the mask, based on the percentage_split parameter.

    :param image: Image to be extended horizontally such that it fills the computational domain
    :param mask: Circular mask used to find object edges confined to a circular aperture
    :param percentage_split: The percentage (0-1) of the mask height where the horizontal extension is applied (default 0.5)
    :return: Extended diffraction pattern
    """

    height, width = image.shape
    new_image = cp.copy(image)

    # Find the last row where the mask has non-zero values
    last_mask_row = cp.max(cp.where(cp.any(mask == 1, axis=1))[0])

    # Determine the row where the horizontal extension stops
    split_row = int(last_mask_row * percentage_split)

    # Find the leftmost non-zero pixel for each row in the top percentage of the mask
    left_indices = cp.argmax(mask[:split_row, :], axis=1)
    left_indices = cp.where(cp.any(mask[:split_row, :], axis=1), left_indices, 0)

    # Find the rightmost non-zero pixel for each row in the top percentage of the mask
    right_indices = width - cp.argmax(mask[:split_row, ::-1], axis=1) - 1
    right_indices = cp.where(cp.any(mask[:split_row, :], axis=1), right_indices, width - 1)

    # Calculate the average of the leftmost and rightmost values for each row in the top percentage
    left_values = new_image[cp.arange(split_row), left_indices]
    right_values = new_image[cp.arange(split_row), right_indices]
    avg_values = (left_values + right_values) / 2

    # Create a matrix with the averaged values propagated across the rows in the top percentage
    avg_propagated = cp.tile(avg_values, (width, 1)).T

    # Create masks for the areas to the left of the leftmost and to the right of the rightmost non-zero pixels in the top percentage
    left_mask = cp.arange(width) < left_indices[:, None]
    right_mask = cp.arange(width) > right_indices[:, None]

    # Update the top portion of the image symmetrically with the averaged values
    new_image[:split_row, :] = cp.where(left_mask | right_mask, avg_propagated, new_image[:split_row, :])

    return new_image


def get_topmost_pixel(img, mask):
    row_indices, col_indices = cp.nonzero(mask)
    topmost_index = cp.argmin(row_indices)
    topmost_row = row_indices[topmost_index]
    topmost_col = col_indices[topmost_index]
    topmost_pixel_value = img[topmost_row, topmost_col]
    return topmost_pixel_value

import cupy as cp

def smooth_from_row_to_constant_downward(image, start_row, constant_value=None, num_smooth_pixels=10):
    """
    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
    downwards, over the specified number of rows, replacing any existing values.

    Parameters:
        image (2D cp.ndarray): The input image (CuPy array).
        start_row (int): The row from which to start the smooth transition.
        constant_value (float): The value to which the transition should lead.
        num_smooth_pixels (int): The number of rows over which to smooth the transition.

    Returns:
        smoothed_image (2D cp.ndarray): The image with a smooth transition downward to the constant value.
    """
    if constant_value is None:
        raise ValueError("constant_value must be provided when using this function with CuPy.")

    # Ensure we're working with a CuPy array and copy to avoid modifying the original
    smoothed_image = cp.array(image, copy=True)

    # Ensure the start_row is within the bounds of the image
    if start_row >= smoothed_image.shape[0] or start_row < 0:
        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")

    # Get the values from the specified start row
    values_at_start_row = smoothed_image[start_row, :].copy()

    # Calculate the range for smoothing (ensure it does not exceed bounds)
    end_smooth_row = min(smoothed_image.shape[0] - 1, start_row + num_smooth_pixels)

    # Smooth transition from the values in the start row to the constant value
    for i, row in enumerate(range(start_row, end_smooth_row + 1)):
        weight = i / num_smooth_pixels  # Linear interpolation weight (Python float is fine)
        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value

    # Set all rows below the smoothing region to the constant value
    if end_smooth_row + 1 < smoothed_image.shape[0]:
        smoothed_image[end_smooth_row + 1:, :] = constant_value

    return smoothed_image


def smooth_from_row_to_constant_upward(image, start_row, mask, constant_value=1, num_smooth_pixels=10):
    """
    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
    upwards, over the specified number of rows, replacing any existing values.

    Parameters:
        image (2D array): The input image.
        start_row (int): The row from which to start the smooth transition.
        constant_value (float): The value to which the transition should lead (e.g., 1).
        num_smooth_pixels (int): The number of rows over which to smooth the transition.

    Returns:
        smoothed_image (2D array): The image with a smooth transition upwards to the constant value,
                                   replacing existing values.
    """
    # Copy the image to avoid modifying the original
    smoothed_image = cp.copy(image)

    if constant_value is None:
        constant_value = get_topmost_pixel(img=image, mask=mask)

    # Ensure the start_row is within the bounds of the image
    if start_row >= smoothed_image.shape[0] or start_row < 0:
        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")

    # Get the values from the specified start row
    values_at_start_row = smoothed_image[start_row, :]

    # Calculate the range for smoothing (ensure it does not exceed bounds)
    end_smooth_row = max(0, start_row - num_smooth_pixels)

    # Smooth transition from the values in the start row to the constant value (e.g., 1)
    for i, row in enumerate(range(start_row, end_smooth_row - 1, -1)):
        weight = i / num_smooth_pixels  # Linear interpolation weight
        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value

    # Set all rows above the smoothing region to the constant value
    smoothed_image[:end_smooth_row, :] = constant_value

    return smoothed_image


#def smooth_from_row_to_constant(image, start_row, mask, constant_value=None, num_smooth_pixels=10):
#    """
#    Smoothly transitions from the values in a specified row to a constant value (e.g., 1)
#    upwards, over the specified number of rows, replacing any existing values.
#
#    Parameters:
#        image (2D array): The input image.
#        start_row (int): The row from which to start the smooth transition.
#        constant_value (float): The value to which the transition should lead (e.g., 1).
#        num_smooth_pixels (int): The number of rows over which to smooth the transition.
#
#    Returns:
#        smoothed_image (2D array): The image with a smooth transition upwards to the constant value,
#                                   replacing existing values.
#    """
#    # Copy the image to avoid modifying the original
#    smoothed_image = cp.copy(image)
#
#    #if constant_value is None:
#    #    constant_value = get_topmost_pixel(img=image, mask=mask)
#
#    # Ensure the start_row is within the bounds of the image
#    if start_row >= smoothed_image.shape[0] or start_row < 0:
#        raise ValueError(f"start_row {start_row} is out of bounds for the image height.")
#
#    # Get the values from the specified start row
#    values_at_start_row = smoothed_image[start_row, :]
#
#    # Calculate the range for smoothing (ensure it does not exceed bounds)
#    end_smooth_row = max(0, start_row - num_smooth_pixels)
#
#    # Smooth transition from the values in the start row to the constant value (e.g., 1)
#    for i, row in enumerate(range(start_row, end_smooth_row - 1, -1)):
#        weight = i / num_smooth_pixels  # Linear interpolation weight
#        smoothed_image[row, :] = (1 - weight) * values_at_start_row + weight * constant_value
#
#    # Set all rows below the smoothing region to the constant value
#    smoothed_image[end_smooth_row:, :] = constant_value
#
#    return smoothed_image


def apply_padding_refractive(image, data_dimensions:DataDimensions, padding_options:Padding, i0_log):
    if padding_options.padding_mode is Padding.PaddingMode.MIRROR_ALL:
        image = flip_and_pad(image,data_dimensions,padding_options)
    image.imag = image.imag - i0_log
    image = image.real * data_dimensions.window + 1j * image.imag * data_dimensions.window
    image.imag = image.imag + i0_log
    return image

def apply_non_negativity(values, psi_0, a0, iteration):

    # For static
    #values.real = cp.minimum(values.real, -26)
    #values.real = cp.maximum(values.real, -32)
    #ellipse_size_y = 0.25  #0.25 # 0.545
    #ellipse_size_x = 0.25  #0.25
    #transition_length_y = 40
    #transition_length_x = 40
    #fade_to_val = -30  # was None or -30
    #num_segments = None
    #values.real, _ = fadeoutImage(img=values.real,
    #                      fadeMethod='ellipse',
    #                      ellipseSize=[ellipse_size_y, ellipse_size_x],
    #                      transitionLength=[transition_length_y, transition_length_x],
    #                      fadeToVal=fade_to_val,
    #                      numSegments=num_segments)
    #values.imag = cp.zeros_like(values.real)

    # For dynamic
    #mask = create_circular_mask(size=values.real.shape[0], percentage=0.4, smooth_pixels=20) # was 0.525
    #values.real = extend_horizontally(values.real, mask, percentage_split=1.0) # 0.68
    # works great!

    phi = values.real  # Current phase values
    desired_max = -17 # Upper bound for the phase that is wanted
    current_max = cp.max(phi)  # Current maximum phase value
    # Shift entire map down so max hits -15
    phi_shifted = phi - (current_max - desired_max)
    # Now apply the min bound
    phi_clamped = cp.maximum(phi_shifted, -55) # was -55
    values.real = phi_clamped

    #if iteration > 501:
    #    values.real = smooth_from_row_to_constant_downward(values.real, 3800, constant_value=-19.55, num_smooth_pixels=100)
    #if iteration < 501:
    #    values.real = cp.minimum(values.real, 5)
    #    values.real = cp.maximum(values.real, -55)

    values.imag = cp.zeros_like(values.real) 

    #if iteration > 700:
    #    values.real[3850:, :] = -19.55

    # commented out
    #if iteration < 500:
    #    values.real = cp.minimum(values.real, -15)
    #    values.real = cp.maximum(values.real, -75)
    #else:
    #    values.real = cp.minimum(values.real, 75)
    #    values.real = cp.maximum(values.real, -75)
    #values.imag = cp.zeros_like(values.real)

    #ellipse_size_y = 0.555
    #ellipse_size_x = 0.555
    #transition_length_y = 20
    #transition_length_x = 20
    #fade_to_val = -19.5
    #num_segments = None
    #if iteration > 500:
    #    values.real, _ = fadeoutImage(img=values.real,
    #                      fadeMethod='ellipse',
    #                      ellipseSize=[ellipse_size_y, ellipse_size_x],
    #                      transitionLength=[transition_length_y, transition_length_x],
    #                      fadeToVal=fade_to_val,
    #                      numSegments=num_segments,
    #                      bottomApply=True)
    #values.imag = cp.maximum(values.imag,a0)
    #    values.real[1940:, :] = -19.5
    #values.imag = cp.zeros_like(values.real)
    return values



def apply_filter(values, filter_kernel_real, filter_kernel_imag):
    # Apply FFT to the real part
    values_real_fft = cp.fft.rfft2(values.real)
    values_real_fft *= filter_kernel_real
    values.real = cp.fft.irfft2(values_real_fft, s=values.real.shape)

    # Apply FFT to the imaginary part
    values_imag_fft = cp.fft.rfft2(values.imag)
    values_imag_fft *= filter_kernel_imag
    values.imag = cp.fft.irfft2(values_imag_fft, s=values.imag.shape)

    return values

def apply_window(values, window, intensities_log):
    values.imag = values.imag - intensities_log
    values *= window
    values.imag = values.imag + intensities_log

    return values
