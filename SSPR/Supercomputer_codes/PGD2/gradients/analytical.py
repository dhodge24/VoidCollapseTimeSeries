import cupy as cp
from typing import List

from utils.utilities import create_circular_mask, fadeoutImage
from parameters.DataDimensions import DataDimensions
from parameters.Measurement import Measurement

# this function is not used, but here in case
def extend_horizontally(image, mask, percentage_split=1.0):
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

def get_gradient(model, measurements:List[Measurement], data_dimensions:DataDimensions, oref_predicted):
    oref_predicted = oref_predicted.astype(cp.complex128)
    obj = cp.exp(1j*oref_predicted)
    #mask = create_circular_mask(obj.shape[0], percentage=0.56, smooth_pixels=1)
    obj_conj = cp.conj(obj)

    obj_propagated = model.propagate_forward_all(obj)
    measurements_predicted = model.get_measurements_from_propagated_all(obj_propagated)

    #mask = create_circular_mask(measurements_predicted[0].shape[0], percentage=0.525, smooth_pixels=1)
    #measurements_predicted = extend_horizontally(measurements_predicted[0], mask, percentage_split=1.0) 

    distance=0
    fraction = cp.sqrt(measurements[distance].data / measurements_predicted[distance])
    #fraction = cp.sqrt(measurements[distance].data / measurements_predicted)
    temp_propagated = obj_propagated[distance]  - obj_propagated[distance] * fraction

    temp = model.propagate_back(temp_propagated, distance)
    grad = -1j * obj_conj * temp
    loss = cp.sum(cp.abs(measurements_predicted[distance] - measurements[distance].data) ** 2) 
    #loss = cp.sum(cp.abs(measurements_predicted - measurements[distance].data) ** 2)

    N = (data_dimensions.fov_range_raw[0][1] - data_dimensions.fov_range_raw[0][0]) * (data_dimensions.fov_range_raw[1][1] - data_dimensions.fov_range_raw[1][0])
    return grad, loss/N
