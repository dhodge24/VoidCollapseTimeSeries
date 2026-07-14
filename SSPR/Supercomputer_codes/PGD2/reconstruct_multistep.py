
from typing import List
import cupy as cp
#from skimage.transform import resize
from cupyx.scipy.ndimage import zoom

from parameters.Padding import Padding
from preprocessing.process_padding_options import process_padding_options
from utils.utilities import cropToCenter

from parameters.Measurement import Measurement
from parameters.BeamSetup import BeamSetup
from parameters.Options import Options
from parameters.DataDimensions import DataDimensions

from reconstruct import reconstruct as reconstruct_singlemode


def resize_guess(x_predicted, nesterov_vt, new_size):
    """
    Resize the real and imaginary parts of complex arrays x_predicted and nesterov_vt using CuPy,
    ensuring they are cast to float32.
    
    Parameters:
        x_predicted (cp.ndarray): Input complex array to resize.
        nesterov_vt (cp.ndarray or None): Optional input complex array to resize.
        new_size (tuple): Desired output shape.

    Returns:
        tuple: Resized x_predicted and nesterov_vt.
    """
    def resize_cupy(array, new_size):
        # Ensure the input is float32
        array = array.astype(cp.float64)
        # Compute zoom factors for each dimension
        zoom_factors = [out_dim / in_dim for out_dim, in_dim in zip(new_size, array.shape)]
        # Apply zoom
        return zoom(array, zoom_factors, mode='reflect')

    # Ensure x_predicted is cast to complex64 (real and imaginary parts are float32)
    x_predicted = x_predicted.astype(cp.complex128)
    
    # Resize the real and imaginary parts separately for x_predicted
    x_predicted_real_resized = resize_cupy(x_predicted.real, new_size)
    x_predicted_imag_resized = resize_cupy(x_predicted.imag, new_size)
    x_predicted = x_predicted_real_resized + 1j * x_predicted_imag_resized

    # If nesterov_vt is None, return immediately
    if nesterov_vt is None:
        return x_predicted, nesterov_vt

    # Ensure nesterov_vt is cast to complex64 (real and imaginary parts are float32)
    nesterov_vt = nesterov_vt.astype(cp.complex128)
    
    # Resize the real and imaginary parts separately for nesterov_vt
    nesterov_vt_real_resized = resize_cupy(nesterov_vt.real, new_size)
    nesterov_vt_imag_resized = resize_cupy(nesterov_vt.imag, new_size)
    nesterov_vt = nesterov_vt_real_resized + 1j * nesterov_vt_imag_resized

    return x_predicted, nesterov_vt

#def resize_guess(x_predicted, nesterov_vt, new_size):
#    # Resize the real and imaginary parts separately for x_predicted
#    x_predicted_real_resized = resize(x_predicted.real, new_size, mode='reflect', anti_aliasing=True)
#    x_predicted_imag_resized = resize(x_predicted.imag, new_size, mode='reflect', anti_aliasing=True)
#    x_predicted = x_predicted_real_resized + 1j * x_predicted_imag_resized
#
#    # If nesterov_vt is None, return immediately
#    if nesterov_vt is None:
#        return x_predicted, nesterov_vt
#
#    # Resize the real and imaginary parts separately for nesterov_vt
#    nesterov_vt_real_resized = resize(nesterov_vt.real, new_size, mode='reflect', anti_aliasing=True)
#    nesterov_vt_imag_resized = resize(nesterov_vt.imag, new_size, mode='reflect', anti_aliasing=True)
#    nesterov_vt = nesterov_vt_real_resized + 1j * nesterov_vt_imag_resized
#
#    return x_predicted, nesterov_vt


def prepare_next_iteration(total_iter,current_iter,x_predicted,nesterov_vt,measurements,beam_setup,support,options):
    measurements_preprocessed, beam_setup_preprocessed, support_preprocessed = process_padding_options(measurements,
                                                                                                       beam_setup,
                                                                                                       support,
                                                                                                       options[current_iter].padding
                                                                                                       )


    x_predicted, nesterov_vt = resize_guess(x_predicted, nesterov_vt, support_preprocessed.total_size)

    return x_predicted,nesterov_vt,measurements_preprocessed,beam_setup_preprocessed,support_preprocessed


def reconstruct(measurements:List[Measurement], beam_setup:BeamSetup, options:List[Options], data_dimensions:DataDimensions, initial_guess):

    measurements_preprocessed, beam_setup_preprocessed, support_preprocessed = process_padding_options(measurements,
                                                                                                       beam_setup,
                                                                                                       data_dimensions,
                                                                                                       options[0].padding
                                                                                                       )

    if initial_guess is None:
        x_predicted = cp.zeros(support_preprocessed.total_size, dtype=cp.complex128)
    else:
        size = support_preprocessed.total_size
        x_predicted = cp.array(initial_guess, dtype=cp.complex128)
        zoom_factors = (size[1] / x_predicted.shape[0], size[0] / x_predicted.shape[1])
        # Resize real and imaginary parts separately using zoom
        x_predicted_real_resized = zoom(x_predicted.real, zoom_factors, mode='reflect')
        x_predicted_imag_resized = zoom(x_predicted.imag, zoom_factors, mode='reflect')
        #x_predicted_real_resized = resize(x_predicted.real, (size[1], size[0]), mode='reflect', anti_aliasing=True)
        #x_predicted_imag_resized = resize(x_predicted.imag, (size[1], size[0]), mode='reflect', anti_aliasing=True)
        x_predicted = x_predicted_real_resized + 1j * x_predicted_imag_resized

    nesterov_vt = cp.zeros(x_predicted.shape, dtype=cp.complex128)

    current_iter_offset = 0

    se_losses_all = cp.zeros(0, dtype=cp.float64)
    for i in range(len(options)):
        if i>0:
            x_predicted, nesterov_vt, measurements_preprocessed, beam_setup_preprocessed, support_preprocessed \
                = prepare_next_iteration(len(options), i, x_predicted, nesterov_vt, measurements, beam_setup, data_dimensions, options)

        x_predicted, se_losses, nesterov_vt = reconstruct_singlemode(measurements_preprocessed, beam_setup_preprocessed,
                                                                     options[i], support_preprocessed, x_predicted,
                                                                     current_iter_offset, nesterov_vt)
        se_losses_all = cp.concatenate((se_losses_all,se_losses))
        current_iter_offset = current_iter_offset + options[i].iterations

    x_predicted.imag = x_predicted.imag + cp.log(options[-1].padding.i0)


    return x_predicted, se_losses_all, support_preprocessed.fov_size
