from copy import deepcopy
import cupy as cp
from typing import List
from cupyx.scipy.ndimage import fourier_gaussian
from tqdm import tqdm


from models.FresnelPropagator import FresnelPropagator
from gradients.analytical import get_gradient
from constraints import regularization
from parameters.Measurement import Measurement
from parameters.DataDimensions import DataDimensions
from parameters.Options import Options
from parameters.BeamSetup import BeamSetup


def reconstruct(measurements: List[Measurement], beam_setup: BeamSetup, options: Options,
                data_dimensions: DataDimensions, init_guess: cp.ndarray, iter_offset, nesterov_vt):
    oref_predicted = deepcopy(init_guess)
    oref_predicted = cp.asarray(oref_predicted, dtype=cp.complex128)
    se_loss_records = cp.zeros(options.iterations, dtype=cp.float64)

    zeros = cp.zeros(oref_predicted.shape, dtype=cp.float64)
    ones = cp.ones(oref_predicted.shape,  dtype=cp.float64)

    i0_log = -cp.log(options.padding.i0) * cp.ones(oref_predicted.shape, dtype=cp.float64)
    model = FresnelPropagator(measurements, beam_setup, oref_predicted.shape, oref_predicted)

    # Filter kernel for omega_ne_fwhm
    if options.omega_ne_fwhm is not None:
        filter_kernel_vt = fourier_gaussian(ones, sigma=options.omega_ne_fwhm / 2.35)[:,
                           0:int(oref_predicted.shape[1] / 2) + 1]
    else:
        filter_kernel_vt = ones[:, 0:int(oref_predicted.shape[1] / 2) + 1]

    # Filter kernel for omega_f_fwhm_phaseshift
    if options.omega_f_fwhm_phaseshift != 0.0:
        filter_kernel_obj_phase = fourier_gaussian(ones, sigma=options.omega_f_fwhm_phaseshift / 2.35)[:,
                                  0:int(oref_predicted.shape[1] / 2) + 1]
    else:
        filter_kernel_obj_phase = ones[:, 0:int(oref_predicted.shape[1] / 2) + 1]

    # Filter kernel for omega_f_fwhm_absorption
    if options.omega_f_fwhm_absorption != 0.0:
        filter_kernel_obj_absorption = fourier_gaussian(ones, sigma=options.omega_f_fwhm_absorption / 2.35)[:,
                                       0:int(oref_predicted.shape[1] / 2) + 1]
    else:
        filter_kernel_obj_absorption = ones[:, 0:int(oref_predicted.shape[1] / 2) + 1]

    for iteration in tqdm(range(options.iterations)):
        oref_predicted = regularization.apply_padding_refractive(oref_predicted, data_dimensions, options.padding,
                                                                 i0_log)

        nesterov_vt = regularization.apply_filter(nesterov_vt, filter_kernel_vt, filter_kernel_vt)
        oref_predicted = regularization.apply_filter(oref_predicted, filter_kernel_obj_phase,
                                                     filter_kernel_obj_absorption)

        oref_predicted_old = oref_predicted
        oref_predicted += -options.nesterov_momentum * nesterov_vt
        grad, loss = get_gradient(
            model=model,
            measurements=measurements,
            data_dimensions=data_dimensions,
            oref_predicted=oref_predicted)

        nesterov_vt = options.nesterov_momentum * nesterov_vt + options.update_rate * grad

        if options.l2_weight_absorption != 0.0:
            nesterov_vt.imag += options.l2_weight_absorption * (
                    oref_predicted.imag / (cp.linalg.norm(oref_predicted.imag, ord=2) + 1e-6)
            )

        oref_predicted = oref_predicted_old - nesterov_vt
        oref_predicted = regularization.apply_non_negativity(oref_predicted, zeros, i0_log, iteration)
        se_loss_records[iteration] = loss

    return oref_predicted, se_loss_records, nesterov_vt

