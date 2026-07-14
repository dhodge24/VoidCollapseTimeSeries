import cupy as cp

from parameters.RecoParams import RecoParams
from reconstruct_multistep import reconstruct
from utils.utilities import cropToCenter


def single_reconstruction(reco_params: RecoParams, initial_guess):
    reco_params.measurements[0].data = cp.sqrt(reco_params.measurements[0].data)

    x_predicted, se_losses_all, fov = reconstruct(measurements=reco_params.measurements,
                                                  beam_setup=reco_params.beam_setup,
                                                  options=reco_params.reco_options,
                                                  data_dimensions=reco_params.data_dimensions,
                                                  initial_guess=initial_guess)

    x_predicted = cropToCenter(x_predicted, fov)

    return x_predicted, se_losses_all
