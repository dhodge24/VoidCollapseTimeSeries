import numpy as np
import cupy as cp
from copy import deepcopy
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from utils.utilities import padToSize, cropToCenter

from utils.file_io import load_img_data
from parameters.Measurement import Measurement
from parameters.BeamSetup import BeamSetup
from parameters.Padding import Padding
from parameters.Options import Options
from parameters.DataDimensions import DataDimensions
from parameters.RecoParams import RecoParams
from base.single_reconstruction import single_reconstruction


dir_main = "/home/dhodge/PGD2/Images/"
dir_save = "tempImages/"
type_run = "sim"  # exp or sim
run = "574"

#tiff_I = "run306_exp_holos_with_speckle_FFC_extended_decon.tiff"
#tiff_ph = "run306_exp_static_ph_fADMM_CTF2.tiff"
#tiff_I = "run307_sim_holos_with_speckle_FFC_extended_decon_lowerpsf_LL.tiff"
#tiff_ph = "run307_sim_dynamic_ph_fADMM_CTF_lowerpsf_LL.tiff"


#tiff_I = "run" + run + "_" + type_run + "_holos_with_speckle_FFC_extended_decon_larger_grid.tiff"
#tiff_ph = "run" + run + "_" + type_run + "_phase_CTF_larger_grid2.tiff"

tiff_I = "I_perfect_larger.tiff"
tiff_ph = "ph_perfect_larger.tiff"

data_path = dir_main + tiff_I
ph_path = dir_main + tiff_ph

ph_guess = load_img_data(ph_path)

z01 = 120.41e-3
measurements = [Measurement(data_path=data_path,
                          data=load_img_data(data_path),
                          z01=z01)]  # 63.58816e-3

lens_mag = 4
z12 = 4.668995
dx = 6.5e-6
beam_setup = BeamSetup(energy=18000,
                       px_size=dx / lens_mag,
                       z02=z01 + z12)

padding_options = Padding(padding_mode=Padding.PaddingMode.NORMAL,
                          padding_factor=1,
                          down_sampling_factor=16,
                          cutting_band=0,
                          i0=1.0)


# static
options_upscale_2 = Options(iterations=700,
                            update_rate=1.1,
                            nesterov_momentum=1.0,
                            l2_weight_absorption=0.0,
                            omega_f_fwhm_phaseshift=1.0, 
                            omega_f_fwhm_absorption=0.0,
                            omega_ne_fwhm=8, # was 16 
                            padding=deepcopy(padding_options),
                            prototype_field=0.0)

options_mainrun_1 = Options(iterations=300,
                          update_rate=1.1,
                          nesterov_momentum=1.0,
                          l2_weight_absorption=0.0,
                          omega_f_fwhm_phaseshift=3.0, # 4
                          omega_f_fwhm_absorption=0.0,
                          omega_ne_fwhm=128, # 150
                          padding=deepcopy(padding_options),
                          prototype_field=0.0)

options_mainrun_2 = Options(iterations=500,
                          update_rate=1.1,
                          nesterov_momentum=1.0,
                          l2_weight_absorption=0.0,
                          omega_f_fwhm_phaseshift=1.0,
                          omega_f_fwhm_absorption=0.0,
                          omega_ne_fwhm=0, 
                          padding=deepcopy(padding_options),
                          prototype_field=0.0)

options_mainrun_3 = Options(iterations=500,
                          update_rate=1.1,
                          nesterov_momentum=1.0,
                          l2_weight_absorption=0.0,
                          omega_f_fwhm_phaseshift=1.0,
                          omega_f_fwhm_absorption=0.0,
                          omega_ne_fwhm=0,
                          padding=deepcopy(padding_options),
                          prototype_field=0.0)


data_dimensions = DataDimensions(total_size=(6000, 6000),  # Was (6000, 6000)
                                 fov_size=(2500, 2500),
                                 window_type='Blackman')

# Dynamic
#options_upscale_2.padding.down_sampling_factor=2
#options_mainrun_1.padding.down_sampling_factor=2
#options_mainrun_2.padding.down_sampling_factor=1
#options_mainrun_3.padding.down_sampling_factor=1

# Static
options_upscale_2.padding.down_sampling_factor=4
options_mainrun_1.padding.down_sampling_factor=2
options_mainrun_2.padding.down_sampling_factor=2
options_mainrun_3.padding.down_sampling_factor=1

########################################################################################################################

reco_params = RecoParams(beam_setup=beam_setup,
                           measurements=measurements,
                           reco_options=[options_upscale_2, options_mainrun_1, options_mainrun_2, options_mainrun_3],
                           data_dimensions=data_dimensions,
                           output_path='')

#reco_params = RecoParams(beam_setup=beam_setup,
#                           measurements=measurements,
#                           reco_options=[options_mainrun_3],
#                           data_dimensions=data_dimensions,
#                           output_path='')

result, loss_records = single_reconstruction(reco_params=reco_params, initial_guess=ph_guess)
# result, loss_records = single_reconstruction(reco_params=reco_params, initial_guess=None)
print(loss_records)


imwrite(dir_main + dir_save + "ph_final_run" + run + type_run  + "perf.tiff", cp.asnumpy(cp.real(result)))
imwrite(dir_main + dir_save + "mu_final_run" + run + type_run  + "perf.tiff", cp.asnumpy(cp.imag(result)))
np.savetxt(dir_main + dir_save + "loss_run" + run  + type_run  + "perf.txt", loss_records, fmt='%.14e', delimiter='\t', header='Loss',comments='')

