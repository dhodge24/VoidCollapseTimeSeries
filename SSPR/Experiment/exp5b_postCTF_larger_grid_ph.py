"""

References:
    1) "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
    by D. Voelz et al. (for sampling criteria)

The purpose of this code is to match the size of the intensity image for forward and backward propagation. The
sampling conditions needs to be met to obtain the correct phase reconstruction result.

"""


import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage


save = True
extend_image = True
plot_result = True

N_pad = 6000  # Pad size to satisfy sampling criteria

# For the image extension - if True
ellipse_size_y = 0.4
ellipse_size_x = 0.4
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250

run_holo = "586"

# Directories with the data
dir_main = "/Users/danielhodge/Desktop/"
dir_exp = "run" + run_holo + "_exp_preprocessed/"

# Import CTF phase reconstruction
tiff_ph = "run" + run_holo + "_exp_phase_CTF3.tiff"

# Save file
tiff_phase_larger_grid = "run" + run_holo + "_exp_phase_CTF_larger_grid3.tiff"

# Import hologram intensity
ph = np.array(imread(dir_main + dir_exp + tiff_ph), dtype=np.float32)

ph = padToSize(img=ph, outputSize=[N_pad, N_pad], padMethod='replicate', padType='both', padValue=None)

if extend_image:
    ph, _ = fadeoutImage(img=ph,
                             fadeMethod='rectangle',
                             ellipseSize=[ellipse_size_y, ellipse_size_x],
                             transitionLength=[transition_length_y, transition_length_x],
                             fadeToVal=fade_to_val,
                             numSegments=num_segments,
                             bottomApply=False)

if plot_result:
    showImg(ph)

imwrite(dir_main + dir_exp + tiff_phase_larger_grid, ph)
