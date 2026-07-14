import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage, reflect_image_2d
from SSPR.utilities_PyTorch import cropToCenter

save = True

# Run number
run_holo = "572"

# # Directories with the data
# dir_main = "/Users/danielhodge/Desktop/"
# dir_sim = "run" + run_holo + "_sim/"
# # dir_sim = "run" + run_holo + "_sim_test/"
#
# # Import CTF phase reconstruction
# tiff_ph = "run" + run_holo + "_sim_phase_CTF2.tiff"
#
# # Save file
# # tiff_phase_larger_grid = "run" + run_holo + "_sim_phase_CTF_larger_grid_test.tiff"
# tiff_phase_larger_grid = "run" + run_holo + "_sim_phase_CTF_larger_grid.tiff"
#
# # Import hologram intensity
# ph = np.array(imread(dir_main + dir_sim + tiff_ph), dtype=np.float32)

ph = np.array(imread("/Users/danielhodge/Desktop/run572_sim_test_no_deconvolve.tiff"))

# ph = np.array(imread("/Users/danielhodge/Desktop/perfect.tiff"), dtype=np.float32)
ph = padToSize(img=ph, outputSize=[6000, 6000], padMethod='replicate', padType='both', padValue=None)

ellipse_size_y = 0.4
ellipse_size_x = 0.4
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250
ph, _ = fadeoutImage(img=ph,
                         fadeMethod='rectangle',
                         ellipseSize=[ellipse_size_y, ellipse_size_x],
                         transitionLength=[transition_length_y, transition_length_x],
                         fadeToVal=fade_to_val,
                         numSegments=num_segments,
                         bottomApply=False)

showImg(ph)

# imwrite(dir_main + dir_sim + tiff_phase_larger_grid, ph)
imwrite("/Users/danielhodge/Desktop/run572_ph_larger_no_deconvolve.tiff", ph)