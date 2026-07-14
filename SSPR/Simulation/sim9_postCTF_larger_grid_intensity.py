import numpy as np
from tifffile import imread, imwrite


from SSPR.utilities import showImg, padToSize, fadeoutImage, reflect_image_2d
from SSPR.utilities_PyTorch import cropToCenter

save = True

run_holo = "586"

# Directories with the data
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = "run" + run_holo + "_sim/"
# dir_sim = "run" + run_holo + "_sim_test/"

# # File to import
# tiff_holo_with_speckle_ffc_extended_decon = "run" + run_holo + "_sim_holos_with_speckle_FFC_extended_decon.tiff"
#
# # File to save
# # tiff_holo_with_speckle_ffc_extended_decon_larger_grid = ("run" + run_holo +
# #                                                          "_sim_holos_with_speckle_FFC_extended_decon_larger_grid_test.tiff")
# tiff_holo_with_speckle_ffc_extended_decon_larger_grid = ("run" + run_holo +
#                                                          "_sim_holos_with_speckle_FFC_extended_decon_larger_grid.tiff")
#
# # Import hologram intensity
# I = np.array(imread(dir_main + dir_sim + tiff_holo_with_speckle_ffc_extended_decon), dtype=np.float32)


# Perfect image
tiff_holo_with_speckle_ffc_extended_decon = "/Users/danielhodge/Desktop/perfect.tiff"
tiff_holo_with_speckle_ffc_extended_decon_larger_grid = "run" + run_holo + "_larger_grid_perfect.tiff"
I = np.array(imread("/Users/danielhodge/Desktop/all_runs/run572_sim/run572_sim_holos_no_speckle.tiff"), dtype=np.float32)[0]


# I = np.array(imread("/Users/danielhodge/Desktop/I_perfect.tiff"), dtype=np.float32)[0]

I = padToSize(img=I, outputSize=[6000, 6000], padMethod='replicate', padType='both', padValue=None)
ellipse_size_y = 0.4
ellipse_size_x = 0.4
transition_length_y = 50
transition_length_x = 50
fade_to_val = 1.0
num_segments = None
I, _ = fadeoutImage(img=I,
                    fadeMethod='rectangle',
                    ellipseSize=[ellipse_size_y, ellipse_size_x],
                    transitionLength=[transition_length_y, transition_length_x],
                    fadeToVal=fade_to_val,
                    numSegments=num_segments,
                    bottomApply=False)

# showImg(I)

# imwrite(dir_main + dir_sim + tiff_holo_with_speckle_ffc_extended_decon_larger_grid, I)

imwrite("/Users/danielhodge/Desktop/I_perfect_larger.tiff", I)