"""

The purpose of this code is to make areas outside the circular area containing the object equal to the average of the
outer segments of the object -- an object extension. Leaving the values at 0 outside the circular area caused phase
retrieval algorithms to fail.


"""

import numpy as np
from tifffile import imread, imwrite
from SSPR.utilities import fadeoutImage, showImg


save = True
use_PCA_corrected = True
use_imreg_corrected = False
extend_img = False

run_wfs = "561"
run_holo = "586"

# Main directories
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = "run" + run_holo + "_sim/"
# dir_sim = "run" + run_holo + "_sim_test/"


# Files to import
tiff_holo_with_speckle_ffc = "run" + run_holo + "_sim_holos_with_speckle_FFC.tiff"

# Files to save
tiff_holo_with_speckle_ffc_extended = "run" + run_holo + "_sim_holos_with_speckle_FFC_extended.tiff"

ellipse_size_y = 0.9
ellipse_size_x = 0.9
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250

# Read in ffc image
ffc_img = np.array(imread(dir_main + dir_sim + tiff_holo_with_speckle_ffc))

# ffc_img[np.isnan(ffc_img) | np.isinf(ffc_img)] = 0
# ffc_img[ffc_img < 0] = 0
# ffc_img[ffc_img > 20] = 1

if extend_img:
    ffc_img, _ = fadeoutImage(img=ffc_img,
                              fadeMethod='ellipse',
                              ellipseSize=[ellipse_size_y, ellipse_size_x],
                              transitionLength=[transition_length_y, transition_length_x],
                              fadeToVal=fade_to_val,
                              numSegments=num_segments,
                              bottomApply=False)

    showImg(ffc_img, clim=(0, 3))

if save:
    imwrite(dir_main + dir_sim + tiff_holo_with_speckle_ffc_extended, ffc_img)
