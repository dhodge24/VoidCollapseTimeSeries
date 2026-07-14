from SSPR.utilities import rotateImage, shiftRotateMagnifyImage, cropToCenter
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

run = "572"
type = "exp"
dir_main = "/Users/danielhodge/Desktop/time_series_recons_cropped/run" + run + "_" + type + "/"
shift = True
shift_y = 225
shift_x = 50

ph_gt = np.array(imread("/Users/danielhodge/Desktop/time_series_recons_cropped/run" + run + "_" + "sim" + "/" +
                        "run" + run + "_" + "sim" + "_ph.tiff"))
len_x = np.shape(ph_gt)[1]
len_y = np.shape(ph_gt)[0]
I_exp = np.array(imread("/Users/danielhodge/Desktop/" + "run" + run + "_" + type + "_I.tiff"))
I_exp = I_exp[abs(shift_y):len_y, 0:len_y-abs(shift_x)]
# I_exp = I_exp[0:len_y-abs(shift_y), 0:len_y-abs(shift_x)]
# I_exp = I_exp[abs(shift_y):len_y-100, abs(shift_x):]  # For run 590

imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_final.tiff", I_exp)

