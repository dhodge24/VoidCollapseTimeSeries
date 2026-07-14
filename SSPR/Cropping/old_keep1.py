from SSPR.utilities import rotateImage, shiftRotateMagnifyImage, cropToCenter
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

run = "572"
type = "sim"
dir_main = "/Users/danielhodge/Desktop/time_series_Recons_cropped/run" + run + "_" + type + "/"
shift = True
shift_y = 200
shift_x = -200

if type == "sim":
    ph = np.array(imread(dir_main + "run" + run + "_" + type + ".tiff"))
    ph_gt = np.array(imread(dir_main + "run" + run + "_" + type + "_ph.tiff"))
    # mu_gt = np.array(imread(dir_main + "run" + run + "_" + type + "_mu.tiff"))
    len_x = np.shape(ph_gt)[1]
    len_y = np.shape(ph_gt)[0]
    I_sim = np.array(imread(dir_main + "run" + run + "_" + type + "_I.tiff"))
    I_sim_raw = np.array(imread(dir_main + "run" + run + "_" + type + "_I_raw.tiff"))
    ph = ph[abs(shift_y):len_y-100, abs(shift_x):]
    ph_gt = ph_gt[abs(shift_y):len_y-100, abs(shift_x):]
    I_sim = I_sim[abs(shift_y):len_y-100, abs(shift_x):]
    I_sim_raw = I_sim_raw[abs(shift_y):len_y-100, abs(shift_x):]

else:
    ph = np.array(imread(dir_main + "run" + run + "_" + type + ".tiff"))
    len_x = np.shape(ph)[1]
    len_y = np.shape(ph)[0]
    I_exp = np.array(imread(dir_main + "run" + run + "_" + type + "_I.tiff"))
    I_exp_raw = np.array(imread(dir_main + "run" + run + "_" + type + "_I_raw.tiff"))
    ph = ph[abs(shift_y):len_y-100, abs(shift_x):]
    I_exp = I_exp[abs(shift_y):len_y-100, abs(shift_x):]
    I_exp_raw = I_exp_raw[abs(shift_y):len_y-100, abs(shift_x):]


plt.figure()
plt.imshow(ph, cmap='Greys_r')
plt.show()

imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_ph_final.tiff", ph)
if type == "sim":
    # imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_raw_final.tiff", I_sim_raw)

    imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_final.tiff", I_sim)
    imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_ph_gt_final.tiff", ph_gt)
else:
    imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_final.tiff", I_exp)

    # imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_raw_final.tiff", I_exp_raw)