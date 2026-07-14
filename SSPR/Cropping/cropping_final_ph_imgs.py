from SSPR.utilities import rotateImage, shiftRotateMagnifyImage, cropToCenter
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

run = "572"
type = "exp"
dir_main = "/Users/danielhodge/Desktop/all_runs/"
dir_sim = "run" + run + "_sim/"
dir_exp = "run" + run + "_exp_preprocessed/"
shift = True
shift_y = 225
shift_x = 50
angle_rotate = -2.5
if type == "sim":
    ph = np.array(imread(dir_main + "time_series_new_recons/run" + run + "_" + type + "_" + "phase.tiff"))[0]
    img = np.array(imread(dir_main + "time_series_new_recons/run" + run + "_" + type + "/ph_final_run" + run + type + "2.tiff"),
                   dtype=np.float32)
    I_sim = np.array(imread(dir_main + dir_sim + "run" + run + "_" + type + "_holos_with_speckle_FFC_extended_decon.tiff"))
    I_sim_raw = np.array(imread(dir_main + dir_sim + "run" + run + "_" + type + "_holos_with_speckle.tiff")[0])
else:
    # img = np.array(imread(dir_main + "time_series_new_recons/run" + run + "_" + type + "/ph_final_run" + run + type + "2.tiff"),
    #                dtype=np.float32)
    # I_exp = np.array(imread(dir_main + dir_exp + "run" + run + "_" + type + "_holos_with_speckle_FFC_extended_decon.tiff"))
    I_exp = np.array(imread(dir_main + dir_exp + "run" + run + "_" + type + "_holos_with_speckle_FFC_extended.tiff"))
    # I_exp_raw = np.array(imread(dir_main + dir_exp + "run" + run + "_" + type + "_preprocessed.tiff")[0])

if type == "sim":
    img = cropToCenter(img=img, newSize=[2100, 2100])
    if shift:
        img = shiftRotateMagnifyImage(img=img, shifts=[shift_y, shift_x], padMethod='constant')
    ph = cropToCenter(img=ph, newSize=[2100, 2100])
    if shift:
        ph = shiftRotateMagnifyImage(img=ph, shifts=[shift_y, shift_x], padMethod='constant')

    I_sim = cropToCenter(img=I_sim, newSize=[2100, 2100])
    if shift:
        I_sim = shiftRotateMagnifyImage(img=I_sim, shifts=[shift_y, shift_x], padMethod='constant')
        # I_sim[1950:] = 0
        # I_sim = shiftRotateMagnifyImage(img=I_sim, shifts=[100, 0], padMethod='constant')
    I_sim_raw = cropToCenter(img=I_sim_raw, newSize=[2100, 2100])
    if shift:
        I_sim_raw = shiftRotateMagnifyImage(img=I_sim_raw, shifts=[shift_y, shift_x], padMethod='constant')
        # I_sim_raw[1950:] = 0
        # I_sim_raw = shiftRotateMagnifyImage(img=I_sim_raw, shifts=[100, 0], padMethod='constant')

else:
    # img = rotateImage(img=img, rotAngleDegree=angle_rotate)
    # img = cropToCenter(img=img, newSize=[2100, 2100])
    # if shift:
    #     img = shiftRotateMagnifyImage(img=img, shifts=[shift_y, shift_x], padMethod='constant')

    I_exp = rotateImage(img=I_exp, rotAngleDegree=angle_rotate)
    I_exp = cropToCenter(img=I_exp, newSize=[2100, 2100])
    if shift:
        I_exp = shiftRotateMagnifyImage(img=I_exp, shifts=[shift_y, shift_x], padMethod='constant')

        # I_exp[0:175, :] = 0
        # I_exp[-75:, :] = 0
        # I_exp[:, 0:175] = 0

    # I_exp_raw = rotateImage(img=I_exp_raw, rotAngleDegree=angle_rotate)
    # I_exp_raw = cropToCenter(img=I_exp_raw, newSize=[2100, 2100])
    # if shift:
    #     I_exp_raw = shiftRotateMagnifyImage(img=I_exp_raw, shifts=[shift_y, shift_x], padMethod='constant')

        # I_exp_raw[0:175, :] = 0
        # I_exp_raw[-75:, :] = 0
        # I_exp_raw[:, 0:175] = 0

if type == "sim":
    plt.figure()
    plt.imshow(ph, cmap='Greys_r')
    plt.show()

    plt.figure()
    plt.imshow(I_sim, cmap='Greys_r')
    plt.show()
else:
    # plt.figure()
    # plt.imshow(img, cmap='Greys_r')
    # plt.show()

    plt.figure()
    plt.imshow(I_exp, cmap='Greys_r')
    plt.show()



# if type == "sim":
#     imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_raw.tiff", I_sim_raw)
# else:
#     imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I_raw.tiff", I_exp_raw)

if type == "sim":
    imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I.tiff", I_sim)
else:
    imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_I.tiff", I_exp)

# imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + ".tiff", img)
# if type == "sim":
#     imwrite("/Users/danielhodge/Desktop/run" + run + "_" + type + "_ph.tiff", ph)

