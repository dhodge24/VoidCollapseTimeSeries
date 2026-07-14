from SSPR.utilities import rotateImage, shiftRotateMagnifyImage, cropToCenter
from tifffile import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

run = "572"
type = "sim"
dir_main = "/Users/danielhodge/Desktop/"
dir_sim = "run" + run + "_sim/"
dir_exp = "run" + run + "_exp_preprocessed/"
shift = True
shift_y = 225
shift_x = 50
angle_rotate = 0

ph = np.array(imread("/Users/danielhodge/Desktop/ph_final_run572simNODECONVOLVE.tiff"), dtype=np.float32)


ph = cropToCenter(img=ph, newSize=[2100, 2100])
if shift:
    ph = shiftRotateMagnifyImage(img=ph, shifts=[shift_y, shift_x], padMethod='constant')
    ph = cropToCenter(img=ph, newSize=[2100, 2100])

# if type == "sim":
#     plt.figure()
#     plt.imshow(ph, cmap='Greys_r')
#     plt.show()


# ph = np.array(imread("/Users/danielhodge/Desktop/ph_final_run572simPERFECT.tiff"), dtype=np.float32)

len_x = 2100
len_y = 2100
ph = ph[abs(shift_y):len_y, 0:len_y-abs(shift_x)]

imwrite("/Users/danielhodge/Desktop/ph_final_run572simNODECONVOLVE2.tiff", ph)

plt.figure()
plt.imshow(ph)
plt.show()


