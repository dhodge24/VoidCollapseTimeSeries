import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite

from utilities import create_circular_mask



img = np.array(imread("/Users/danielhodge/Desktop/IFESTAR_imgs/run582_exp_evt_1_preprocessed"))
mask = create_circular_mask(size=img.shape[0], percentage=0.9, smooth_pixels=1)

out = img * mask

plt.figure()
plt.imshow(out, clim=(0, 500))
plt.show()

imwrite("/Users/danielhodge/Desktop/run582_exp_preprocessed_cropped.tiff", out)
