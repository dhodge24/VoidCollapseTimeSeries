import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import signal

# My own imports
from SSPR.utilities import cropToCenter, create_circular_mask


Ny, Nx = (2560, 2560)
E = 18000  # Initial energy of the beam in eV
lam = (1240 / E) * 1e-9  # Wavelength
z01 = 120.41e-3  # Distance from source to sample
z12 = 4.668995  # Distance from sample to detector
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01  # Magnification
z_eff = z12 / M  # Effective propagation distance
scale_fac = 4  # Lens magnification factor at scintillator
det_pixel_size = 6.5e-6  # Detector pixel size
dx_eff = det_pixel_size / M / scale_fac  # Effective pixel size in x
dy_eff = det_pixel_size / M / scale_fac  # Effective pixel size in y
extent_x = Nx * dx_eff  # Object domain length in x
extent_y = Ny * dy_eff  # Object domain length in y



I_raw = imread("/Users/danielhodge/Desktop/Run_572_raw.tiff")
W_raw = imread("/Users/danielhodge/Desktop/Run_561_raw.tiff")
D = imread("/Users/danielhodge/Desktop/AVG_run381_darks.tiff")



# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# fig.text(0.015, 0.73, 'xRAGE Simulation', fontsize=16, fontweight='bold', rotation=90, ha="center", va='center')
# fig.text(0.012, 0.25, "Experiment", fontsize=16, fontweight='bold', rotation=90, ha="center", va="center")

# Plot images with colorbars
im0 = axs[0].imshow(I_raw, clim=(0, 1000), cmap="Greys_r")
axs[0].set_title('Raw Dynamic XPC Image', size=16, loc='center', fontweight='bold')

im1 = axs[1].imshow(W_raw, clim=(0, 2500), cmap="Greys_r")
axs[1].set_title('Raw White Field Image', size=16, fontweight='bold')

im2 = axs[2].imshow(D, clim=(0, 300), cmap="Greys_r")
axs[2].set_title('Averaged Dark Field Image', size=16, fontweight='bold')

# Remove axes
for ax in axs.flatten():
    ax.axis("off")

# Define subplot labels
labels = ['(a)', '(b)', '(c)']
x_pos, y_pos = 0.025, 0.95
for i, ax in enumerate(axs.flatten()):
    ax.text(x_pos, y_pos, labels[i], transform=ax.transAxes,
            fontsize=16, fontweight="bold", color='white', va="top", ha="left")

# ----------------
# SCALE BARS
# ----------------
for ax in axs.flatten():
    scalebar = ScaleBar(dx=dx_eff, units='m', location='upper right',
                        length_fraction=0.3, height_fraction=0.025,
                        box_alpha=0, color='white', font_properties={"size": 16})
    ax.add_artist(scalebar)

# ----------------
# COLORBARS
# ----------------
for im, ax in zip([im0, im1, im2], axs.flatten()):
    cbar_ax = ax.inset_axes([0.80, 0.09, 0.15, 0.04])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=10, width=1, length=2, direction="out", pad=2, color='white', labelcolor='white')
    cbar.outline.set_edgecolor('white')

plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/tempPlot.pdf', dpi=300, transparent=False)
plt.show()
