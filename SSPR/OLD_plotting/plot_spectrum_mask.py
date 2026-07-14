
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


run_wfs = "561"
run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/"
dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"

log_P = np.array(imread(dir_main + dir_holo_preprocessed + "log_power_spectrum.tiff"), dtype=np.float32)
masked_peaks = np.array(imread(dir_main + dir_holo_preprocessed + "masked_peaks.tiff"), dtype=np.float32)
refined_coordinates = np.loadtxt(dir_main + dir_holo_preprocessed + "refined_coordinates.txt")

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot images with colorbars
im0 = axs[0].imshow(log_P, clim=(0, 50), cmap="Greys_r")
axs[0].set_title('Log Power Spectrum', size=16, loc='center', fontweight='bold')

# Draw circles around the peaks and annotate
for j, (y, x) in enumerate(refined_coordinates):
    circle = plt.Circle((x, y), radius=50, color='red', fill=False)
    axs[0].add_artist(circle)

im1 = axs[1].imshow(masked_peaks, clim=(0, 50), cmap="Greys_r")
axs[1].set_title('Masked Peaks', size=16, fontweight='bold')

# Remove axes
for ax in axs.flatten():
    ax.axis("off")

# Define subplot labels
labels = ['(a)', '(b)']
x_pos, y_pos = 0.025, 0.95
for i, ax in enumerate(axs.flatten()):
    ax.text(x_pos, y_pos, labels[i], transform=ax.transAxes,
            fontsize=16, fontweight="bold", color='white', va="top", ha="left")

# ----------------
# COLORBARS
# ----------------
for im, ax in zip([im0, im1], axs.flatten()):
    cbar_ax = ax.inset_axes([0.80, 0.09, 0.15, 0.04])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=10, width=1, length=2, direction="out", pad=2, color='white', labelcolor='white')
    cbar.outline.set_edgecolor('white')

plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/power_spec_and_peaks.pdf', bbox_inches='tight', dpi=300, transparent=False)
plt.show()
