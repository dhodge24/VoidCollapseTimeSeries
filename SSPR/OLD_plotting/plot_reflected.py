
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


run_wfs = "561"
run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/"
# dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"

wf = np.array(imread(dir_main + "run561_exp_preprocessed_no_Talbot-1.tif"), dtype=np.float32)
holo = np.array(imread(dir_main + "run572_exp_preprocessed_no_Talbot.tiff"), dtype=np.float32)[0]

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot images with colorbars
im0 = axs[0].imshow(wf, clim=(0, 1800), cmap="Greys_r")
axs[0].set_title('White Field Image Reflected', size=16, loc='center', fontweight='bold')

im1 = axs[1].imshow(holo, clim=(0, 600), cmap="Greys_r")
axs[1].set_title('Void Image Reflected', size=16, fontweight='bold')

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
plt.savefig('/Users/danielhodge/Desktop/reflected_images.pdf', dpi=300, transparent=False)
plt.show()
