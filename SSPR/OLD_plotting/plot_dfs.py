
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


run_wfs = "561"
run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/"
dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"

dfx = np.array(imread(dir_main + "dfx_evt_2.tiff"), dtype=np.float32)
dfy = np.array(imread(dir_main + "dfy_evt_2.tiff"), dtype=np.float32)

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot images with colorbars
im0 = axs[0].imshow(dfx, clim=(-15, 15), cmap="Greys_r")
axs[0].set_title('Displacement Field X', size=16, loc='center', fontweight='bold')

im1 = axs[1].imshow(dfy, clim=(-15, 15), cmap="Greys_r")
axs[1].set_title('Displacement Field Y', size=16, fontweight='bold')

# Remove axes
for ax in axs.flatten():
    ax.axis("off")

# Define subplot labels
labels = ['(a)', '(b)']
x_pos, y_pos = 0.025, 0.95
for i, ax in enumerate(axs.flatten()):
    ax.text(x_pos, y_pos, labels[i], transform=ax.transAxes,
            fontsize=16, fontweight="bold", color='black', va="top", ha="left")

# ----------------
# COLORBARS
# ----------------
for im, ax in zip([im0, im1], axs.flatten()):
    cbar_ax = ax.inset_axes([0.80, 0.09, 0.15, 0.04])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8, width=1, length=2, direction="out", pad=2, color='white', labelcolor='yellow')
    cbar.outline.set_edgecolor('white')

plt.tight_layout()
plt.savefig('/Users/danielhodge/Desktop/dfxys_void.pdf', dpi=300, transparent=False)
plt.show()
