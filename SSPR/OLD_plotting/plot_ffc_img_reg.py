import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt

run_holo = "572"
type = "exp"

dir_main = "/Users/danielhodge/Desktop/all_runs/"
dir_holo_preprocessed = "run" + run_holo + "_" + type + "_preprocessed/"

# ffc = np.array(imread(dir_main + "run572_exp_holos_with_speckle_FFC.tiff"), dtype=np.float32)
ffc = np.array(imread(dir_main + dir_holo_preprocessed + "run572_exp_holos_with_speckle_FFC_extended_decon.tiff"), dtype=np.float32)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Plot image
im = ax.imshow(ffc, clim=(0, 2), cmap="Greys_r")
# ax.set_title('PCA Void Image', size=16, fontweight='bold')
ax.set_title('Deconvolved and Inpainted Void Image', size=16, fontweight='bold')
ax.axis("off")

# # Subplot label
# ax.text(0.025, 0.95, '(a)', transform=ax.transAxes,
#         fontsize=16, fontweight="bold", color='black',
#         va="top", ha="left")

# ----------------
# COLORBAR
# ----------------
cbar_ax = ax.inset_axes([0.80, 0.09, 0.15, 0.04])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=15, width=1, length=2,
                    direction="out", pad=2,
                    color='white', labelcolor='yellow')
cbar.outline.set_edgecolor('white')

plt.tight_layout()
# plt.savefig('/Users/danielhodge/Desktop/img_reg_ffc.pdf', dpi=300, transparent=False)
# plt.savefig('/Users/danielhodge/Desktop/exemplar_inpaint.pdf', dpi=300, transparent=False)
plt.show()
