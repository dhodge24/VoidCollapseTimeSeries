import os
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# RUN TO PLOT
# ----------------
run = "582"
delay_time = 10.52

# ----------------
# TOGGLES
# ----------------
show_axes = False

# ----------------
# PATHS
# ----------------
img_path = f"/Users/danielhodge/Desktop/GT_adjusted_maps/run{run}_GT_maps/density_total_GT_adjusted.tiff"

# ----------------
# CONSTANTS (same geometry)
# ----------------
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac
dx_um = dx_eff * 1e6

# ----------------
# DISPLAY RANGE (adjust this!)
# ----------------
clim_img = (0, 5.5)   # <-- change based on density range

# ----------------
# LOAD IMAGE
# ----------------
img = np.array(imread(img_path), dtype=np.float32)

# ----------------
# AXES IN MICRONS
# ----------------
ny, nx = img.shape
extent = [
    -nx * dx_um / 2,
     nx * dx_um / 2,
     ny * dx_um / 2,
    -ny * dx_um / 2
]

# ----------------
# PLOT
# ----------------
fig, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(
    img,
    cmap="seismic",
    vmin=clim_img[0],
    vmax=clim_img[1],
    interpolation="none",
    aspect="equal",
    extent=extent
)

# ----------------
# AXIS CONTROL
# ----------------
if show_axes:
    ax.set_xlabel("x (µm)", fontsize=20)
    ax.set_ylabel("y (µm)", fontsize=20)
    ax.tick_params(labelsize=16)
else:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

# ----------------
# TITLE
# ----------------
ax.set_title("xRAGE Density", fontsize=22, fontweight="bold")

# ----------------
# SCALE BAR
# ----------------
scalebar = ScaleBar(
    dx=1,
    units="µm",
    fixed_value=25,
    location="upper right",
    height_fraction=0.008,
    width_fraction=0.035,
    box_alpha=1.0,
    box_color="white",
    color="black",
    font_properties={"size": 20}
)
ax.add_artist(scalebar)

# ----------------
# PANEL LABEL
# ----------------
panel_label = "(a)"

ax.text(
    -0.12, 1.05, panel_label,
    transform=ax.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top",
    ha="left",
    clip_on=False
)

# ----------------
# COLORBAR (BOTTOM)
# ----------------
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.1)

cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

# --- 3 ticks ---
vmin, vmax = clim_img
vmid = 0.5 * (vmin + vmax)

cbar.set_ticks([vmin, vmid, vmax])
cbar.set_ticklabels([f"{vmin:.1f}", f"{vmid:.1f}", f"{vmax:.1f}"])

# Tick styling
cbar.ax.tick_params(labelsize=14)

# Label (now horizontal)
cbar.set_label("Mass Density ($\\mathbf{g/cm^3}$)", fontsize=16, fontweight="bold")

# ----------------
# LAYOUT + SAVE
# ----------------
plt.tight_layout()

save_path = f"/Users/danielhodge/Desktop/single_GT_density_run{run}.pdf"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True
)

plt.show()