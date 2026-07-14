import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from SSPR.utilities import shiftRotateMagnifyImage

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
# simulated mass density reconstruction
sim_path = f"/Users/danielhodge/Desktop/Inverse_Abel/run{run}_sim/run_{run}_sim_inverse_Abel_data_regularized_Gaussian_blurred8_mass_density.tiff"


# experimental mass density reconstruction
exp_path = f"/Users/danielhodge/Desktop/Inverse_Abel/run{run}_exp/run_{run}_exp_inverse_Abel_data_regularized_Gaussian_blurred8_mass_density.tiff"

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
# DISPLAY RANGE
# ----------------
clim_img = (0.0, 5.5)

# ----------------
# HELPER: MAKE HALF COMPOSITE
# left half = simulation
# right half = experiment
# ----------------
def make_half_composite(img_left, img_right):
    comp = np.zeros_like(img_left, dtype=np.float32)
    mid = img_left.shape[1] // 2
    comp[:, :mid] = img_left[:, :mid]
    comp[:, mid:] = img_right[:, mid:]
    return comp

# ----------------
# LOAD IMAGES
# ----------------
sim_img = np.array(imread(sim_path), dtype=np.float32)
exp_img = np.array(imread(exp_path), dtype=np.float32)

# if TIFF has leading frame dimension
if sim_img.ndim == 3:
    sim_img = sim_img[0]
if exp_img.ndim == 3:
    exp_img = exp_img[0]

# ----------------
# OPTIONAL TRANSFORM ON EXPERIMENT
# ----------------
exp_img = shiftRotateMagnifyImage(
    img=exp_img,
    magnify=[1, 1],
    rotAngleDegree=0,
    shifts=[-45, -50.4996321295979],
    padMethod='replicate'
)

# ----------------
# MAKE LEFT-RIGHT COMPOSITE
# ----------------
composite = make_half_composite(sim_img, exp_img)
# composite = make_half_composite(exp_img, sim_img)

# ----------------
# AXES IN MICRONS
# ----------------
ny, nx = composite.shape
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
    composite,
    cmap="inferno",
    vmin=clim_img[0],
    vmax=clim_img[1],
    interpolation="none",
    aspect="equal",
    extent=extent
)

# Divider line
ax.axvline(0, color="white", linewidth=1.0)

# ----------------
# AXIS CONTROL
# ----------------
if show_axes:
    ax.set_xlabel("x (µm)", fontsize=20)
    ax.set_ylabel("y (µm)", fontsize=20)
    ax.tick_params(labelsize=14)
else:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

# ----------------
# TOP LABELS
# ----------------
ax.text(
    0.25, 1.01, "Simulation\nRecon",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=22,
    fontweight="bold"
)

ax.text(
    0.75, 1.01, "Experiment\nRecon",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=22,
    fontweight="bold"
)

ax.text(
    0.50, 1.01, "|",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=22,
    fontweight="bold"
)

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
ax.text(
    -0.12, 1.05, "(j)",
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
cbar.set_ticks([0.0, 2.75, 5.5])
cbar.set_ticklabels(["0.0", "2.8", "5.5"])
cbar.ax.tick_params(labelsize=14)
cbar.set_label(
    "Mass Density ($\\mathbf{g/cm^3}$)",
    fontsize=16,
    fontweight="bold",
    labelpad=8
)

# ----------------
# LAYOUT + SAVE
# ----------------
plt.tight_layout()

save_path = f"/Users/danielhodge/Desktop/single_mass_density_sim_exp_composite_run{run}.pdf"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True
)

plt.show()