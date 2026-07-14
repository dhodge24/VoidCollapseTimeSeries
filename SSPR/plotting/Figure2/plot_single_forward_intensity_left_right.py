
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from SSPR.utilities import padToSize, cropToCenter, shiftRotateMagnifyImage

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
sim_path = f"/Users/danielhodge/Desktop/all_runs/run{run}_sim/run{run}_sim_holos_with_speckle_orig.tiff"
exp_path = f"/Users/danielhodge/Desktop/run{run}_exp/Run_{run}_evt_1_Zyla_0.tiff"

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
clim_img = (0, 600)

# ----------------
# HELPER: MAKE HALF COMPOSITE
# left half = simulation
# right half = experimental
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
sim_img = np.array(imread(sim_path), dtype=np.float32)[0]
exp_img = np.array(imread(exp_path), dtype=np.float32)

# ----------------
# TARGET SIZE
# ----------------
target_y, target_x = 2500, 2500

# ----------------
# PREP SIM IMAGE
# ----------------
# If needed, force sim to 2500 x 2500
if sim_img.shape[0] > target_y or sim_img.shape[1] > target_x:
    sim_img = cropToCenter(sim_img, [min(sim_img.shape[0], target_y), min(sim_img.shape[1], target_x)])

if sim_img.shape != (target_y, target_x):
    sim_img = padToSize(
        img=sim_img,
        outputSize=[target_y, target_x],
        padMethod='constant',
        padType='both',
        padValue=0
    )

# ----------------
# PREP EXP IMAGE
# original exp shape is (2160, 2560)
# Step 1: crop x from 2560 -> 2500 while keeping y = 2160
# Step 2: pad y from 2160 -> 2500
# ----------------
if exp_img.shape[1] > target_x:
    exp_img = cropToCenter(exp_img, [exp_img.shape[0], target_x])

if exp_img.shape[0] < target_y or exp_img.shape[1] < target_x:
    exp_img = padToSize(
        img=exp_img,
        outputSize=[target_y, target_x],
        padMethod='constant',
        padType='both',
        padValue=0
    )

# ----------------
# ROTATE EXPERIMENT CLOCKWISE BY 4.5 DEGREES
# If it looks backward, change -4.5 to +4.5
# ----------------
exp_img = shiftRotateMagnifyImage(
    img=exp_img,
    magnify=[1, 1],
    rotAngleDegree=-4.5,
    shifts=[0, -50.4996321295979],
    padMethod='replicate'
)


# ----------------
# MAKE LEFT-RIGHT COMPOSITE
# ----------------
composite = make_half_composite(sim_img, exp_img)

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
# TITLE
# ----------------
# ax.set_title("Simulation | Experiment", fontsize=22, fontweight="bold")

# ----------------
# TOP LABELS
# ----------------
ax.text(
    0.25, 1.01, "Simulation",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=22,
    fontweight="bold"
)

ax.text(
    0.75, 1.01, "Experiment",
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
    -0.12, 1.05, "(e)",
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
cbar.set_ticks([0, 300, 600])
cbar.set_ticklabels(["0", "300", "600"])
cbar.ax.tick_params(labelsize=14)
cbar.set_label(
    "Intensity",
    fontsize=16,
    fontweight="bold",
    labelpad=8
)

# ----------------
# LAYOUT + SAVE
# ----------------
plt.tight_layout()

save_path = f"/Users/danielhodge/Desktop/single_sim_exp_composite_run{run}.pdf"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    transparent=True
)

plt.show()