import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift

# ----------------
# RUNS / DELAYS
# ----------------
runs = ["572", "576", "580", "582", "590"]
delay_times = [6.82, 8.12, 9.67, 10.52, 13.42]

# ----------------
# OPTIONAL SHIFTS FOR GT ALIGNMENT
# Put the shift you want for each run here.
# Format: (shift_y, shift_x)
# If no shift is needed, leave as (0, 0)
# ----------------
gt_shifts = {
    "572": (0, 25.000069739608307),
    "576": (0, -54.99983132112288 ),
    "580": (0, 75.49991558517377),
    "582": (0, 50.4996321295979),
    "590": (0, -49.99998557735444 ),
}

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_data = "time_series_recons_cropped/"

# ----------------
# CONSTANTS
# ----------------
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac

# ----------------
# HELPER: MAKE HALF COMPOSITE
# left half = recon
# right half = ground truth
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp

# ----------------
# LOAD DATA AND BUILD COMPOSITES
# ----------------
sim_composites = []
exp_composites = []

for run in runs:
    dir_sim = dir_main + dir_data + f"run{run}_sim/"
    dir_exp = dir_main + dir_data + f"run{run}_exp/"

    gt_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_gt_final.tiff"), dtype=np.float32)
    sim_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_final.tiff"), dtype=np.float32)
    exp_phase = np.array(imread(dir_exp + f"run{run}_exp_ph_final.tiff"), dtype=np.float32)

    # shift GT if desired
    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    gt_phase_shifted = shift(gt_phase, shift=(shift_y, shift_x), order=3, mode='nearest')

    sim_comp = make_half_composite(sim_phase, gt_phase_shifted)
    exp_comp = make_half_composite(exp_phase, gt_phase_shifted)

    sim_composites.append(sim_comp)
    exp_composites.append(exp_comp)

# ----------------
# ORGANIZE IMAGES
# rows = delays
# cols = [sim composite, exp composite]
# ----------------
images_grid = []
for i in range(len(runs)):
    images_grid.append([
        sim_composites[i],
        exp_composites[i]
    ])

# ----------------
# COLOR LIMITS
# ----------------
phase_clim = (-56, -15)

clims_grid = []
for _ in range(len(runs)):
    clims_grid.append([
        phase_clim,
        phase_clim
    ])

# ----------------
# TITLES
# ----------------
col_titles = [
    "Sim vs xRAGE",
    "Exp vs xRAGE"
]

# ----------------
# CREATE FIGURE
# ----------------
nrows = 5
ncols = 2

fig, axs = plt.subplots(nrows, ncols, figsize=(8.8, 15), constrained_layout=False)

# If matplotlib returns 1D axes in some cases, force 2D handling
axs = np.atleast_2d(axs)

for r in range(nrows):
    for c in range(ncols):
        ax = axs[r, c]

        im = ax.imshow(
            images_grid[r][c],
            cmap="RdBu_r",
            vmin=clims_grid[r][c][0],
            vmax=clims_grid[r][c][1],
            interpolation="none",
            aspect="equal"
        )

        ax.set_axis_off()

        # Column titles only on first row
        if r == 0:
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=18)

        # labels above each half
        ax.text(0.25, 1.01, "Recon",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold")

        ax.text(0.75, 1.01, "xRAGE GT",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold")

        ax.text(0.50, 1.01, "|",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold")

        # Row labels = x-ray delays
        if c == 0:
            ax.text(
                -0.10, 0.5,
                f"X-Ray Delay\n{delay_times[r]} ns",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold"
            )

            # scalebar only in first column
            scalebar = ScaleBar(
                dx=dx_eff * 1e6,
                units='µm',
                fixed_value=25,
                location='upper right',
                height_fraction=0.012,
                width_fraction=0.04,
                box_alpha=1.0,
                pad=0.02,
                border_pad=0.02,
                sep=1.0,
                color='black',
                font_properties={"size": 10}
            )
            ax.add_artist(scalebar)

        # colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Phase (rad)", fontsize=10, fontweight="bold")

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.14,
    right=0.96,
    top=0.95,
    bottom=0.04,
    wspace=0.18,
    hspace=0.08
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x2_phase_sim_exp_composite.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()