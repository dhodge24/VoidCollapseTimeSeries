import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# RUNS / DELAYS
# ----------------
# Replace these with your actual 5 run numbers
runs = ["572", "576", "580", "582", "590"]
delay_times = [6.82, 8.12, 9.67, 10.52, 13.42]

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
# LOAD DATA
# ----------------
gt_phase_imgs = []
sim_phase_imgs = []
exp_phase_imgs = []

for run in runs:
    dir_sim = dir_main + dir_data + f"run{run}_sim/"
    dir_exp = dir_main + dir_data + f"run{run}_exp/"

    gt_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_gt_final.tiff"), dtype=np.float32)
    sim_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_final.tiff"), dtype=np.float32)
    exp_phase = np.array(imread(dir_exp + f"run{run}_exp_ph_final.tiff"), dtype=np.float32)

    gt_phase_imgs.append(gt_phase)
    sim_phase_imgs.append(sim_phase)
    exp_phase_imgs.append(exp_phase)

# ----------------
# ORGANIZE IMAGES
# rows = delays
# cols = [raw intensity, sim recon, exp recon]
# ----------------
images_grid = []
for i in range(len(runs)):
    images_grid.append([
        gt_phase_imgs[i],
        sim_phase_imgs[i],
        exp_phase_imgs[i]
    ])

# ----------------
# COLOR LIMITS
# adjust as needed
# ----------------
phase_gt_clim = (-56, -15)
phase_clim = (-56, -15)

clims_grid = []
for _ in range(len(runs)):
    clims_grid.append([
        phase_gt_clim,
        phase_clim,
        phase_clim
    ])

# ----------------
# TITLES
# ----------------
col_titles = [
    "xRAGE GT Phase",
    "xRAGE Sim Recon",
    "Exp Recon"
]

# ----------------
# CREATE FIGURE
# ----------------
nrows = 5
ncols = 3

fig, axs = plt.subplots(nrows, ncols, figsize=(12, 15), constrained_layout=False)

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
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=10)

        # Row labels = x-ray delays
        if c == 0:
            ax.text(
                -0.08, 0.5,
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

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)

        cbar.set_label("Phase (rad)", fontsize=10, fontweight="bold")

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.12,
    right=0.96,
    top=0.93,
    bottom=0.05,
    wspace=0.30,
    hspace=0.0
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x3_intensity_sim_exp_recon_phase.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)
plt.tight_layout()
plt.show()