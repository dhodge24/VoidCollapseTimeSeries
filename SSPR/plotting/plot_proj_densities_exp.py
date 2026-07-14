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
# ----------------
gt_shifts = {
    "572": (0, 25.000069739608307),
    "576": (0, -54.99983132112288),
    "580": (0, 75.49991558517377),
    "582": (45, 50.4996321295979),
    "590": (0, -49.99998557735444),
}

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_ped = "time_series_recon_cropped_proj_elec_dens/"
dir_areal = "time_series_recon_cropped_areal_dens/"

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
# left half = reconstruction
# right half = GT
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp

# ----------------
# HELPER: SINGLE COLORBAR
# ----------------
def add_single_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.22)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(axis="y", labelsize=7, pad=1)
    cbar.set_label(label, fontsize=8.0, fontweight="bold", labelpad=4)

    return cbar

# ----------------
# LOAD DATA AND BUILD COMPOSITES
# ----------------
ped_composites = []
areal_composites = []

for run in runs:
    # ----------------
    # EXP reconstruction paths
    # ----------------
    ped_exp_dir = dir_main + dir_ped + f"run{run}_exp/"
    areal_exp_dir = dir_main + dir_areal + f"run{run}_exp/"

    # ----------------
    # RECON FILES
    # ----------------
    ped_recon_path = ped_exp_dir + f"proj_elec_density_run{run}exp.tiff"
    areal_recon_path = areal_exp_dir + f"areal_density_run{run}exp.tiff"

    # ----------------
    # GT FILES
    # ----------------
    ped_gt_path = dir_main + f"GT_adjusted_maps/run{run}_GT_maps/projected_electron_density_total_adjusted.tiff"
    areal_gt_path = dir_main + f"GT_adjusted_maps/run{run}_GT_maps/areal_density_total_adjusted.tiff"

    ped_recon = np.array(imread(ped_recon_path), dtype=np.float32)
    ped_gt = np.array(imread(ped_gt_path), dtype=np.float32)

    areal_recon = np.array(imread(areal_recon_path), dtype=np.float32)
    areal_gt = np.array(imread(areal_gt_path), dtype=np.float32)

    # ----------------
    # OPTIONAL GT SHIFT
    # ----------------
    shift_y, shift_x = gt_shifts.get(run, (0, 0))

    ped_gt_shifted = shift(ped_gt, shift=(shift_y, shift_x), order=3, mode="nearest")
    areal_gt_shifted = shift(areal_gt, shift=(shift_y, shift_x), order=3, mode="nearest")

    # ----------------
    # MAKE COMPOSITES
    # ----------------
    ped_composites.append(make_half_composite(ped_recon, ped_gt_shifted))
    areal_composites.append(make_half_composite(areal_recon, areal_gt_shifted))

# ----------------
# COLOR LIMITS
# ----------------
ped_all = np.concatenate([img.ravel() for img in ped_composites])
areal_all = np.concatenate([img.ravel() for img in areal_composites])

ped_clim = (np.nanmin(ped_all), np.nanmax(ped_all))
areal_clim = (np.nanmin(areal_all), np.nanmax(areal_all))

# # Optional manual limits:
# ped_clim = (0, 28)
# areal_clim = (0, 0.12)

# ----------------
# FIGURE CONTENT
# ----------------
images_grid = [[ped_composites[i], areal_composites[i]] for i in range(len(runs))]
col_titles = ["Projected Electron Density", "Areal Density"]
cbar_labels = [
    r"Projected Electron Density ($\mathbf{10^{6}}$ e$^-$/nm$\mathbf{^2}$)",
    r"Areal Density (g/cm$\mathbf{^2}$)"
]

# ----------------
# CREATE FIGURE
# ----------------
nrows, ncols = 5, 2
fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(9.4, 15.0),
    constrained_layout=False
)
axs = np.atleast_2d(axs)

for r in range(nrows):
    for c in range(ncols):
        ax = axs[r, c]

        if c == 0:
            vmin, vmax = ped_clim
        else:
            vmin, vmax = areal_clim

        im = ax.imshow(
            images_grid[r][c],
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            aspect="equal"
        )

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=18)

        ax.text(
            0.25, 1.01, "Recon",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            0.75, 1.01, "xRAGE GT",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

        ax.text(
            0.50, 1.01, "|",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

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

            scalebar = ScaleBar(
                dx=dx_eff * 1e6,
                units="µm",
                fixed_value=25,
                location="upper right",
                height_fraction=0.012,
                width_fraction=0.04,
                box_alpha=1.0,
                pad=0.02,
                border_pad=0.02,
                sep=1.0,
                color="black",
                font_properties={"size": 10}
            )
            ax.add_artist(scalebar)

        add_single_colorbar(fig, ax, im, cbar_labels[c])

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.17,
    right=0.965,
    top=0.95,
    bottom=0.04,
    wspace=0.10,
    hspace=0.08
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x2_PED_Areal_exp_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()