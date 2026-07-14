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
dir_inverse_abel = "inverse_Abel/"
dir_gt = "GT_adjusted_maps/"

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
# MASS DENSITY CONVERSION
# electron density recon units are scaled as 10^20 e-/cm^3
# ----------------
N_A = 6.02214076e23
elec_fac = 10e20
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8

# ----------------
# HELPER: MAKE HALF COMPOSITE
# left half = reconstruction
# right half = GT
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img, dtype=np.float32)
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
elec_composites = []
mass_composites = []

for run in runs:
    # ----------------
    # SIM RECON PATH
    # ----------------
    sim_dir = dir_main + dir_inverse_abel + f"run{run}_sim/"
    sim_path = sim_dir + f"run_{run}_sim_inverse_Abel_data_regularized_Gaussian_blurred8.tiff"

    # ----------------
    # GT PATHS
    # ----------------
    gt_elec_path = dir_main + dir_gt + f"run{run}_GT_maps/electron_density_total_GT_adjusted.tiff"
    gt_mass_path = dir_main + dir_gt + f"run{run}_GT_maps/density_total_GT_adjusted.tiff"

    # ----------------
    # READ DATA
    # ----------------
    elec_recon = np.array(imread(sim_path), dtype=np.float32)
    gt_elec = np.array(imread(gt_elec_path), dtype=np.float32)
    gt_mass = np.array(imread(gt_mass_path), dtype=np.float32)

    # ----------------
    # SUBTRACT OFFSET AND CLIP NEGATIVES
    # ----------------
    elec_recon = elec_recon - 120
    elec_recon[elec_recon < 0] = 0

    # ----------------
    # CONVERT ELECTRON DENSITY RECON TO MASS DENSITY
    # ----------------
    mass_recon = elec_recon * elec_fac * A_SU8 / (N_A * Z_SU8)

    # ----------------
    # OPTIONAL GT SHIFT
    # ----------------
    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    gt_elec_shifted = shift(gt_elec, shift=(shift_y, shift_x), order=3, mode="nearest")
    gt_mass_shifted = shift(gt_mass, shift=(shift_y, shift_x), order=3, mode="nearest")

    # ----------------
    # MAKE COMPOSITES
    # ----------------
    elec_composites.append(make_half_composite(elec_recon, gt_elec_shifted))
    mass_composites.append(make_half_composite(mass_recon, gt_mass_shifted))

# ----------------
# MANUAL COLOR LIMITS (per row)
# first list = electron density column
# second list = mass density column
# ----------------
elec_clims = [
    (150, 1200),   # run 572
    (150, 1300),   # run 576
    (0, 1400),     # run 580
    (150, 1700),   # run 582
    (150, 1400),   # run 590
]

mass_clims = [
    (0.00, 4.0),  # run 572
    (0.00, 4.0),  # run 576
    (0.00, 4.0),  # run 580
    (0.00, 5.5),  # run 582
    (0.00, 4.5),  # run 590
]

# ----------------
# FIGURE CONTENT
# ----------------
images_grid = [[elec_composites[i], mass_composites[i]] for i in range(len(runs))]
col_titles = ["Electron Density", "Mass Density"]
cbar_labels = [
    r"Electron Density ($\mathbf{10^{20}}$ e$\mathbf{^-}$/cm$\mathbf{^3}$)",
    r"Mass Density (g/cm$\mathbf{^3}$)"
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
            vmin, vmax = elec_clims[r]
        else:
            vmin, vmax = mass_clims[r]

        im = ax.imshow(
            images_grid[r][c],
            cmap="inferno",
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
    "/Users/danielhodge/Desktop/5x2_inverse_Abel_sim_elec_mass_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()