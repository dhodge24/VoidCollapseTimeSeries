import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
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
# right side of colorbar only
# ----------------
N_A = 6.02214076e23
elec_fac = 10e20   # kept exactly as written in your code
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8

mass_conv = elec_fac * A_SU8 / (N_A * Z_SU8)
# equivalent mass density = electron density * mass_conv

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
# HELPER: DOUBLE-SIDED COLORBAR
# left side = electron density
# right side = equivalent mass density
# ----------------
def add_double_colorbar(
    fig,
    ax,
    cmap,
    elec_clim,
    mass_conv,
    gap=0.010,
    cbar_width=0.010,
    tick_fs=7,
    label_fs=8,
    labelpad_left=8,
    labelpad_right=8,
    elec_ticks=None,
):
    pos = ax.get_position()

    cax = fig.add_axes([
        pos.x1 + gap,
        pos.y0,
        cbar_width,
        pos.height
    ])

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(elec_clim[0], elec_clim[1])
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_linewidth(0.8)

    # ----------------
    # LEFT SIDE: ELECTRON DENSITY
    # ----------------
    if elec_ticks is None:
        elec_ticks = np.linspace(elec_clim[0], elec_clim[1], 5)

    cbar.set_ticks(elec_ticks)
    cbar.ax.set_yticklabels([f"{t:.0f}" for t in elec_ticks])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(axis="y", labelsize=tick_fs, pad=1)
    cbar.set_label(
        r"Electron Density ($\mathbf{10^{20}}$ e$\mathbf{^-}$/cm$\mathbf{^3}$)",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad_left
    )

    # ----------------
    # RIGHT SIDE: MASS DENSITY
    # same color positions as the electron-density ticks
    # ----------------
    axr = cbar.ax.twinx()
    axr.set_ylim(elec_clim[0], elec_clim[1])
    axr.set_yticks(elec_ticks)

    mass_tick_vals = [t * mass_conv for t in elec_ticks]
    axr.set_yticklabels([f"{t:.2f}" for t in mass_tick_vals])

    axr.yaxis.set_ticks_position("right")
    axr.yaxis.set_label_position("right")
    axr.tick_params(axis="y", labelsize=tick_fs, pad=1)
    axr.set_ylabel(
        r"Mass Density (g/cm$\mathbf{^3}$)",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad_right
    )

    return cbar, axr

# ----------------
# LOAD DATA AND BUILD COMPOSITES
# both columns are electron density
# col 1 = sim recon | GT
# col 2 = exp recon | GT
# ----------------
sim_composites = []
exp_composites = []

for run in runs:
    # ----------------
    # RECON PATHS
    # ----------------
    sim_dir = dir_main + dir_inverse_abel + f"run{run}_sim/"
    exp_dir = dir_main + dir_inverse_abel + f"run{run}_exp/"

    sim_path = sim_dir + f"run_{run}_sim_inverse_Abel_data_regularized_Gaussian_blurred8.tiff"
    exp_path = exp_dir + f"run_{run}_exp_inverse_Abel_data_regularized_Gaussian_blurred8.tiff"

    # ----------------
    # GT PATH
    # ----------------
    gt_elec_path = dir_main + dir_gt + f"run{run}_GT_maps/electron_density_total_GT_adjusted.tiff"

    # ----------------
    # READ DATA
    # ----------------
    sim_recon = np.array(imread(sim_path), dtype=np.float32)
    exp_recon = np.array(imread(exp_path), dtype=np.float32)
    gt_elec = np.array(imread(gt_elec_path), dtype=np.float32)

    # ----------------
    # SUBTRACT OFFSET AND CLIP NEGATIVES
    # ----------------
    sim_recon = sim_recon - 120
    exp_recon = exp_recon - 120

    sim_recon[sim_recon < 0] = 0
    exp_recon[exp_recon < 0] = 0

    # ----------------
    # OPTIONAL GT SHIFT
    # ----------------
    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    gt_elec_shifted = shift(gt_elec, shift=(shift_y, shift_x), order=3, mode="nearest")

    # ----------------
    # MAKE COMPOSITES
    # ----------------
    sim_composites.append(make_half_composite(sim_recon, gt_elec_shifted))
    exp_composites.append(make_half_composite(exp_recon, gt_elec_shifted))

# ----------------
# MANUAL ELECTRON-DENSITY COLOR LIMITS (per row)
# same limits used for both columns in each row
# ----------------
elec_clims = [
    (0, 1300),   # run 572
    (50, 1800),   # run 576
    (0, 2100),     # run 580
    (150, 2300),   # run 582
    (150, 1600),   # run 590
]

# Optional manual electron-density ticks per row
elec_ticks_per_row = [
    [200, 500, 800, 1100, 1300],   # run 572
    [200, 600, 1000, 1400, 1800],  # run 576
    [0, 500, 1000, 1500, 2100],    # run 580
    [200, 800, 1400, 1800, 2300],  # run 582
    [200, 500, 900, 1300, 1600],   # run 590
]

# ----------------
# FIGURE CONTENT
# ----------------
images_grid = [[sim_composites[i], exp_composites[i]] for i in range(len(runs))]
col_titles = ["Simulation", "Experimental"]

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

# ----------------
# DRAW IMAGES FIRST
# ----------------
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r, c]
        vmin, vmax = elec_clims[r]

        ax.imshow(
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
            0.50, 1.01, "|",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
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

# ----------------
# FINALIZE LAYOUT FIRST
# ----------------
fig.subplots_adjust(
    left=0.16,
    right=0.76,
    top=0.96,
    bottom=0.04,
    wspace=0.55,
    hspace=0.025
)

fig.canvas.draw()

# ----------------
# NOW ADD DOUBLE COLORBARS
# both columns use electron-density image scaling,
# with equivalent mass-density values on the right
# ----------------
for r in range(nrows):
    for c in range(ncols):
        add_double_colorbar(
            fig=fig,
            ax=axs[r, c],
            cmap="inferno",
            elec_clim=elec_clims[r],
            mass_conv=mass_conv,
            gap=0.05,
            cbar_width=0.010,
            tick_fs=6,
            label_fs=7,
            labelpad_left=4,
            labelpad_right=4,
            elec_ticks=elec_ticks_per_row[r]
        )

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x2_inverse_Abel_sim_exp_doublecolorbar_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()