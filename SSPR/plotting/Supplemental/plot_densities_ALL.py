import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import shift

plt.close("all")

# ----------------
# RUNS / DELAYS (YOUR ORDER)
# ----------------
runs = ["572", "574", "576", "578", "580", "582", "584", "590", "586"]
delay_times = [6.82, 7.97, 8.12, 8.96, 9.67, 10.52, 11.72, 13.42, 14.41]

# ----------------
# OPTIONAL SHIFTS FOR GT ALIGNMENT
# ----------------
gt_shifts = {
    "572": (0, 25.000069739608307),
    "574": (25, 25.00006607127864),
    "576": (0, -54.99983132112288),
    "578": (0, -29.499867937574663),
    "580": (0, 75.49991558517377),
    "582": (45, 50.4996321295979),
    "584": (0, -44.49995552005612),
    "590": (0, -49.99998557735444),
    "586": (0, -49.99998396918704 ),
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
# ----------------
N_A = 6.02214076e23
elec_fac = 10e20
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8

mass_conv = elec_fac * A_SU8 / (N_A * Z_SU8)

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
# LOAD DATA AND BUILD COMPOSITES
# row 0 = simulation recon | GT
# row 1 = experimental recon | GT
# columns = delay times
# ----------------
sim_composites = []
exp_composites = []

for run in runs:
    sim_dir = dir_main + dir_inverse_abel + f"run{run}_sim/"
    exp_dir = dir_main + dir_inverse_abel + f"run{run}_exp/"

    sim_path = sim_dir + f"run_{run}_sim_inverse_Abel_data_regularized_Gaussian_blurred8.tiff"
    exp_path = exp_dir + f"run_{run}_exp_inverse_Abel_data_regularized_Gaussian_blurred8.tiff"
    gt_elec_path = dir_main + dir_gt + f"run{run}_GT_maps/electron_density_total_GT_adjusted.tiff"

    sim_recon = np.array(imread(sim_path), dtype=np.float32)
    exp_recon = np.array(imread(exp_path), dtype=np.float32)
    gt_elec = np.array(imread(gt_elec_path), dtype=np.float32)

    # subtract offset and clip
    sim_recon = sim_recon - 60
    exp_recon = exp_recon - 60
    sim_recon[sim_recon < 0] = 0
    exp_recon[exp_recon < 0] = 0

    # shift GT if needed
    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    gt_elec_shifted = shift(
        gt_elec,
        shift=(shift_y, shift_x),
        order=3,
        mode="nearest"
    )

    sim_composites.append(make_half_composite(sim_recon, gt_elec_shifted))
    exp_composites.append(make_half_composite(exp_recon, gt_elec_shifted))

# ----------------
# GLOBAL COLOR LIMITS
# ----------------
elec_clim = (0, 2500)
elec_ticks = [0, 1250, 2500]

# ----------------
# FIGURE
# ----------------
nrows, ncols = 2, len(runs)

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(21.0, 5.5),
    constrained_layout=False
)

axs = np.atleast_2d(axs)
row_titles = ["Simulation", "Experiment"]

# ----------------
# PLOT IMAGES
# ----------------
for c, run in enumerate(runs):
    for r in range(nrows):
        ax = axs[r, c]
        img = sim_composites[c] if r == 0 else exp_composites[c]

        im = ax.imshow(
            img,
            cmap="plasma",
            vmin=elec_clim[0],
            vmax=elec_clim[1],
            interpolation="none",
            aspect="equal"
        )

        ax.set_axis_off()

        # column title = delay time
        if r == 0:
            ax.set_title(
                f"{delay_times[c]} ns",
                fontsize=16,
                fontweight="bold",
                pad=16
            )

        # row label
        if c == 0:
            ax.text(
                -0.08, 0.5,
                row_titles[r],
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=16,
                fontweight="bold"
            )

        # top labels inside each image
        ax.text(
            0.25, 1.01, "Recon",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
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
            0.75, 1.01, "xRAGE",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

        # scale bar only first column
        if c == 0:
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
# SPACING
# less vertical space between rows
# more horizontal space between columns
# leave more room at right for two-sided colorbar labels
# ----------------
fig.subplots_adjust(
    left=0.07,
    right=0.83,
    top=0.87,
    bottom=0.10,
    wspace=0.08,
    hspace=0.04
)

fig.canvas.draw()

# ----------------
# SINGLE SHARED DOUBLE-SIDED COLORBAR
# left side = electron density
# right side = equivalent mass density
# ----------------
cax = fig.add_axes([
    0.875,   # shifted farther left so left/right labels have room
    0.20,
    0.016,
    0.58
])

sm = plt.cm.ScalarMappable(cmap="plasma")
sm.set_clim(elec_clim[0], elec_clim[1])
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
cbar.outline.set_linewidth(0.8)

# left side: electron density
cbar.set_ticks(elec_ticks)
cbar.ax.set_yticklabels([f"{t:.0f}" for t in elec_ticks])
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.yaxis.set_label_position("left")
cbar.ax.tick_params(axis="y", labelsize=10, pad=1.0)

cbar.set_label(
    r"Electron Density ($\mathbf{10^{20}}$ e$\mathbf{^-}$/cm$\mathbf{^3}$)",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

# right side: equivalent mass density
axr = cbar.ax.twinx()
axr.set_ylim(elec_clim[0], elec_clim[1])
axr.set_yticks(elec_ticks)

mass_tick_vals = [t * mass_conv for t in elec_ticks]
axr.set_yticklabels([f"{t:.2f}" for t in mass_tick_vals])

axr.yaxis.set_ticks_position("right")
axr.yaxis.set_label_position("right")
axr.tick_params(axis="y", labelsize=10, pad=1.0)

axr.set_ylabel(
    r"Mass Density (g/cm$\mathbf{^3}$)",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/2x9_inverse_Abel_sim_exp_doublecolorbar_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()