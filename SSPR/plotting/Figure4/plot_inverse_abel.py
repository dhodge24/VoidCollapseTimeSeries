import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import shift

# ----------------
# ONLY SECOND COLUMN FROM ORIGINAL PLOT
# original columns: 572, 580, 582, 590
# second column = run 580 / 9.67 ns
# ----------------
runs = ["580"]
delay_times = [9.67]

# ----------------
# OPTIONAL SHIFTS FOR GT ALIGNMENT
# ----------------
gt_shifts = {
    "580": (0, 75.49991558517377),
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

    # shift GT
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
elec_clim = (0, 1800)
elec_ticks = [0, 900, 1800]

# ----------------
# FIGURE
# ----------------
nrows, ncols = 2, 1

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(5.2, 6.1),
    constrained_layout=False
)

axs = np.atleast_1d(axs)

row_titles = ["Simulation", "Experiment"]

# ----------------
# PLOT IMAGES
# ----------------
for r in range(nrows):
    ax = axs[r]

    if r == 0:
        img = sim_composites[0]
    else:
        img = exp_composites[0]

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
            f"X-Ray Delay {delay_times[0]} ns",
            fontsize=18,
            fontweight="bold",
            pad=20
        )

    # row label
    ax.text(
        -0.12, 0.5,
        row_titles[r],
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=18,
        fontweight="bold"
    )

    # top labels inside each image
    ax.text(
        0.25, 1.01, "Recon",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        0.50, 1.01, "|",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        0.75, 1.01, "xRAGE GT",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

    # scale bar on both images, remove this block if you do not want it
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
        font_properties={"size": 12}
    )
    ax.add_artist(scalebar)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.18,
    right=0.72,
    top=0.86,
    bottom=0.09,
    hspace=0.2
)

fig.canvas.draw()

# ----------------
# SINGLE SHARED DOUBLE-SIDED COLORBAR
# left side = electron density
# right side = equivalent mass density
# ----------------
cax = fig.add_axes([
    0.80,
    0.20,
    0.035,
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
cbar.ax.tick_params(axis="y", labelsize=10, pad=0.5)

cbar.set_label(
    r"Electron Density ($\mathbf{10^{20}}$ e$\mathbf{^-}$/cm$\mathbf{^3}$)",
    fontsize=13,
    fontweight="bold",
    labelpad=3
)

# right side: equivalent mass density
axr = cbar.ax.twinx()
axr.set_ylim(elec_clim[0], elec_clim[1])
axr.set_yticks(elec_ticks)

mass_tick_vals = [t * mass_conv for t in elec_ticks]
axr.set_yticklabels([f"{t:.2f}" for t in mass_tick_vals])

axr.yaxis.set_ticks_position("right")
axr.yaxis.set_label_position("right")
axr.tick_params(axis="y", labelsize=10, pad=0.5)

axr.set_ylabel(
    r"Mass Density (g/cm$\mathbf{^3}$)",
    fontsize=13,
    fontweight="bold",
    labelpad=3
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/second_column_inverse_Abel_sim_exp_doublecolorbar_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

plt.show()