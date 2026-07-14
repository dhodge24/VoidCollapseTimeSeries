import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import shift

# ----------------
# RUN / DELAY
# third row from original 5-row figure = run 580
# ----------------
run = "580"
delay_time = 9.67

# ----------------
# OPTIONAL SHIFT FOR GT ALIGNMENT
# ----------------
gt_shifts = {
    "580": (0, 75.49991558517377),
}

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_ped = "time_series_recon_cropped_proj_elec_dens/"
dir_areal = "time_series_recon_cropped_areal_dens/"
dir_gt = "GT_adjusted_maps/"

# projected electron density paths
ped_sim_path = dir_main + dir_ped + f"run{run}_sim/" + f"proj_elec_density_run{run}sim.tiff"
ped_exp_path = dir_main + dir_ped + f"run{run}_exp/" + f"proj_elec_density_run{run}exp.tiff"
ped_gt_path  = dir_main + dir_gt  + f"run{run}_GT_maps/" + "projected_electron_density_total_adjusted.tiff"

# areal density paths, only used to define/verify right-side scale if wanted
areal_sim_path = dir_main + dir_areal + f"run{run}_sim/" + f"areal_density_run{run}sim.tiff"
areal_exp_path = dir_main + dir_areal + f"run{run}_exp/" + f"areal_density_run{run}exp.tiff"
areal_gt_path  = dir_main + dir_gt    + f"run{run}_GT_maps/" + "areal_density_total_adjusted.tiff"

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
# HELPERS
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img, dtype=np.float32)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp


def add_double_colorbar_fixed(
    fig,
    cmap,
    ped_clim,
    areal_clim,
    cax_position=[0.805, 0.20, 0.035, 0.58],
    tick_fs=10,
    label_fs=13,
    labelpad_left=5,
    labelpad_right=6,
    ped_ticks=None,
    areal_ticks=None,
):
    """
    Add one fixed-position double-sided colorbar.

    Left side  = projected electron density
    Right side = corresponding areal density
    """
    cax = fig.add_axes(cax_position)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(ped_clim[0], ped_clim[1])
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_linewidth(0.8)

    # ----------------
    # left side: projected electron density
    # ----------------
    if ped_ticks is None:
        ped_ticks = np.linspace(ped_clim[0], ped_clim[1], 7)

    cbar.set_ticks(ped_ticks)
    cbar.ax.set_yticklabels([f"{t:.0f}" for t in ped_ticks])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(axis="y", labelsize=tick_fs, pad=1)

    cbar.set_label(
        r"Projected Electron Density ($\mathbf{10^{6}}$ e$\mathbf{^-}$/nm$\mathbf{^2}$)",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad_left
    )

    # ----------------
    # right side: corresponding areal density scale
    # ----------------
    if areal_ticks is None:
        areal_ticks = np.linspace(areal_clim[0], areal_clim[1], 7)

    axr = cbar.ax.twinx()
    axr.set_ylim(areal_clim[0], areal_clim[1])
    axr.set_yticks(areal_ticks)
    axr.set_yticklabels([f"{t:.4f}" for t in areal_ticks])

    axr.yaxis.set_ticks_position("right")
    axr.yaxis.set_label_position("right")
    axr.tick_params(axis="y", labelsize=tick_fs, pad=1)

    axr.set_ylabel(
        r"Areal Density (g/cm$\mathbf{^2}$)",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad_right
    )

    return cbar, axr

# ----------------
# LOAD DATA
# ----------------
ped_sim = np.array(imread(ped_sim_path), dtype=np.float32)
ped_exp = np.array(imread(ped_exp_path), dtype=np.float32)
ped_gt  = np.array(imread(ped_gt_path), dtype=np.float32)

# Load areal-density maps only if you want to compute automatic areal limits later
areal_sim = np.array(imread(areal_sim_path), dtype=np.float32)
areal_exp = np.array(imread(areal_exp_path), dtype=np.float32)
areal_gt  = np.array(imread(areal_gt_path), dtype=np.float32)

# clip negatives if present
ped_sim[ped_sim < 0] = 0
ped_exp[ped_exp < 0] = 0
ped_gt[ped_gt < 0] = 0

areal_sim[areal_sim < 0] = 0
areal_exp[areal_exp < 0] = 0
areal_gt[areal_gt < 0] = 0

# shift GT
shift_y, shift_x = gt_shifts.get(run, (0, 0))

ped_gt_shifted = shift(
    ped_gt,
    shift=(shift_y, shift_x),
    order=3,
    mode="nearest"
)

areal_gt_shifted = shift(
    areal_gt,
    shift=(shift_y, shift_x),
    order=3,
    mode="nearest"
)

# build PED composites for plotting
sim_ped_composite = make_half_composite(ped_sim, ped_gt_shifted)
exp_ped_composite = make_half_composite(ped_exp, ped_gt_shifted)

# build areal composites only if you want automatic limits
sim_areal_composite = make_half_composite(areal_sim, areal_gt_shifted)
exp_areal_composite = make_half_composite(areal_exp, areal_gt_shifted)

# ----------------
# COLOR LIMITS
# PED image was divided by 10e6 = 1e7,
# so left colorbar units are 10^7 e-/nm^2.
# ----------------
ped_clim = (0, 30)
ped_ticks = [0, 5, 10, 15, 20, 25, 30]

# Corresponding areal-density scale for SU-8.
# 1 projected-density unit = 0.00307 g/cm^2
# 30 projected-density units = 0.0921 g/cm^2
areal_clim = (0, 0.0921)
areal_ticks = [0.0000, 0.0154, 0.0307, 0.0461, 0.0614, 0.0768, 0.0921]

# ----------------
# FIGURE
# ----------------
fig, axs = plt.subplots(
    2, 1,
    figsize=(5.2, 6.1),
    constrained_layout=False
)

row_titles = ["Simulation", "Experiment"]
images = [sim_ped_composite, exp_ped_composite]
cmap_name = "seismic"

# ----------------
# DRAW IMAGES
# ----------------
for r in range(2):
    ax = axs[r]

    ax.imshow(
        images[r],
        cmap=cmap_name,
        vmin=ped_clim[0],
        vmax=ped_clim[1],
        interpolation="none",
        aspect="equal"
    )

    ax.set_axis_off()

    if r == 0:
        ax.set_title(
            f"X-Ray Delay {delay_time} ns",
            fontsize=18,
            fontweight="bold",
            pad=20
        )

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
    right=0.74,
    top=0.86,
    bottom=0.09,
    hspace=0.2
)

fig.canvas.draw()

# ----------------
# SINGLE DOUBLE-SIDED COLORBAR
# ----------------
add_double_colorbar_fixed(
    fig=fig,
    cmap=cmap_name,
    ped_clim=ped_clim,
    areal_clim=areal_clim,
    cax_position=[0.805, 0.20, 0.035, 0.58],
    tick_fs=10,
    label_fs=13,
    labelpad_left=5,
    labelpad_right=6,
    ped_ticks=ped_ticks,
    areal_ticks=areal_ticks
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/third_row_PED_areal_doublecolorbar_1col_2rows_seismic.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

plt.show()