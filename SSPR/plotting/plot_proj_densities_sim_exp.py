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
dir_ped = "time_series_recon_cropped_proj_elec_dens/"
dir_areal = "time_series_recon_cropped_areal_dens/"
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
# HELPERS
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp


def add_double_colorbar(
    fig,
    ax,
    cmap,
    ped_clim,
    areal_clim,
    gap=0.010,
    cbar_width=0.010,
    tick_fs=7,
    label_fs=8,
    labelpad_left=8,
    labelpad_right=8,
    ped_ticks=[1, 5, 10, 15, 20, 25, 30],
    areal_ticks=[0.0025, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
):
    """
    Place one double-sided colorbar immediately to the right of ax
    using FINAL axis position in figure coordinates.
    """
    pos = ax.get_position()

    cax = fig.add_axes([
        pos.x1 + gap,
        pos.y0,
        cbar_width,
        pos.height
    ])

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(ped_clim[0], ped_clim[1])
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_linewidth(0.8)

    # Left side: PED
    if ped_ticks is None:
        ped_ticks = np.linspace(ped_clim[0], ped_clim[1], 5)

    cbar.set_ticks(ped_ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ped_ticks])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(axis="y", labelsize=tick_fs, pad=1)
    cbar.set_label(
        r"Projected Electron Density ($\mathbf{10^{6}}$ e$\mathbf{^-}$/nm$\mathbf{^2}$)",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad_left
    )

    # Right side: Areal density
    axr = cbar.ax.twinx()
    axr.set_ylim(areal_clim[0], areal_clim[1])

    if areal_ticks is None:
        areal_ticks = np.linspace(areal_clim[0], areal_clim[1], 5)

    axr.set_yticks(areal_ticks)
    axr.set_yticklabels([f"{t:.3f}" for t in areal_ticks])
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
sim_ped_composites = []
exp_ped_composites = []
sim_areal_composites = []
exp_areal_composites = []

for run in runs:
    ped_sim_path = dir_main + dir_ped + f"run{run}_sim/" + f"proj_elec_density_run{run}sim.tiff"
    areal_sim_path = dir_main + dir_areal + f"run{run}_sim/" + f"areal_density_run{run}sim.tiff"

    ped_exp_path = dir_main + dir_ped + f"run{run}_exp/" + f"proj_elec_density_run{run}exp.tiff"
    areal_exp_path = dir_main + dir_areal + f"run{run}_exp/" + f"areal_density_run{run}exp.tiff"

    ped_gt_path = dir_main + dir_gt + f"run{run}_GT_maps/" + "projected_electron_density_total_adjusted.tiff"
    areal_gt_path = dir_main + dir_gt + f"run{run}_GT_maps/" + "areal_density_total_adjusted.tiff"

    ped_sim = np.array(imread(ped_sim_path), dtype=np.float32)
    areal_sim = np.array(imread(areal_sim_path), dtype=np.float32)

    ped_exp = np.array(imread(ped_exp_path), dtype=np.float32)
    areal_exp = np.array(imread(areal_exp_path), dtype=np.float32)

    ped_gt = np.array(imread(ped_gt_path), dtype=np.float32)
    areal_gt = np.array(imread(areal_gt_path), dtype=np.float32)

    shift_y, shift_x = gt_shifts.get(run, (0, 0))
    ped_gt_shifted = shift(ped_gt, shift=(shift_y, shift_x), order=3, mode="nearest")
    areal_gt_shifted = shift(areal_gt, shift=(shift_y, shift_x), order=3, mode="nearest")

    sim_ped_composites.append(make_half_composite(ped_sim, ped_gt_shifted))
    exp_ped_composites.append(make_half_composite(ped_exp, ped_gt_shifted))

    sim_areal_composites.append(make_half_composite(areal_sim, areal_gt_shifted))
    exp_areal_composites.append(make_half_composite(areal_exp, areal_gt_shifted))

# ----------------
# COLOR LIMITS
# ----------------
ped_all = np.concatenate([img.ravel() for img in sim_ped_composites + exp_ped_composites])
areal_all = np.concatenate([img.ravel() for img in sim_areal_composites + exp_areal_composites])

ped_clim = (np.nanmin(ped_all), np.nanmax(ped_all))
areal_clim = (np.nanmin(areal_all), np.nanmax(areal_all))

# Optional manual limits:
# ped_clim = (0, 28)
# areal_clim = (0, 0.12)

# ----------------
# FIGURE CONTENT
# plot PED composites
# col 1 = sim recon | GT
# col 2 = exp recon | GT
# ----------------
images_grid = [
    [sim_ped_composites[i], exp_ped_composites[i]]
    for i in range(len(runs))
]

col_titles = [
    "Simulation",
    "Experiment"
]

# ----------------
# CREATE FIGURE
# ----------------
nrows, ncols = 5, 2
fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(9.2, 13.0),
    constrained_layout=False
)
axs = np.atleast_2d(axs)

# ----------------
# DRAW IMAGES ONLY FIRST
# ----------------
for r in range(nrows):
    for c in range(ncols):
        ax = axs[r, c]

        ax.imshow(
            images_grid[r][c],
            cmap="RdBu",
            vmin=ped_clim[0],
            vmax=ped_clim[1],
            interpolation="none",
            aspect="equal"
        )

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=12.5, fontweight="bold", pad=18)

        ax.text(
            0.25, 1.01, "Recon",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold"
        )
        ax.text(
            0.50, 1.01, "|",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold"
        )
        ax.text(
            0.75, 1.01, "xRAGE GT",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold"
        )

        if c == 0:
            ax.text(
                -0.10, 0.5,
                f"X-Ray Delay\n{delay_times[r]} ns",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=11, fontweight="bold"
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
                font_properties={"size": 9}
            )
            ax.add_artist(scalebar)

# ----------------
# FINALIZE IMAGE LAYOUT FIRST
# ----------------
fig.subplots_adjust(
    left=0.16,
    right=0.76,
    top=0.96,
    bottom=0.04,
    wspace=0.55,
    hspace=0.025
)

# Force final axis positions to exist
fig.canvas.draw()

# ----------------
# NOW ADD COLORBARS
# ----------------
for r in range(nrows):
    for c in range(ncols):
        add_double_colorbar(
            fig=fig,
            ax=axs[r, c],
            cmap="RdBu",
            ped_clim=ped_clim,
            areal_clim=areal_clim,
            gap=0.05,
            cbar_width=0.010,
            tick_fs=6,
            label_fs=7,
            labelpad_left=4,
            labelpad_right=4
        )

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x2_sim_exp_PED_doublecolorbar_vs_GT.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False
)

plt.show()