import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"

# Row 1
gt1_path = dir_main + "GT_adjusted_maps/run576_GT_maps/areal_density_total_adjusted.tiff"
recon1_path = dir_main + "time_series_recon_cropped_areal_dens/run576_sim/areal_density_run576sim.tiff"

# Row 2 (sampling test)
gt2_path = dir_main + "time_series_new_recons/run576_sim_sampling_test/areal_density_sim.tiff"
recon2_path = dir_main + "time_series_new_recons/run576_sim_sampling_test/areal_density_final_run576sim_test.tiff"

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
dy_eff = dx_eff

# ----------------
# LOAD DATA
# ----------------
gt1_img = np.array(imread(gt1_path), dtype=np.float32)
recon1_img = np.array(imread(recon1_path), dtype=np.float32)

gt2_img = np.array(imread(gt2_path), dtype=np.float32)
recon2_img = np.array(imread(recon2_path), dtype=np.float32)

# ----------------
# ROW SETUP
# ----------------
rows = [
    {
        "label": "X-Ray Delay 8.12 ns",
        "gt": gt1_img,
        "recon": recon1_img,
    },
    {
        "label": "X-Ray Delay 8.12 ns\nMore Sampling",
        "gt": gt2_img,
        "recon": recon2_img,
    }
]

col_titles = [
    "Ground Truth\nAreal Density",
    "Simulation Recon",
    "Vertical Profile",
    "Horizontal Profile"
]

# ----------------
# PROFILE LOCATIONS
# ----------------
ny0, nx0 = gt1_img.shape
base_center_x = nx0 // 2 + 50
base_center_y = ny0 // 2 + 350

vertical_x_shifts = [0, 200]
horizontal_y_shifts = [0, 340]

# ----------------
# COLOR LIMITS
# ----------------
areal_clim = (0.0, 0.09)

# ----------------
# PROFILE LIMITS (TUNE THESE)
# ----------------
vertical_profile_ylims = [
    (0.0, 0.09),
    (0.0, 0.09),
]

horizontal_profile_ylims = [
    (0.02, 0.04),
    (0.02, 0.04),
]

# ----------------
# FIGURE
# ----------------
fig, axs = plt.subplots(
    2, 4,
    figsize=(12, 5.8),
    gridspec_kw={"width_ratios": [1, 1, 0.9, 0.9]}
)

for r, row in enumerate(rows):
    gt = row["gt"]
    recon = row["recon"]

    ny, nx = recon.shape

    cx = base_center_x + vertical_x_shifts[r]
    cy = base_center_y + horizontal_y_shifts[r]

    cx = np.clip(cx, 0, nx - 1)
    cy = np.clip(cy, 0, ny - 1)

    x_um = (np.arange(nx) - cx) * dx_eff * 1e6
    y_um = (np.arange(ny) - cy) * dy_eff * 1e6

    vertical_gt = gt[:, cx]
    vertical_recon = recon[:, cx]

    horizontal_gt = gt[cy, :]
    horizontal_recon = recon[cy, :]

    # ----------------
    # COLUMN 1: GT
    # ----------------
    ax = axs[r, 0]
    im0 = ax.imshow(
        gt,
        cmap="seismic",
        vmin=areal_clim[0],
        vmax=areal_clim[1],
        interpolation="none",
        aspect="equal"
    )

    ax.axvline(cx, color="black", linestyle="--", linewidth=1.6)
    ax.axhline(cy, color="darkorange", linestyle="--", linewidth=1.6)
    ax.set_axis_off()

    if r == 0:
        ax.set_title(col_titles[0], fontsize=10, fontweight="bold", pad=4)

    ax.text(
        -0.08, 0.5, row["label"],
        transform=ax.transAxes,
        rotation=90,
        va="center", ha="center",
        fontsize=10,
        fontweight="bold"
    )

    scalebar = ScaleBar(
        dx=dx_eff * 1e6,
        units='µm',
        fixed_value=25,
        location='upper right',
        height_fraction=0.012,
        width_fraction=0.04,
        box_alpha=1.0,
        box_color='white',
        color='black',
        font_properties={"size": 7}
    )
    ax.add_artist(scalebar)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.set_label("Areal Density ($\mathbf{g/cm^2}$)", fontsize=8, fontweight="bold", labelpad=3)

    # ----------------
    # COLUMN 2: RECON
    # ----------------
    ax = axs[r, 1]
    im1 = ax.imshow(
        recon,
        cmap="seismic",
        vmin=areal_clim[0],
        vmax=areal_clim[1],
        interpolation="none",
        aspect="equal"
    )

    ax.axvline(cx, color="turquoise", linestyle="--", linewidth=1.6)
    ax.axhline(cy, color="limegreen", linestyle="--", linewidth=1.6)
    ax.set_axis_off()

    if r == 0:
        ax.set_title(col_titles[1], fontsize=10, fontweight="bold", pad=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.set_label("Areal Density ($\mathbf{g/cm^2}$)", fontsize=8, fontweight="bold", labelpad=3)

    # ----------------
    # VERTICAL PROFILE
    # ----------------
    ax = axs[r, 2]
    ax.plot(y_um, vertical_gt, color="black", linewidth=1.8, label="GT")
    ax.plot(y_um, vertical_recon, color="turquoise", linewidth=1.8, label="Recon")

    ax.set_xlim(y_um[0], y_um[-1])
    ax.set_ylim(*vertical_profile_ylims[r])

    ax.set_xlabel("y (µm)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Areal Density [$\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7, pad=1)

    ax.set_box_aspect(1)

    if r == 0:
        ax.set_title(col_titles[2], fontsize=10, fontweight="bold", pad=4)
        ax.legend(fontsize=7, frameon=False)

    # ----------------
    # HORIZONTAL PROFILE
    # ----------------
    ax = axs[r, 3]
    ax.plot(x_um, horizontal_gt, color="darkorange", linewidth=1.8, label="GT")
    ax.plot(x_um, horizontal_recon, color="limegreen", linewidth=1.8, label="Recon")

    ax.set_xlim(x_um[0], x_um[-1])
    ax.set_ylim(*horizontal_profile_ylims[r])

    ax.set_xlabel("x (µm)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Areal Density [$\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7, pad=1)

    ax.set_box_aspect(1)

    if r == 0:
        ax.set_title(col_titles[3], fontsize=10, fontweight="bold", pad=4)
        ax.legend(fontsize=7, frameon=False)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.08,
    right=0.94,
    bottom=0.10,
    top=0.93,
    wspace=0.32,
    hspace=0.02
)

# ----------------
# SAVE
# ----------------
plt.savefig(
    dir_main + "run576_2x4_areal_density_profiles.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()