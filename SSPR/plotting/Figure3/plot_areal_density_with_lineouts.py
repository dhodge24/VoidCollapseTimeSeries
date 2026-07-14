import string
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# RUNS / DELAYS
# ----------------
# Swapped the 2nd and 3rd rows relative to the original order:
# original: ["572", "580", "582", "590"]
# new:      ["572", "582", "580", "590"]
runs = ["572", "582", "580", "590"]

# Delay times are still kept here in case you want to use them later,
# but the row labels now use Time 1, Time 2, etc.
delay_times = [6.82, 10.52, 9.67, 13.42]

n_runs = len(runs)
row_labels = [f"({c})" for c in string.ascii_lowercase[:n_runs]]
# time_labels = [f"Time {i + 1}" for i in range(n_runs)]
time_labels = ["Time 1", "Time 5", "Time 6", "Time 8"]

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_gt = "GT_adjusted_maps/"
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
dy_eff = dx_eff

# ----------------
# LOAD DATA
# ----------------
gt_imgs = []
sim_imgs = []
exp_imgs = []

for run in runs:
    gt_path = dir_main + dir_gt + f"run{run}_GT_maps/areal_density_total_adjusted.tiff"
    sim_path = dir_main + dir_areal + f"run{run}_sim/areal_density_run{run}sim.tiff"
    exp_path = dir_main + dir_areal + f"run{run}_exp/areal_density_run{run}exp.tiff"

    gt_imgs.append(np.array(imread(gt_path), dtype=np.float32))
    sim_imgs.append(np.array(imread(sim_path), dtype=np.float32))
    exp_imgs.append(np.array(imread(exp_path), dtype=np.float32))

# ----------------
# PROFILE LOCATIONS
# ----------------
# These are now reordered to match the swapped run order:
# original:
# 572, 580, 582, 590
#
# new:
# 572, 582, 580, 590
profile_positions = [
    (gt_imgs[0].shape[1] // 2 - 50,  gt_imgs[0].shape[0] // 2 + 300),  # run 572
    (gt_imgs[1].shape[1] // 2 - 50,  gt_imgs[1].shape[0] // 2 + 300),  # run 582
    (gt_imgs[2].shape[1] // 2 - 100, gt_imgs[2].shape[0] // 2 + 450),  # run 580
    (gt_imgs[3].shape[1] // 2 + 50,  gt_imgs[3].shape[0] // 2 + 700),  # run 590
]

# ----------------
# IMAGE COLOR LIMITS
# ----------------
areal_clim = (0.0, 0.09)

# ----------------
# PROFILE Y-LIMITS
# ----------------
# Reordered to match the new row order.
# In this case they are all identical, but this keeps the logic correct
# if you later use row-specific limits.
vertical_profile_ylims = [
    (0.0, 0.09),  # run 572
    (0.0, 0.09),  # run 582
    (0.0, 0.09),  # run 580
    (0.0, 0.09),  # run 590
]

horizontal_profile_ylims = [
    (0.01, 0.05),  # run 572
    (0.01, 0.05),  # run 582
    (0.01, 0.05),  # run 580
    (0.01, 0.05),  # run 590
]

# ----------------
# TITLES
# ----------------
col_titles = [
    "xRAGE GT Areal Density",
    "xRAGE Sim Recon",
    "Exp Recon",
    "Vertical Profile",
    "Horizontal Profile"
]

# ----------------
# LINE STYLES / COLORS
# ----------------
# Column 1 / GT: solid black
# Column 2 / Simulation: dashed green
# Column 3 / Experiment: dashed orange
profile_styles = {
    "GT": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.2,
        "label": "GT"
    },
    "Sim": {
        "color": "magenta",
        "linestyle": "-",
        "linewidth": 1.2,
        "label": "Sim"
    },
    "Exp": {
        "color": "darkorange",
        "linestyle": "-",
        "linewidth": 1.2,
        "label": "Exp"
    }
}

image_line_styles = [
    profile_styles["GT"],
    profile_styles["Sim"],
    profile_styles["Exp"]
]

# ----------------
# CREATE FIGURE
# ----------------
fig, axs = plt.subplots(
    n_runs, 5,
    figsize=(14.8, 9.5),
    constrained_layout=False,
    gridspec_kw={"width_ratios": [1, 1, 1, 0.82, 0.82]}
)

for r in range(n_runs):
    gt_img = gt_imgs[r]
    sim_img = sim_imgs[r]
    exp_img = exp_imgs[r]

    # ----------------
    # ROW LABEL: (a), (b), (c), ...
    # ----------------
    axs[r, 0].text(
        -0.18, 1.05,
        row_labels[r],
        transform=axs[r, 0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ny, nx = gt_img.shape
    cx, cy = profile_positions[r]

    y_um = (np.arange(ny) - cy) * dy_eff * 1e6
    x_um = (np.arange(nx) - cx) * dx_eff * 1e6

    gt_v = gt_img[:, cx]
    sim_v = sim_img[:, cx]
    exp_v = exp_img[:, cx]

    gt_h = gt_img[cy, :]
    sim_h = sim_img[cy, :]
    exp_h = exp_img[cy, :]

    # ----------------
    # IMAGE PANELS
    # ----------------
    for c in range(3):
        ax = axs[r, c]
        img = [gt_img, sim_img, exp_img][c]
        style = image_line_styles[c]

        im = ax.imshow(
            img,
            cmap="seismic",
            vmin=areal_clim[0],
            vmax=areal_clim[1],
            interpolation="none",
            aspect="equal"
        )

        # Vertical and horizontal profile-location lines now match
        # the corresponding dataset style.
        ax.axvline(
            cx,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0
        )
        ax.axhline(
            cy,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0
        )

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=4)

        if c == 0:
            ax.text(
                -0.08, 0.5,
                time_labels[r],
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

            scalebar = ScaleBar(
                dx=dx_eff * 1e6,
                units="µm",
                fixed_value=25,
                location="upper right",
                height_fraction=0.012,
                width_fraction=0.04,
                color="black",
                font_properties={"size": 8}
            )
            ax.add_artist(scalebar)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.2%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)

        cbar.set_ticks([0.000, 0.043, 0.085])
        cbar.set_ticklabels(["0.000", "0.043", "0.085"])
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(
            "Areal Density [$\mathbf{g/cm^2}$]",
            fontsize=8,
            fontweight="bold"
        )

    # ----------------
    # VERTICAL PROFILE
    # ----------------
    axv = axs[r, 3]

    axv.plot(
        y_um,
        gt_v,
        color=profile_styles["GT"]["color"],
        linestyle=profile_styles["GT"]["linestyle"],
        linewidth=profile_styles["GT"]["linewidth"],
        label=profile_styles["GT"]["label"]
    )
    axv.plot(
        y_um,
        sim_v,
        color=profile_styles["Sim"]["color"],
        linestyle=profile_styles["Sim"]["linestyle"],
        linewidth=profile_styles["Sim"]["linewidth"],
        label=profile_styles["Sim"]["label"]
    )
    axv.plot(
        y_um,
        exp_v,
        color=profile_styles["Exp"]["color"],
        linestyle=profile_styles["Exp"]["linestyle"],
        linewidth=profile_styles["Exp"]["linewidth"],
        label=profile_styles["Exp"]["label"]
    )

    axv.set_xlim(y_um[0], y_um[-1])
    axv.set_ylim(*vertical_profile_ylims[r])
    axv.margins(x=0)

    axv.set_xlabel("y (µm)", fontsize=8, fontweight="bold", labelpad=4)
    axv.set_ylabel("Areal Density [$\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

    axv.yaxis.tick_right()
    axv.yaxis.set_label_position("right")
    axv.tick_params(axis="y", labelsize=7)
    axv.tick_params(axis="x", labelsize=7, pad=2)

    # Add legend to every vertical profile plot
    axv.legend(fontsize=6.5, frameon=False, loc="best")

    if r == 0:
        axv.set_title(col_titles[3], fontsize=10.5, fontweight="bold")

    # ----------------
    # HORIZONTAL PROFILE
    # ----------------
    axh = axs[r, 4]

    axh.plot(
        x_um,
        gt_h,
        color=profile_styles["GT"]["color"],
        linestyle=profile_styles["GT"]["linestyle"],
        linewidth=profile_styles["GT"]["linewidth"],
        label=profile_styles["GT"]["label"]
    )
    axh.plot(
        x_um,
        sim_h,
        color=profile_styles["Sim"]["color"],
        linestyle=profile_styles["Sim"]["linestyle"],
        linewidth=profile_styles["Sim"]["linewidth"],
        label=profile_styles["Sim"]["label"]
    )
    axh.plot(
        x_um,
        exp_h,
        color=profile_styles["Exp"]["color"],
        linestyle=profile_styles["Exp"]["linestyle"],
        linewidth=profile_styles["Exp"]["linewidth"],
        label=profile_styles["Exp"]["label"]
    )

    axh.set_xlim(x_um[0], x_um[-1])
    axh.set_ylim(*horizontal_profile_ylims[r])
    axh.margins(x=0)

    axh.set_xlabel("x (µm)", fontsize=8, fontweight="bold", labelpad=4)
    axh.set_ylabel("Areal Density [$\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

    axh.yaxis.tick_right()
    axh.yaxis.set_label_position("right")
    axh.tick_params(axis="y", labelsize=7)
    axh.tick_params(axis="x", labelsize=7, pad=2)

    # Add legend to every horizontal profile plot
    axh.legend(fontsize=6.5, frameon=False, loc="best")

    if r == 0:
        axh.set_title(col_titles[4], fontsize=10.5, fontweight="bold")

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.08,
    right=0.94,
    bottom=0.08,
    top=0.95,
    wspace=0.32,
    hspace=0.20
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/4x5_areal_density_profiles_with_lines_labeled_time_ordered.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()