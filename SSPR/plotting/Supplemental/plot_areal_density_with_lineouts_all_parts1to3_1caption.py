import string
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# RUNS / DELAYS
# ----------------
# Original 9-row order:
# ["572", "574", "576", "578", "580", "582", "584", "590", "586"]
#
# Updated order with rows 5 and 6 swapped:
runs = ["572", "574", "576", "578", "582", "580", "584", "590", "586"]
delay_times = [6.82, 7.97, 8.12, 8.96, 10.52, 9.67, 11.72, 13.42, 14.41]

n_runs = len(runs)
row_labels = [f"({c})" for c in string.ascii_lowercase[:n_runs]]
time_labels = [f"Time {i + 1}" for i in range(n_runs)]

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
profile_positions = [
    (gt_imgs[0].shape[1] // 2 - 50,  gt_imgs[0].shape[0] // 2 + 300),  # run 572
    (gt_imgs[1].shape[1] // 2 + 50,  gt_imgs[1].shape[0] // 2 + 350),  # run 574
    (gt_imgs[2].shape[1] // 2 + 50,  gt_imgs[2].shape[0] // 2 + 350),  # run 576
    (gt_imgs[3].shape[1] // 2 - 50,  gt_imgs[3].shape[0] // 2 + 400),  # run 578
    (gt_imgs[4].shape[1] // 2 - 50,  gt_imgs[4].shape[0] // 2 + 300),  # run 582
    (gt_imgs[5].shape[1] // 2 - 100, gt_imgs[5].shape[0] // 2 + 450),  # run 580
    (gt_imgs[6].shape[1] // 2 + 50,  gt_imgs[6].shape[0] // 2 + 500),  # run 584
    (gt_imgs[7].shape[1] // 2 + 50,  gt_imgs[7].shape[0] // 2 + 700),  # run 590
    (gt_imgs[8].shape[1] // 2 + 50,  gt_imgs[8].shape[0] // 2 + 700),  # run 586
]

# ----------------
# DISPLAY SETTINGS
# ----------------
areal_clim = (0.0, 0.09)

vertical_profile_ylims = [(0.0, 0.09)] * n_runs

horizontal_profile_ylims = [
    (0.01, 0.05),    # run 572
    (0.01, 0.05),    # run 574
    (0.01, 0.05),    # run 576
    (0.01, 0.05),    # run 578
    (0.01, 0.05),    # run 582
    (0.01, 0.05),    # run 580
    (0.02, 0.06),    # run 584
    (0.01, 0.05),    # run 590
    (0.035, 0.065),  # run 586
]

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
# FUNCTION TO PLOT ONE 3-ROW CHUNK
# ----------------
def plot_chunk(start_idx, end_idx, part_num):
    chunk_indices = list(range(start_idx, end_idx))
    n_chunk = len(chunk_indices)

    fig, axs = plt.subplots(
        n_chunk, 5,
        figsize=(14.8, 2.3 * n_chunk),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.82, 0.82]}
    )

    if n_chunk == 1:
        axs = np.expand_dims(axs, axis=0)

    for local_r, global_r in enumerate(chunk_indices):
        gt_img = gt_imgs[global_r]
        sim_img = sim_imgs[global_r]
        exp_img = exp_imgs[global_r]

        # Shift row labels farther left to avoid overlap
        axs[local_r, 0].text(
            -0.24, 1.05,
            row_labels[global_r],
            transform=axs[local_r, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

        ny, nx = gt_img.shape
        cx, cy = profile_positions[global_r]

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
            ax = axs[local_r, c]
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

            # Only give titles to part 1
            if part_num == 1 and local_r == 0:
                ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=4)

            if c == 0:
                ax.text(
                    -0.08, 0.5,
                    time_labels[global_r],
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
                "Areal Density [$\\mathbf{g/cm^2}$]",
                fontsize=8,
                fontweight="bold"
            )

        # ----------------
        # VERTICAL PROFILE
        # ----------------
        axv = axs[local_r, 3]

        axv.plot(
            y_um, gt_v,
            color=profile_styles["GT"]["color"],
            linestyle=profile_styles["GT"]["linestyle"],
            linewidth=profile_styles["GT"]["linewidth"],
            label=profile_styles["GT"]["label"]
        )
        axv.plot(
            y_um, sim_v,
            color=profile_styles["Sim"]["color"],
            linestyle=profile_styles["Sim"]["linestyle"],
            linewidth=profile_styles["Sim"]["linewidth"],
            label=profile_styles["Sim"]["label"]
        )
        axv.plot(
            y_um, exp_v,
            color=profile_styles["Exp"]["color"],
            linestyle=profile_styles["Exp"]["linestyle"],
            linewidth=profile_styles["Exp"]["linewidth"],
            label=profile_styles["Exp"]["label"]
        )

        axv.set_xlim(y_um[0], y_um[-1])
        axv.set_ylim(*vertical_profile_ylims[global_r])
        axv.margins(x=0)

        axv.set_xlabel("y (µm)", fontsize=8, fontweight="bold", labelpad=4)
        axv.set_ylabel("Areal Density [$\\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

        axv.yaxis.tick_right()
        axv.yaxis.set_label_position("right")
        axv.tick_params(axis="y", labelsize=7)
        axv.tick_params(axis="x", labelsize=7, pad=2)
        axv.legend(fontsize=6.5, frameon=False, loc="best")

        # Only give titles to part 1
        if part_num == 1 and local_r == 0:
            axv.set_title(col_titles[3], fontsize=10.5, fontweight="bold")

        # ----------------
        # HORIZONTAL PROFILE
        # ----------------
        axh = axs[local_r, 4]

        axh.plot(
            x_um, gt_h,
            color=profile_styles["GT"]["color"],
            linestyle=profile_styles["GT"]["linestyle"],
            linewidth=profile_styles["GT"]["linewidth"],
            label=profile_styles["GT"]["label"]
        )
        axh.plot(
            x_um, sim_h,
            color=profile_styles["Sim"]["color"],
            linestyle=profile_styles["Sim"]["linestyle"],
            linewidth=profile_styles["Sim"]["linewidth"],
            label=profile_styles["Sim"]["label"]
        )
        axh.plot(
            x_um, exp_h,
            color=profile_styles["Exp"]["color"],
            linestyle=profile_styles["Exp"]["linestyle"],
            linewidth=profile_styles["Exp"]["linewidth"],
            label=profile_styles["Exp"]["label"]
        )

        axh.set_xlim(x_um[0], x_um[-1])
        axh.set_ylim(*horizontal_profile_ylims[global_r])
        axh.margins(x=0)

        axh.set_xlabel("x (µm)", fontsize=8, fontweight="bold", labelpad=4)
        axh.set_ylabel("Areal Density [$\\mathbf{g/cm^2}$]", fontsize=8, fontweight="bold")

        axh.yaxis.tick_right()
        axh.yaxis.set_label_position("right")
        axh.tick_params(axis="y", labelsize=7)
        axh.tick_params(axis="x", labelsize=7, pad=2)
        axh.legend(fontsize=6.5, frameon=False, loc="best")

        # Only give titles to part 1
        if part_num == 1 and local_r == 0:
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
    save_path = f"/Users/danielhodge/Desktop/areal_densities_all_part{part_num}.pdf"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03
    )
    plt.show()
    plt.close(fig)

# ----------------
# CREATE 3 SEPARATE FIGURES
# ----------------
plot_chunk(0, 3, 1)   # rows 1-3
plot_chunk(3, 6, 2)   # rows 4-6
plot_chunk(6, 9, 3)   # rows 7-9