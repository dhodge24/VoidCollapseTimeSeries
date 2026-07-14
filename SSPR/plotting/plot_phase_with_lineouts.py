import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
# RUNS / DELAYS
# ----------------
runs = ["572", "576", "580", "582", "590"]
delay_times = [6.82, 8.12, 9.67, 10.52, 13.42]

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_data = "time_series_recons_cropped/"

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
gt_phase_imgs = []
sim_phase_imgs = []
exp_phase_imgs = []

for run in runs:
    dir_sim = dir_main + dir_data + f"run{run}_sim/"
    dir_exp = dir_main + dir_data + f"run{run}_exp/"

    gt_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_gt_final.tiff"), dtype=np.float32)
    sim_phase = np.array(imread(dir_sim + f"run{run}_sim_ph_final.tiff"), dtype=np.float32)
    exp_phase = np.array(imread(dir_exp + f"run{run}_exp_ph_final.tiff"), dtype=np.float32)

    gt_phase_imgs.append(gt_phase)
    sim_phase_imgs.append(sim_phase)
    exp_phase_imgs.append(exp_phase)

# ----------------
# MANUAL PROFILE LOCATIONS
# ----------------
profile_positions = [
    (gt_phase_imgs[0].shape[1] // 2 - 50,  gt_phase_imgs[0].shape[0] // 2 + 300),
    (gt_phase_imgs[1].shape[1] // 2 + 50,  gt_phase_imgs[1].shape[0] // 2 + 350),
    (gt_phase_imgs[2].shape[1] // 2 - 100, gt_phase_imgs[2].shape[0] // 2 + 450),
    (gt_phase_imgs[3].shape[1] // 2 - 50,  gt_phase_imgs[3].shape[0] // 2 + 300),
    (gt_phase_imgs[4].shape[1] // 2 + 50,  gt_phase_imgs[4].shape[0] // 2 + 700),
]

# ----------------
# PROFILE LIMITS PER ROW
# ----------------
vertical_profile_ylims = [(-56, -15)] * 5
horizontal_profile_ylims = [
    (-23, -13),
    (-22, -14.5),
    (-26, -15),
    (-25, -16),
    (-26, -14.5),
]

# ----------------
# ORGANIZE IMAGES
# ----------------
images_grid = []
for i in range(len(runs)):
    images_grid.append([
        gt_phase_imgs[i],
        sim_phase_imgs[i],
        exp_phase_imgs[i]
    ])

# ----------------
# IMAGE COLOR LIMITS
# ----------------
phase_gt_clim = (-56, -15)
phase_clim = (-56, -15)

clims_grid = [[phase_gt_clim, phase_clim, phase_clim] for _ in range(len(runs))]

# ----------------
# TITLES
# ----------------
col_titles = [
    "xRAGE GT Phase",
    "xRAGE Sim Recon",
    "Exp Recon",
    "Vertical Profile",
    "Horizontal Profile"
]

# ----------------
# LINE COLORS
# ----------------
vertical_colors = ["black", "magenta", "cyan"]
horizontal_colors = ["blue", "green", "orange"]

# ----------------
# CREATE FIGURE
# ----------------
fig, axs = plt.subplots(
    5, 5,
    figsize=(14.8, 11.8),
    constrained_layout=False,
    gridspec_kw={"width_ratios": [1, 1, 1, 0.82, 0.82]}
)

for r in range(5):
    gt_img = images_grid[r][0]
    sim_img = images_grid[r][1]
    exp_img = images_grid[r][2]

    ny, nx = gt_img.shape
    center_x, center_y = profile_positions[r]

    y_um = (np.arange(ny) - center_y) * dy_eff * 1e6
    x_um = (np.arange(nx) - center_x) * dx_eff * 1e6

    gt_vertical = gt_img[:, center_x]
    sim_vertical = sim_img[:, center_x]
    exp_vertical = exp_img[:, center_x]

    gt_horizontal = gt_img[center_y, :]
    sim_horizontal = sim_img[center_y, :]
    exp_horizontal = exp_img[center_y, :]

    # ----------------
    # IMAGE PANELS
    # ----------------
    for c in range(3):
        ax = axs[r, c]
        img = images_grid[r][c]

        im = ax.imshow(
            img, cmap="RdBu_r",
            vmin=clims_grid[r][c][0],
            vmax=clims_grid[r][c][1],
            interpolation="none",
            aspect="equal"
        )

        ax.axvline(center_x, color=vertical_colors[c], linestyle="--", linewidth=2.0)
        ax.axhline(center_y, color=horizontal_colors[c], linestyle="--", linewidth=2.0)

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=4)

        if c == 0:
            ax.text(-0.08, 0.5, f"X-Ray Delay\n{delay_times[r]} ns",
                    transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold")

            scalebar = ScaleBar(dx=dx_eff * 1e6, units='µm',
                                fixed_value=25, location='upper right',
                                height_fraction=0.012, width_fraction=0.04,
                                color='black', font_properties={"size": 8})
            ax.add_artist(scalebar)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.2%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Phase (rad)", fontsize=8, fontweight="bold")

    # ----------------
    # VERTICAL PROFILE
    # ----------------
    axv = axs[r, 3]
    axv.plot(y_um, gt_vertical, color="black", label="GT")
    axv.plot(y_um, sim_vertical, color="magenta", label="Sim")
    axv.plot(y_um, exp_vertical, color="cyan", label="Exp")

    axv.set_xlim(y_um[0], y_um[-1])
    axv.set_ylim(*vertical_profile_ylims[r])
    axv.set_xlabel("y (µm)", fontsize=8, fontweight="bold")
    axv.set_ylabel("Phase (rad)", fontsize=8, fontweight="bold")

    # MOVE TO RIGHT
    axv.yaxis.tick_right()
    axv.yaxis.set_label_position("right")
    axv.tick_params(axis='y', labelsize=7)
    axv.tick_params(axis='x', labelsize=7)

    axv.set_box_aspect(1)

    if r == 0:
        axv.set_title(col_titles[3], fontsize=10.5, fontweight="bold")
        axv.legend(fontsize=6.5, frameon=False)

    # ----------------
    # HORIZONTAL PROFILE
    # ----------------
    axh = axs[r, 4]
    axh.plot(x_um, gt_horizontal, color="blue", label="GT")
    axh.plot(x_um, sim_horizontal, color="green", label="Sim")
    axh.plot(x_um, exp_horizontal, color="orange", label="Exp")

    axh.set_xlim(x_um[0], x_um[-1])
    axh.set_ylim(*horizontal_profile_ylims[r])
    axh.set_xlabel("x (µm)", fontsize=8, fontweight="bold")
    axh.set_ylabel("Phase (rad)", fontsize=8, fontweight="bold")

    # MOVE TO RIGHT
    axh.yaxis.tick_right()
    axh.yaxis.set_label_position("right")
    axh.tick_params(axis='y', labelsize=7)
    axh.tick_params(axis='x', labelsize=7)

    axh.set_box_aspect(1)

    if r == 0:
        axh.set_title(col_titles[4], fontsize=10.5, fontweight="bold")
        axh.legend(fontsize=6.5, frameon=False)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.08,
    right=0.94,
    bottom=0.08,
    top=0.95,
    wspace=0.32,
    hspace=0.24
)

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    "/Users/danielhodge/Desktop/5x5_phase_profiles_with_lines.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)
plt.show()