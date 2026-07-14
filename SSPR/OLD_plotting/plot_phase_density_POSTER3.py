import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# ----------------
# RUNS
# ----------------
run1 = "572"
run2 = "578"
run3 = "582"
run4 = "590"

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_inten_phase = "time_series_recons_cropped/"
dir_proj_elec_dens = "time_series_recon_cropped_proj_elec_dens/"
dir_areal_dens = "time_series_recon_cropped_areal_dens/"
dir_gt = "GT_adjusted_maps/"

# ----------------
# DIRECTORIES
# ----------------
# FFC / phase
dir_img1_sim_inten_phase = dir_main + dir_inten_phase + f"run{run1}_sim/"
dir_img1_exp_inten_phase = dir_main + dir_inten_phase + f"run{run1}_exp/"
dir_img2_sim_inten_phase = dir_main + dir_inten_phase + f"run{run2}_sim/"
dir_img2_exp_inten_phase = dir_main + dir_inten_phase + f"run{run2}_exp/"
dir_img3_sim_inten_phase = dir_main + dir_inten_phase + f"run{run3}_sim/"
dir_img3_exp_inten_phase = dir_main + dir_inten_phase + f"run{run3}_exp/"
dir_img4_sim_inten_phase = dir_main + dir_inten_phase + f"run{run4}_sim/"
dir_img4_exp_inten_phase = dir_main + dir_inten_phase + f"run{run4}_exp/"

# projected electron density
dir_img1_sim_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run1}_sim/"
dir_img1_exp_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run1}_exp/"
dir_img2_sim_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run2}_sim/"
dir_img2_exp_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run2}_exp/"
dir_img3_sim_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run3}_sim/"
dir_img3_exp_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run3}_exp/"
dir_img4_sim_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run4}_sim/"
dir_img4_exp_proj_elec_dens = dir_main + dir_proj_elec_dens + f"run{run4}_exp/"

# areal density
dir_img1_sim_areal_density = dir_main + dir_areal_dens + f"run{run1}_sim/"
dir_img1_exp_areal_density = dir_main + dir_areal_dens + f"run{run1}_exp/"
dir_img2_sim_areal_density = dir_main + dir_areal_dens + f"run{run2}_sim/"
dir_img2_exp_areal_density = dir_main + dir_areal_dens + f"run{run2}_exp/"
dir_img3_sim_areal_density = dir_main + dir_areal_dens + f"run{run3}_sim/"
dir_img3_exp_areal_density = dir_main + dir_areal_dens + f"run{run3}_exp/"
dir_img4_sim_areal_density = dir_main + dir_areal_dens + f"run{run4}_sim/"
dir_img4_exp_areal_density = dir_main + dir_areal_dens + f"run{run4}_exp/"

# areal density GTs
dir_img1_gt_areal_density = dir_main + dir_gt + f"run{run1}_GT_maps/"
dir_img2_gt_areal_density = dir_main + dir_gt + f"run{run2}_GT_maps/"
dir_img3_gt_areal_density = dir_main + dir_gt + f"run{run3}_GT_maps/"
dir_img4_gt_areal_density = dir_main + dir_gt + f"run{run4}_GT_maps/"

# ----------------
# LOAD IMAGES
# ----------------
# Run 1
img1_sim_inten = np.array(imread(dir_img1_sim_inten_phase + f"run{run1}_sim_I_final.tiff"))
img1_sim_phase = np.array(imread(dir_img1_sim_inten_phase + f"run{run1}_sim_ph_final.tiff"))
img1_sim_proj_elec_dens = np.array(imread(dir_img1_sim_proj_elec_dens + f"proj_elec_density_run{run1}sim.tiff"))
img1_sim_areal_density = np.array(imread(dir_img1_sim_areal_density + f"areal_density_run{run1}sim.tiff"))

img1_exp_inten = np.array(imread(dir_img1_exp_inten_phase + f"run{run1}_exp_I_final.tiff"))
img1_exp_phase = np.array(imread(dir_img1_exp_inten_phase + f"run{run1}_exp_ph_final.tiff"))
img1_exp_proj_elec_dens = np.array(imread(dir_img1_exp_proj_elec_dens + f"proj_elec_density_run{run1}exp.tiff"))
img1_exp_areal_density = np.array(imread(dir_img1_exp_areal_density + f"areal_density_run{run1}exp.tiff"))

# Run 2
img2_sim_inten = np.array(imread(dir_img2_sim_inten_phase + f"run{run2}_sim_I_final.tiff"))
img2_sim_phase = np.array(imread(dir_img2_sim_inten_phase + f"run{run2}_sim_ph_final.tiff"))
img2_sim_proj_elec_dens = np.array(imread(dir_img2_sim_proj_elec_dens + f"proj_elec_density_run{run2}sim.tiff"))
img2_sim_areal_density = np.array(imread(dir_img2_sim_areal_density + f"areal_density_run{run2}sim.tiff"))

img2_exp_inten = np.array(imread(dir_img2_exp_inten_phase + f"run{run2}_exp_I_final.tiff"))
img2_exp_phase = np.array(imread(dir_img2_exp_inten_phase + f"run{run2}_exp_ph_final.tiff"))
img2_exp_proj_elec_dens = np.array(imread(dir_img2_exp_proj_elec_dens + f"proj_elec_density_run{run2}exp.tiff"))
img2_exp_areal_density = np.array(imread(dir_img2_exp_areal_density + f"areal_density_run{run2}exp.tiff"))

# Run 3
img3_sim_inten = np.array(imread(dir_img3_sim_inten_phase + f"run{run3}_sim_I_final.tiff"))
img3_sim_phase = np.array(imread(dir_img3_sim_inten_phase + f"run{run3}_sim_ph_final.tiff"))
img3_sim_proj_elec_dens = np.array(imread(dir_img3_sim_proj_elec_dens + f"proj_elec_density_run{run3}sim.tiff"))
img3_sim_areal_density = np.array(imread(dir_img3_sim_areal_density + f"areal_density_run{run3}sim.tiff"))

img3_exp_inten = np.array(imread(dir_img3_exp_inten_phase + f"run{run3}_exp_I_final.tiff"))
img3_exp_phase = np.array(imread(dir_img3_exp_inten_phase + f"run{run3}_exp_ph_final.tiff"))
img3_exp_proj_elec_dens = np.array(imread(dir_img3_exp_proj_elec_dens + f"proj_elec_density_run{run3}exp.tiff"))
img3_exp_areal_density = np.array(imread(dir_img3_exp_areal_density + f"areal_density_run{run3}exp.tiff"))

# Run 4
img4_sim_inten = np.array(imread(dir_img4_sim_inten_phase + f"run{run4}_sim_I_final.tiff"))
img4_sim_phase = np.array(imread(dir_img4_sim_inten_phase + f"run{run4}_sim_ph_final.tiff"))
img4_sim_proj_elec_dens = np.array(imread(dir_img4_sim_proj_elec_dens + f"proj_elec_density_run{run4}sim.tiff"))
img4_sim_areal_density = np.array(imread(dir_img4_sim_areal_density + f"areal_density_run{run4}sim.tiff"))

img4_exp_inten = np.array(imread(dir_img4_exp_inten_phase + f"run{run4}_exp_I_final.tiff"))
img4_exp_phase = np.array(imread(dir_img4_exp_inten_phase + f"run{run4}_exp_ph_final.tiff"))
img4_exp_proj_elec_dens = np.array(imread(dir_img4_exp_proj_elec_dens + f"proj_elec_density_run{run4}exp.tiff"))
img4_exp_areal_density = np.array(imread(dir_img4_exp_areal_density + f"areal_density_run{run4}exp.tiff"))

# GTs
img1_gt_areal_density = np.array(imread(dir_img1_gt_areal_density + "areal_density_total_adjusted.tiff"))
img2_gt_areal_density = np.array(imread(dir_img2_gt_areal_density + "areal_density_total_adjusted.tiff"))
img3_gt_areal_density = np.array(imread(dir_img3_gt_areal_density + "areal_density_total_adjusted.tiff"))
img4_gt_areal_density = np.array(imread(dir_img4_gt_areal_density + "areal_density_total_adjusted.tiff"))

# ----------------
# CONSTANTS
# ----------------
E = 18000  # eV
z01 = 120.41e-3  # Source to sample distance
z12 = 4.668995   # Sample to detector distance
z02 = z01 + z12  # Source to detector distance
M = z02 / z01    # Geometric magnification
scale_fac = 4    # Additional magnification with lens
det_pixel_size = 6.5e-6  # Physical detector pixel size
dx_eff = det_pixel_size / M / scale_fac  # Effective pixel size

# ----------------
# ORGANIZE AS 3 IMAGE ROWS x 8 COLS
# ----------------
images_grid = [
    [
        img1_sim_inten, img1_exp_inten,
        img2_sim_inten, img2_exp_inten,
        img3_sim_inten, img3_exp_inten,
        img4_sim_inten, img4_exp_inten
    ],
    [
        img1_sim_phase, img1_exp_phase,
        img2_sim_phase, img2_exp_phase,
        img3_sim_phase, img3_exp_phase,
        img4_sim_phase, img4_exp_phase
    ],
    [
        img1_sim_areal_density, img1_exp_areal_density,
        img2_sim_areal_density, img2_exp_areal_density,
        img3_sim_areal_density, img3_exp_areal_density,
        img4_sim_areal_density, img4_exp_areal_density
    ],
]

clims_grid = [
    [(0, 2)] * 8,
    [(-56, 0)] * 8,
    [(0, 0.1)] * 8,
]

delay_times = [6.82, 8.12, 9.67, 10.52, 13.42]

col_titles = []
for t in delay_times:
    col_titles.append(f"Sim:\nX-Ray Delay\n{t} ns")
    col_titles.append(f"Exp:\nX-Ray Delay\n{t} ns")

row_labels = [
    "FFC\nImage",
    "Phase",
    "Areal\nDensity",
]

cbar_labels = [
    "FFC\nImage\n($\mathbf{I/I_0}$)",
    "Phase\n(rad)",
    "Areal\nDensity\n(g/cm$\mathbf{^2}$)",
]

# ----------------
# CREATE FIGURE
# ----------------
fig, axs = plt.subplots(4, 8, figsize=(20, 10), constrained_layout=False)

ims = []

# ----------------
# FIRST 3 IMAGE ROWS
# ----------------
for r in range(3):
    row_ims = []
    for c in range(8):
        ax = axs[r, c]

        im = ax.imshow(
            images_grid[r][c],
            cmap="inferno",
            vmin=clims_grid[r][c][0],
            vmax=clims_grid[r][c][1],
            aspect="equal",
            interpolation="none"
        )

        ax.set_axis_off()

        if r == 0:
            ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=6)

        row_ims.append(im)
    ims.append(row_ims)

# ----------------
# ADD BLUE DASHED CENTER LINE TO 3RD ROW IMAGES
# ----------------
for c in range(8):
    ax = axs[2, c]
    img = images_grid[2][c]
    center_col = img.shape[1] // 2

    ax.axvline(
        x=center_col,
        color='blue',
        linestyle='--',
        linewidth=1.3,
        alpha=0.95
    )

# ----------------
# 4TH ROW: VERTICAL LINEOUTS
# ----------------
recon_areal_list = [
    img1_sim_areal_density, img1_exp_areal_density,
    img2_sim_areal_density, img2_exp_areal_density,
    img3_sim_areal_density, img3_exp_areal_density,
    img4_sim_areal_density, img4_exp_areal_density
]

gt_areal_list = [
    img1_gt_areal_density, img1_gt_areal_density,
    img2_gt_areal_density, img2_gt_areal_density,
    img3_gt_areal_density, img3_gt_areal_density,
    img4_gt_areal_density, img4_gt_areal_density
]

for c in range(8):
    ax = axs[3, c]

    recon = recon_areal_list[c]
    gt = gt_areal_list[c]

    recon_line = recon[:, recon.shape[1] // 2]
    gt_line = gt[:, gt.shape[1] // 2]

    pixels_recon = np.arange(len(recon_line))
    pixels_gt = np.arange(len(gt_line))

    ax.plot(pixels_recon, recon_line, color='blue', linewidth=1.6, label="Reconstruction")
    ax.plot(pixels_gt, gt_line, color='black', linewidth=1.4, label="xRAGE")

    ax.legend(
        loc="best",
        fontsize=8,
        frameon=False
    )

    ax.set_xlim(0, max(len(recon_line), len(gt_line)) - 1)
    ax.set_ylim(0, 0.1)
    ax.tick_params(labelsize=8, width=0.7, length=2.5, direction="out")

    if c != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(
            "Areal Density\n(g/cm$\mathbf{^2}$)",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlabel(
        "Pixels",
        fontsize=9,
        fontweight="bold"
    )

    ax.axhline(
        y=0,
        alpha=0
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# ----------------
# ROW LABELS
# only for first 3 rows
# ----------------
for r in range(3):
    axs[r, 0].text(
        -0.18, 0.5,
        row_labels[r],
        transform=axs[r, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
        color="black",
        linespacing=0.9
    )

# ----------------
# SCALE BARS
# only first column of first 3 rows
# ----------------
for r in range(3):
    c = 0
    scalebar = ScaleBar(
        dx=dx_eff * 1e6,
        units='µm',
        fixed_value=25,
        location='upper right',
        height_fraction=0.012,
        width_fraction=0.04,
        box_alpha=1.0,
        pad=0.02,
        border_pad=0.02,
        sep=1.0,
        color='black',
        font_properties={"size": 10.5}
    )
    axs[r, c].add_artist(scalebar)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.115,
    right=0.885,
    top=0.93,
    bottom=0.07,
    wspace=0.18,
    hspace=0.10
)

# ----------------
# ONE SHARED COLORBAR PER IMAGE ROW
# only for first 3 rows
# ----------------
for r in range(3):
    row_pos = axs[r, -1].get_position()

    cbar_height = row_pos.height
    cbar_y = row_pos.y0 + (row_pos.height - cbar_height) / 2

    cax = fig.add_axes([0.892, cbar_y, 0.0065, cbar_height])

    cbar = fig.colorbar(ims[r][-1], cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=7, width=0.7, length=2.2, direction="out")

    cbar.set_label(
        cbar_labels[r],
        fontsize=8.5,
        fontweight="bold",
        rotation=90,
        labelpad=7
    )

# ----------------
# SAVE / SHOW
# ----------------
plt.savefig(
    '/Users/danielhodge/Desktop/exp_sim_4x8_plots.pdf',
    dpi=300,
    transparent=False,
    bbox_inches='tight'
)
plt.show()