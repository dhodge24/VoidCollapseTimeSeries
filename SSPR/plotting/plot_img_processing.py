import os
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
dir_raw = "/Users/danielhodge/Desktop/runs"
dir_all_runs = "/Users/danielhodge/Desktop/all_runs"

# ----------------
# CONSTANTS
# ----------------
# Same geometry / effective pixel size parameters as your attached baseline
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac
dy_eff = dx_eff

# ----------------
# MANUAL IMAGE RANGES
# ----------------
# First two columns: one (min, max) per row so you can adjust each plot individually
clim_raw_list = [
    (0, 550),   # run 572
    (0, 400),   # run 576
    (0, 650),   # run 580
    (0, 600),   # run 582
    (0, 450),   # run 590
]

clim_reflected_list = [
    (0, 400),   # run 572
    (0, 250),   # run 576
    (0, 500),   # run 580
    (0, 500),   # run 582
    (0, 300),   # run 590
]

# Last two columns: global ranges
clim_pca_reg = (0, 2.0)
clim_deconv  = (0, 2.5)

# ----------------
# COLORMAP / LABELS
# ----------------
cmap_img = "gray"

col_titles = [
    "Raw Intensity",
    "Reflected",
    "PCA + Image Registration",
    "Deconvolution + Inpainting"
]

cbar_labels = [
    "Intensity",
    "Intensity",
    "I/I0",
    "I/I0"
]

# ----------------
# HELPERS
# ----------------
def find_raw_file(run):
    """
    Looks for a file in /Users/danielhodge/Desktop/runs
    with a name starting like:
    Run_{run}_evt_1_Zyla_0
    """
    prefix = f"Run_{run}_evt_1_Zyla_0"
    matches = []

    for fname in os.listdir(dir_raw):
        if fname.startswith(prefix):
            matches.append(fname)

    if len(matches) == 0:
        raise FileNotFoundError(f"No raw file found for run {run} with prefix: {prefix}")
    if len(matches) > 1:
        print(f"Warning: multiple raw matches for run {run}. Using first match: {matches[0]}")

    return os.path.join(dir_raw, matches[0])


def add_scalebar(ax):
    scalebar = ScaleBar(
        dx=dx_eff * 1e6,   # microns per pixel
        units='µm',
        fixed_value=25,
        location='upper right',
        height_fraction=0.018,
        width_fraction=0.035,
        box_alpha=1.0,
        box_color='white',
        color='black',
        font_properties={"size": 8}
    )
    ax.add_artist(scalebar)


def show_img_with_cbar(fig, ax, img, clim, cmap, cbar_label):
    if clim is None:
        im = ax.imshow(
            img,
            cmap=cmap,
            interpolation="none",
            aspect="equal"
        )
    else:
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            interpolation="none",
            aspect="equal"
        )

    ax.set_axis_off()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.025)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(cbar_label, fontsize=8, fontweight="bold")

    return im


# ----------------
# LOAD DATA
# ----------------
raw_imgs = []
reflected_imgs = []
pca_reg_imgs = []
deconv_inpaint_imgs = []

for run in runs:
    raw_path = find_raw_file(run)
    preproc_dir = os.path.join(dir_all_runs, f"run{run}_exp_preprocessed")

    reflected_path = os.path.join(
        preproc_dir,
        f"run{run}_exp_preprocessed_no_Talbot.tiff"
    )
    pca_reg_path = os.path.join(
        preproc_dir,
        f"run{run}_before_inpaint.tiff"
    )
    deconv_path = os.path.join(
        preproc_dir,
        f"run{run}_exp_holos_with_speckle_FFC_extended_decon.tiff"
    )

    raw_img = np.array(imread(raw_path), dtype=np.float32)

    reflected_img = np.array(imread(reflected_path), dtype=np.float32)
    if reflected_img.ndim > 2:
        reflected_img = reflected_img[0]

    pca_reg_img = np.array(imread(pca_reg_path), dtype=np.float32)
    deconv_img = np.array(imread(deconv_path), dtype=np.float32)

    raw_imgs.append(raw_img)
    reflected_imgs.append(reflected_img)
    pca_reg_imgs.append(pca_reg_img)
    deconv_inpaint_imgs.append(deconv_img)

# ----------------
# ORGANIZE IMAGE GRID
# ----------------
images_grid = []
for i in range(len(runs)):
    images_grid.append([
        raw_imgs[i],
        reflected_imgs[i],
        pca_reg_imgs[i],
        deconv_inpaint_imgs[i]
    ])

clims_grid = []
for i in range(len(runs)):
    clims_grid.append([
        clim_raw_list[i],
        clim_reflected_list[i],
        clim_pca_reg,
        clim_deconv
    ])

# ----------------
# CREATE FIGURE
# ----------------
fig, axs = plt.subplots(
    5, 4,
    figsize=(12, 12),
    constrained_layout=False
)

for r in range(5):
    for c in range(4):
        ax = axs[r, c]
        img = images_grid[r][c]
        clim = clims_grid[r][c]

        show_img_with_cbar(
            fig=fig,
            ax=ax,
            img=img,
            clim=clim,
            cmap=cmap_img,
            cbar_label=cbar_labels[c]
        )

        if r == 0:
            ax.set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=6)

        if c == 0:
            ax.text(
                -0.08, 0.5,
                f"X-Ray Delay\n{delay_times[r]} ns",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
                clip_on=False
            )
            add_scalebar(ax)

# ----------------
# SPACING
# ----------------
fig.subplots_adjust(
    left=0.10,
    right=0.96,
    bottom=0.04,
    top=0.965,
    wspace=0.22,
    hspace=0.04
)

# ----------------
# SAVE / SHOW
# ----------------
save_path = "/Users/danielhodge/Desktop/5x4_intensity_processing_comparison.pdf"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)
plt.show()