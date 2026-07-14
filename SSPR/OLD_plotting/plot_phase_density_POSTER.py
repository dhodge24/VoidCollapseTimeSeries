import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle


run1 = "572"
run2 = "578"
run3 = "582"
run4 = "590"


dir_main = "/Users/danielhodge/Desktop/"
dir_inten_phase = "time_series_recons_cropped/"
dir_proj_elec_dens = "time_series_recon_cropped_proj_elec_dens/"
dir_areal_dens = "time_series_recon_cropped_areal_dens/"

# FFC Imgs/Phase Imgs directories
dir_img1_sim_inten_phase = dir_main + dir_inten_phase + "run" + run1 + "_" + "sim/"
dir_img1_exp_inten_phase = dir_main + dir_inten_phase + "run" + run1 + "_" + "exp/"
dir_img2_sim_inten_phase = dir_main + dir_inten_phase + "run" + run2 + "_" + "sim/"
dir_img2_exp_inten_phase = dir_main + dir_inten_phase + "run" + run2 + "_" + "exp/"
dir_img3_sim_inten_phase = dir_main + dir_inten_phase + "run" + run3 + "_" + "sim/"
dir_img3_exp_inten_phase = dir_main + dir_inten_phase + "run" + run3 + "_" + "exp/"
dir_img4_sim_inten_phase = dir_main + dir_inten_phase + "run" + run4 + "_" + "sim/"
dir_img4_exp_inten_phase = dir_main + dir_inten_phase + "run" + run4 + "_" + "exp/"

# Projected electron density directories
dir_img1_sim_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run1 + "_" + "sim/"
dir_img1_exp_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run1 + "_" + "exp/"
dir_img2_sim_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run2 + "_" + "sim/"
dir_img2_exp_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run2 + "_" + "exp/"
dir_img3_sim_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run3 + "_" + "sim/"
dir_img3_exp_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run3 + "_" + "exp/"
dir_img4_sim_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run4 + "_" + "sim/"
dir_img4_exp_proj_elec_dens  = dir_main + dir_proj_elec_dens + "run" + run4 + "_" + "exp/"

# Areal density directories
dir_img1_sim_areal_density  = dir_main + dir_areal_dens + "run" + run1 + "_" + "sim/"
dir_img1_exp_areal_density  = dir_main + dir_areal_dens + "run" + run1 + "_" + "exp/"
dir_img2_sim_areal_density  = dir_main + dir_areal_dens + "run" + run2 + "_" + "sim/"
dir_img2_exp_areal_density = dir_main + dir_areal_dens + "run" + run2 + "_" + "exp/"
dir_img3_sim_areal_density  = dir_main + dir_areal_dens + "run" + run3 + "_" + "sim/"
dir_img3_exp_areal_density = dir_main + dir_areal_dens + "run" + run3 + "_" + "exp/"
dir_img4_sim_areal_density = dir_main + dir_areal_dens + "run" + run4 + "_" + "sim/"
dir_img4_exp_areal_density  = dir_main + dir_areal_dens + "run" + run4 + "_" + "exp/"


# Run 1
img1_sim_inten = np.array(imread(dir_img1_sim_inten_phase + "run" + run1 + "_sim_I_final.tiff"))
img1_sim_phase = np.array(imread(dir_img1_sim_inten_phase + "run" + run1 + "_sim_ph_final.tiff"))
img1_sim_proj_elec_dens = np.array(imread(dir_img1_sim_proj_elec_dens + "proj_elec_density_run" + run1 + "sim.tiff"))
img1_sim_areal_density = np.array(imread(dir_img1_sim_areal_density + "areal_density_run" + run1 + "sim.tiff"))

img1_exp_inten = np.array(imread(dir_img1_exp_inten_phase + "run" + run1 + "_exp_I_final.tiff"))
img1_exp_phase = np.array(imread(dir_img1_exp_inten_phase + "run" + run1 + "_exp_ph_final.tiff"))
img1_exp_proj_elec_dens = np.array(imread(dir_img1_exp_proj_elec_dens + "proj_elec_density_run" + run1 + "exp.tiff"))
img1_exp_areal_density = np.array(imread(dir_img1_exp_areal_density + "areal_density_run" + run1 + "exp.tiff"))

# Run 2
img2_sim_inten = np.array(imread(dir_img2_sim_inten_phase + "run" + run2 + "_sim_I_final.tiff"))
img2_sim_phase = np.array(imread(dir_img2_sim_inten_phase + "run" + run2 + "_sim_ph_final.tiff"))
img2_sim_proj_elec_dens = np.array(imread(dir_img2_sim_proj_elec_dens + "proj_elec_density_run" + run2 + "sim.tiff"))
img2_sim_areal_density = np.array(imread(dir_img2_sim_areal_density + "areal_density_run" + run2 + "sim.tiff"))

img2_exp_inten = np.array(imread(dir_img2_exp_inten_phase + "run" + run2 + "_exp_I_final.tiff"))
img2_exp_phase = np.array(imread(dir_img2_exp_inten_phase + "run" + run2 + "_exp_ph_final.tiff"))
img2_exp_proj_elec_dens = np.array(imread(dir_img2_exp_proj_elec_dens + "proj_elec_density_run" + run2 + "exp.tiff"))
img2_exp_areal_density = np.array(imread(dir_img2_exp_areal_density + "areal_density_run" + run2 + "exp.tiff"))

# Run 3
img3_sim_inten = np.array(imread(dir_img3_sim_inten_phase + "run" + run3 + "_sim_I_final.tiff"))
img3_sim_phase = np.array(imread(dir_img3_sim_inten_phase + "run" + run3 + "_sim_ph_final.tiff"))
img3_sim_proj_elec_dens = np.array(imread(dir_img3_sim_proj_elec_dens + "proj_elec_density_run" + run3 + "sim.tiff"))
img3_sim_areal_density = np.array(imread(dir_img3_sim_areal_density + "areal_density_run" + run3 + "sim.tiff"))

img3_exp_inten = np.array(imread(dir_img3_exp_inten_phase + "run" + run3 + "_exp_I_final.tiff"))
img3_exp_phase = np.array(imread(dir_img3_exp_inten_phase + "run" + run3 + "_exp_ph_final.tiff"))
img3_exp_proj_elec_dens = np.array(imread(dir_img3_exp_proj_elec_dens + "proj_elec_density_run" + run3 + "exp.tiff"))
img3_exp_areal_density = np.array(imread(dir_img3_exp_areal_density + "areal_density_run" + run3 + "exp.tiff"))

# Run 4
img4_sim_inten = np.array(imread(dir_img4_sim_inten_phase + "run" + run4 + "_sim_I_final.tiff"))
img4_sim_phase = np.array(imread(dir_img4_sim_inten_phase + "run" + run4 + "_sim_ph_final.tiff"))
img4_sim_proj_elec_dens = np.array(imread(dir_img4_sim_proj_elec_dens + "proj_elec_density_run" + run4 + "sim.tiff"))
img4_sim_areal_density = np.array(imread(dir_img4_sim_areal_density + "areal_density_run" + run4 + "sim.tiff"))

img4_exp_inten = np.array(imread(dir_img4_exp_inten_phase + "run" + run4 + "_exp_I_final.tiff"))
img4_exp_phase = np.array(imread(dir_img4_exp_inten_phase + "run" + run4 + "_exp_ph_final.tiff"))
img4_exp_proj_elec_dens = np.array(imread(dir_img4_exp_proj_elec_dens + "proj_elec_density_run" + run4 + "exp.tiff"))
img4_exp_areal_density = np.array(imread(dir_img4_exp_areal_density + "areal_density_run" + run4 + "exp.tiff"))

# ----------------
# Constants
# ----------------
Ny, Nx = (2500, 2500)
E = 18000  # eV
lam = (1240 / E) * 1e-9
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
z_eff = z12 / M
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac
dy_eff = det_pixel_size / M / scale_fac

x0, x1 = 1200, 1350
y0, y1 = 1000, 1150
roi = (x0, x1, y0, y1)

rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                 linewidth=2, edgecolor='white', facecolor='none')


# mask = create_circular_mask(size=ph.shape[0], percentage=mask_percentage, smooth_pixels=smooth_pixels)

# images = [
#     img1_sim_inten, img1_sim_phase, img1_sim_proj_elec_dens, img1_sim_areal_density,
#     img2_sim_inten, img2_sim_phase, img2_sim_proj_elec_dens, img2_sim_areal_density,
#     img3_sim_inten, img3_sim_phase, img3_sim_proj_elec_dens, img3_sim_areal_density,
#     img4_sim_inten, img4_sim_phase, img4_sim_proj_elec_dens, img4_sim_areal_density,
# ]

images = [
    img1_exp_inten, img1_exp_phase, img1_exp_proj_elec_dens, img1_exp_areal_density,
    img2_exp_inten, img2_exp_phase, img2_exp_proj_elec_dens, img2_exp_areal_density,
    img3_exp_inten, img3_exp_phase, img3_exp_proj_elec_dens, img3_exp_areal_density,
    img4_exp_inten, img4_exp_phase, img4_exp_proj_elec_dens, img4_exp_areal_density,
]

clims = [
    (0, 2), (-56, 0), (0, 35), (0, 0.1),
    (0, 2), (-56, 0), (0, 35), (0, 0.1),
    (0, 2), (-56, 0), (0, 35), (0, 0.1),
    (0, 2), (-56, 0), (0, 35), (0, 0.1),
]

titles = [
    "FFC Image", "Phase", "Projected Electron Density", "Areal Density",
    "FFC Image", "Phase", "Projected Electron Density", "Areal Density",
    "FFC Image", "Phase", "Projected Electron Density", "Areal Density",
    "FFC Image", "Phase", "Projected Electron Density", "Areal Density",
]

# ----------------
# CREATE 4x4 FIGURE
# ----------------
fig, axs = plt.subplots(4, 4, figsize=(15, 12))
axs_flat = axs.ravel()

# ----------------
# PLOT + TITLES
# ----------------
# ims = []
# for ax, img, clim, title in zip(axs_flat, images, clims, titles):
#     im = ax.imshow(img, clim=clim, cmap="inferno")
#     ax.set_title(title, size=14, fontweight="bold")
#     ax.axis("off")
#     ims.append(im)

# ----------------
# PLOT + TITLES (keep size identical)
# ----------------
ims = []
for i, (ax, img, clim, title) in enumerate(zip(axs_flat, images, clims, titles)):
    im = ax.imshow(img, clim=clim, cmap="inferno")  # <- no extent
    ax.set_title(title, size=14, fontweight="bold")
    ax.axis("off")
    ims.append(im)

# # ----------------
# # "AXIS LABELS" as text (no axes/ticks/spines)
# # ----------------
# for i, ax in enumerate(axs_flat):
#     col = i % 4
#     if col in (1, 2, 3):  # phase, proj e dens, areal dens
#         ax.text(0.5, -0.08, r"X Position [$\mathbf{\mu}$m]",
#                 transform=ax.transAxes, ha="center", va="top",
#                 fontsize=12, fontweight="bold", color="white")
#         ax.text(-0.10, 0.5, r"Y Position [$\mathbf{\mu}$m]",
#                 transform=ax.transAxes, ha="right", va="center",
#                 rotation=90, fontsize=12, fontweight="bold", color="white")


# ----------------
# SUBPLOT LABELS (a) ... (p)
# ----------------
labels = [f"({chr(ord('a') + i)})" for i in range(16)]
x_pos, y_pos = 0.025, 0.95
for ax, lab in zip(axs_flat, labels):
    ax.text(x_pos, y_pos, lab, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color="cyan",
            va="top", ha="left")

# ----------------
# SCALE BARS (one per axis)
# ----------------
for ax in axs_flat:
    scalebar = ScaleBar(
        dx=dx_eff * 1e6,   # convert pixel size from meters → µm
        units='µm',        # display unit
        fixed_value=25,    # force exactly 25 µm
        location='upper right',
        height_fraction=0.025,
        box_alpha=0,
        color='cyan',
        font_properties={"size": 12}
    )
    ax.add_artist(scalebar)

# ----------------
# COLORBARS (match each subplot height exactly)
# ----------------
from mpl_toolkits.axes_grid1 import make_axes_locatable

for i, (im, ax) in enumerate(zip(ims, axs_flat)):
    col = i % 4

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)  # tweak size/pad

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=9, width=1, length=2, direction="out")

    if col == 1:
        cbar.set_label("Phase (rad)", fontsize=11, fontweight="bold")
    elif col == 2:
        cbar.set_label(
            r"Projected Electron Density ($\mathbf{10^6}$ e$^-$/nm$\mathbf{^2}$)",
            fontsize=8.5, fontweight="bold"
        )
    elif col == 3:
        cbar.set_label(r"Areal Density (g/cm$\mathbf{^2}$)", fontsize=11, fontweight="bold")

fig.tight_layout(h_pad=0.5, w_pad=2.5)
plt.savefig('/Users/danielhodge/Desktop/exp_4x4_plots.pdf', dpi=300, transparent=False)
plt.show()