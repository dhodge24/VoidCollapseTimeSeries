import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift

run_holo = "572"
type = "exp"

# for run 574, we shifted +25 pixels in y and x
# for run 576 we shifted -54.99983132112288 in x
shift_y = 0
shift_x = 25.000069739608307

# ----------------
# COLOR LIMITS
# ----------------
areal_clim = (0.01, 0.1)

color_letters = "black"
color_scale_bar = "white"

# ----------------
# PATHS
# ----------------
dir_main = "/Users/danielhodge/Desktop/"
dir_recons = "time_series_recons_cropped/"
dir_runs = "run" + run_holo + "_" + type + "/"
dir_gt_maps = "GT_adjusted_maps/"
dir_run_gt_maps = "run" + run_holo + "_GT_maps/"

phase_path = dir_main + dir_recons + dir_runs + "run" + run_holo + "_" + type + "_ph_final.tiff"
areal_density_gt_path = dir_main + dir_gt_maps + dir_run_gt_maps + "areal_density_total_adjusted.tiff"

# ----------------
# LOAD IMAGES
# ----------------
phase = np.array(imread(phase_path), dtype=np.float32)
areal_density_gt = np.array(imread(areal_density_gt_path), dtype=np.float32)

# ----------------
# COMPUTE THE RECONSTRUCTED AREAL DENSITY
# ----------------
Ny, Nx = phase.shape

E = 18000  # eV
lam = (1240 / E) * 1e-9  # meters
z01 = 120.41e-3
z12 = 4.668995
z02 = z01 + z12
M = z02 / z01
scale_fac = 4
det_pixel_size = 6.5e-6
dx_eff = det_pixel_size / M / scale_fac  # meters per pixel

c = 2.9979e8
m_e = 9.1094e-31
eps0 = 8.852e-12
e = 1.6022e-19
N_A = 6.022e23

n_c = ((2 * np.pi * c) / lam) ** 2 * (m_e * eps0) / e**2

# projected electron density in e-/cm^2
m_to_cm = 1e-2
projected_electron_density_recon_cm = -phase * lam * n_c / np.pi * m_to_cm**2

# Assume purely SU-8: C87 H118 O16
A_SU8 = 87 * 12.011 + 118 * 1.0079 + 16 * 16   # g/mol
Z_SU8 = 87 * 6 + 118 * 1 + 16 * 8              # unitless

# areal density in g/cm^2
areal_density_recon = projected_electron_density_recon_cm * A_SU8 / (N_A * Z_SU8)

# ----------------
# MAKE LEFT/RIGHT COMPOSITE
# left half = reconstruction
# right half = xRAGE
# ----------------
def make_half_composite(img, gt):
    comp = np.zeros_like(img)
    mid = img.shape[1] // 2
    comp[:, :mid] = img[:, :mid]
    comp[:, mid:] = gt[:, mid:]
    return comp

areal_density_gt = shift(areal_density_gt, shift=(shift_y, shift_x), order=3, mode='nearest')
areal_comp = make_half_composite(areal_density_recon, areal_density_gt)

# ----------------
# FIGURE
# ----------------
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

im = ax.imshow(areal_comp, cmap="jet", clim=areal_clim)
# ax.set_title(
#     "Areal Density\nLeft: Reconstruction | Right: xRAGE",
#     fontsize=12,
#     fontweight="bold"
# )
# ax.axis("off")

ax.set_title(
    "Areal Density",
    fontsize=16,
    fontweight="bold",
    y=1.10   # raise the title
)

ax.text(0.25, 1.02, "Reconstruction",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold")

ax.text(0.75, 1.02, "xRAGE",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold")

ax.text(0.5, 1.02, "|",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold")

# scalebar
scalebar = ScaleBar(
    dx=dx_eff * 1e6,   # microns per pixel
    units='µm',
    fixed_value=25,
    location='upper right',
    height_fraction=0.025,
    box_alpha=0,
    color=color_scale_bar,
    font_properties={"size": 10}
)
ax.add_artist(scalebar)

# # ----------------
# # SUBPLOT LABEL
# # ----------------
# ax.text(
#     0.03, 0.96, "(a)",
#     transform=ax.transAxes,
#     fontsize=13,
#     fontweight="bold",
#     color=color_letters,
#     va="top",
#     ha="left"
# )

# ----------------
# COLORBAR
# ----------------
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=9)
cbar.set_label(
    r"Areal Density (g/cm$\mathbf{^2}$)",
    fontsize=10,
    fontweight="bold"
)

fig.tight_layout()
plt.savefig(
    f"/Users/danielhodge/Desktop/run{run_holo}_{type}_areal_density_composite.pdf",
    dpi=300,
    transparent=False
)
plt.show()